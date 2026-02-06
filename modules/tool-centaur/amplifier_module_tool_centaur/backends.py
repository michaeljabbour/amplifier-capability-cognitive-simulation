"""
Centaur inference backends.

Each backend implements one way to run the Centaur model.  The tool module
selects a backend at mount-time based on configuration.

Architecture
------------
``CentaurBackend`` is a :pep:`544` structural protocol.  Any object that
provides ``predict``, ``close``, and ``health_check`` with the right
signatures is a valid backend -- no inheritance required.

``create_backend`` is the factory that maps a config dict to a concrete
backend instance.

Backend types
~~~~~~~~~~~~~
===========  ===============  ============================================
type         Class            When to use
===========  ===============  ============================================
rest_api     RestApiBackend   vLLM, TGI, or any OpenAI-compatible endpoint
cli          CliBackend       Existing inference scripts
local_gpu    LocalGpuBackend  Direct GPU access (80-160 GB VRAM)
===========  ===============  ============================================
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared data class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackendResult:
    """Raw result returned by every backend.

    Attributes
    ----------
    completion:
        The generated text (everything the model produced after the prompt).
    token_log_probs:
        Optional mapping ``{token_str: log_probability}`` for the **first
        generated token** (the choice token).  ``None`` when the backend
        does not support log-probs.  Values are natural-log probabilities
        (negative floats).
    metadata:
        Arbitrary backend-specific info (model name, latency, etc.).
    """

    completion: str
    token_log_probs: dict[str, float] | None = None
    metadata: dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class CentaurBackend(Protocol):
    """Structural protocol that every backend must satisfy.

    Implementations do **not** need to inherit from this class -- they only
    need to provide methods with matching signatures.
    """

    async def predict(self, prompt: str, max_tokens: int = 16) -> BackendResult:
        """Run inference on a formatted Centaur prompt.

        Parameters
        ----------
        prompt:
            Complete prompt string (already formatted with ``<<``/``>>``
            markers by :mod:`prompt`).
        max_tokens:
            Maximum new tokens to generate.  Centaur predictions are
            typically 1-3 tokens; 16 is a safe upper bound.

        Returns
        -------
        BackendResult
            The model's raw completion plus optional log-probs.
        """
        ...

    async def close(self) -> None:
        """Release resources (HTTP clients, GPU memory, temp files)."""
        ...

    async def health_check(self) -> bool:
        """Return ``True`` if the backend is ready to serve predictions."""
        ...


# =========================================================================
# Backend: Remote REST API
# =========================================================================


class RestApiBackend:
    """Backend for remote model servers exposing an OpenAI-compatible API.

    Works with vLLM, TGI (``--api-type openai``), and any server that
    implements ``POST /v1/completions``.

    Configuration
    -------------
    api_url : str
        Base URL of the inference server (e.g. ``http://gpu-box:8000``).
    api_key : str
        Bearer token.  Resolved from an environment variable at mount-time;
        never stored in YAML.
    model_name : str
        Model identifier sent in the request body.  Leave empty to let the
        server pick its default.
    timeout : int
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        api_url: str,
        api_key: str = "",
        model_name: str = "",
        timeout: int = 60,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        self._client: object | None = None  # httpx.AsyncClient (lazy)

    # -- helpers ----------------------------------------------------------

    async def _get_client(self) -> object:
        """Lazy-init an ``httpx.AsyncClient``."""
        if self._client is None:
            import httpx

            headers: dict[str, str] = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers=headers,
            )
        return self._client

    # -- protocol ---------------------------------------------------------

    async def predict(self, prompt: str, max_tokens: int = 16) -> BackendResult:
        import httpx

        client: httpx.AsyncClient = await self._get_client()  # type: ignore[assignment]

        payload: dict[str, object] = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "logprobs": 10,
        }
        if self.model_name:
            payload["model"] = self.model_name

        response = await client.post(
            f"{self.api_url}/v1/completions",
            json=payload,
        )
        response.raise_for_status()
        data: dict[str, object] = response.json()

        choices = data.get("choices", [])
        if not isinstance(choices, list) or len(choices) == 0:
            raise RuntimeError(f"Unexpected API response: no choices in {data!r}")

        choice: dict[str, object] = choices[0]  # type: ignore[assignment]
        completion = str(choice.get("text", ""))

        # Parse log-probs when the server provides them.
        token_log_probs: dict[str, float] | None = None
        logprobs_obj = choice.get("logprobs")
        if isinstance(logprobs_obj, dict):
            top_logprobs_list = logprobs_obj.get("top_logprobs")
            if isinstance(top_logprobs_list, list) and len(top_logprobs_list) > 0:
                first_token_probs = top_logprobs_list[0]
                if isinstance(first_token_probs, dict):
                    token_log_probs = {
                        str(tok): float(lp) for tok, lp in first_token_probs.items()
                    }

        return BackendResult(
            completion=completion,
            token_log_probs=token_log_probs,
            metadata={
                "backend": "rest_api",
                "model": str(data.get("model", self.model_name)),
            },
        )

    async def close(self) -> None:
        if self._client is not None:
            import httpx

            assert isinstance(self._client, httpx.AsyncClient)
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> bool:
        try:
            import httpx

            client: httpx.AsyncClient = await self._get_client()  # type: ignore[assignment]
            r = await client.get(f"{self.api_url}/health")
            return r.status_code == 200
        except Exception:
            return False


# =========================================================================
# Backend: CLI subprocess
# =========================================================================


class CliBackend:
    """Backend that delegates to an external inference script.

    The command template receives three placeholders:

    - ``{prompt_file}``  -- path to a temp file containing the prompt text
    - ``{output_file}``  -- path where the script must write a JSON result
    - ``{max_tokens}``   -- integer max-tokens value

    The JSON written to *output_file* must have the shape::

        {
            "completion": "Q>>",
            "token_log_probs": {"Q": -0.31, "Z": -1.35}   // optional
        }

    Configuration
    -------------
    command : str
        Shell command template.
    timeout : int
        Subprocess timeout in seconds.
    """

    def __init__(self, command: str, timeout: int = 120) -> None:
        self.command = command
        self.timeout = timeout

    async def predict(self, prompt: str, max_tokens: int = 16) -> BackendResult:
        import asyncio
        import json
        import tempfile

        # Write prompt to a temp file the script can read.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as pf:
            pf.write(prompt)
            prompt_file = pf.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as of:
            output_file = of.name

        cmd = self.command.format(
            prompt_file=prompt_file,
            output_file=output_file,
            max_tokens=max_tokens,
        )

        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout
            )

            if proc.returncode != 0:
                raise RuntimeError(
                    f"CLI backend exited with code {proc.returncode}: "
                    f"{stderr.decode(errors='replace')}"
                )

            with open(output_file) as f:
                result: dict[str, object] = json.load(f)

            completion = str(result.get("completion", ""))

            raw_lp = result.get("token_log_probs")
            token_log_probs: dict[str, float] | None = None
            if isinstance(raw_lp, dict):
                token_log_probs = {str(k): float(v) for k, v in raw_lp.items()}

            return BackendResult(
                completion=completion,
                token_log_probs=token_log_probs,
                metadata={"backend": "cli", "command": cmd.split()[0]},
            )
        finally:
            # Clean up temp files regardless of success/failure.
            for path in (prompt_file, output_file):
                try:
                    os.unlink(path)
                except OSError:
                    pass

    async def close(self) -> None:
        pass  # No persistent resources.

    async def health_check(self) -> bool:
        return True  # CLI is "available" if the system can run commands.


# =========================================================================
# Backend: Local GPU
# =========================================================================


class LocalGpuBackend:
    """Backend for direct GPU inference via ``unsloth`` or ``transformers``.

    .. warning::

       This backend has **heavy** dependencies (``torch``, ``transformers``,
       optionally ``unsloth``).  These are **not** declared in
       ``pyproject.toml`` -- they must be pre-installed in the runtime
       environment.

    Configuration
    -------------
    model_name : str
        HuggingFace model ID, e.g.
        ``"marcelbinz/Llama-3.1-Centaur-70B-adapter"``.
    loader : str
        ``"unsloth"`` (80 GB, recommended) or ``"transformers"`` (160 GB+).
    load_in_4bit : bool
        Quantise the model (``unsloth`` only).
    max_seq_length : int
        Maximum prompt length in tokens.
    device_map : str
        ``transformers`` device-map strategy.
    """

    def __init__(
        self,
        model_name: str = "marcelbinz/Llama-3.1-Centaur-70B-adapter",
        loader: str = "unsloth",
        load_in_4bit: bool = True,
        max_seq_length: int = 32768,
        device_map: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.loader = loader
        self.load_in_4bit = load_in_4bit
        self.max_seq_length = max_seq_length
        self.device_map = device_map
        self._model: object | None = None
        self._tokenizer: object | None = None

    # -- model loading ----------------------------------------------------

    def _load_model_sync(self) -> None:
        """Synchronous model loading (runs in an executor)."""
        if self.loader == "unsloth":
            from unsloth import FastLanguageModel  # type: ignore[import-untyped]

            self._model, self._tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=self.max_seq_length,
                dtype=None,
                load_in_4bit=self.load_in_4bit,
            )
            FastLanguageModel.for_inference(self._model)
        else:
            import torch  # type: ignore[import-not-found]  # env-specific
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-untyped,import-not-found]

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device_map,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        logger.info("Loaded Centaur: %s via %s", self.model_name, self.loader)

    async def _ensure_model(self) -> None:
        """Lazy-load the model on first call."""
        if self._model is not None:
            return
        import asyncio

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_model_sync)

    # -- inference --------------------------------------------------------

    def _run_inference_sync(
        self, prompt: str, max_tokens: int
    ) -> tuple[str, dict[str, float] | None]:
        """Synchronous inference (runs in an executor)."""
        import torch  # type: ignore[import-not-found]  # env-specific

        tokenizer = self._tokenizer
        model = self._model
        assert tokenizer is not None and model is not None

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)  # type: ignore[union-attr]
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(  # type: ignore[union-attr]
                **inputs,
                max_new_tokens=max_tokens,
                temperature=1e-7,  # Near-deterministic
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )

        new_tokens = outputs.sequences[0][input_len:]
        completion: str = tokenizer.decode(new_tokens, skip_special_tokens=True)  # type: ignore[union-attr]

        # Extract first-token log-probs.
        token_log_probs: dict[str, float] | None = None
        if outputs.scores:
            first_logits = outputs.scores[0][0]
            log_probs = torch.log_softmax(first_logits, dim=-1)
            top_k = torch.topk(log_probs, k=min(10, log_probs.shape[-1]))
            token_log_probs = {
                tokenizer.decode([idx.item()]): lp.item()  # type: ignore[union-attr]
                for idx, lp in zip(top_k.indices, top_k.values)
            }

        return completion, token_log_probs

    # -- protocol ---------------------------------------------------------

    async def predict(self, prompt: str, max_tokens: int = 16) -> BackendResult:
        import asyncio

        await self._ensure_model()
        loop = asyncio.get_running_loop()
        completion, token_log_probs = await loop.run_in_executor(
            None, self._run_inference_sync, prompt, max_tokens
        )
        return BackendResult(
            completion=completion,
            token_log_probs=token_log_probs,
            metadata={
                "backend": "local_gpu",
                "model": self.model_name,
                "loader": self.loader,
            },
        )

    async def close(self) -> None:
        self._model = None
        self._tokenizer = None
        # Python GC + CUDA will reclaim GPU memory.

    async def health_check(self) -> bool:
        try:
            import torch  # type: ignore[import-not-found]  # env-specific

            return torch.cuda.is_available()
        except ImportError:
            return False


# =========================================================================
# Factory
# =========================================================================


def create_backend(config: dict[str, object]) -> CentaurBackend:
    """Instantiate a backend from a configuration dict.

    Parameters
    ----------
    config:
        Must contain a ``"type"`` key (``"rest_api"``, ``"cli"``, or
        ``"local_gpu"``).  Remaining keys are forwarded to the backend
        constructor.

    Returns
    -------
    CentaurBackend
        A ready-to-use backend instance.

    Raises
    ------
    ValueError
        If ``type`` is missing or unrecognised.
    """
    backend_type = str(config.get("type", "rest_api"))

    if backend_type == "rest_api":
        api_key = os.environ.get(str(config.get("api_key_env", "CENTAUR_API_KEY")), "")
        backend: CentaurBackend = RestApiBackend(
            api_url=str(config.get("api_url", "http://localhost:8000")),
            api_key=api_key,
            model_name=str(config.get("model_name", "")),
            timeout=int(config.get("timeout", 60)),  # type: ignore[arg-type]
        )
        return backend

    if backend_type == "cli":
        cmd = config.get("command")
        if not cmd:
            raise ValueError("CLI backend requires a 'command' in config")
        return CliBackend(
            command=str(cmd),
            timeout=int(config.get("timeout", 120)),  # type: ignore[arg-type]
        )

    if backend_type == "local_gpu":
        return LocalGpuBackend(
            model_name=str(
                config.get("model_name", "marcelbinz/Llama-3.1-Centaur-70B-adapter")
            ),
            loader=str(config.get("loader", "unsloth")),
            load_in_4bit=bool(config.get("load_in_4bit", True)),
            max_seq_length=int(config.get("max_seq_length", 32768)),  # type: ignore[arg-type]
            device_map=str(config.get("device_map", "auto")),
        )

    raise ValueError(
        f"Unknown backend type: {backend_type!r}.  "
        f"Expected 'rest_api', 'cli', or 'local_gpu'."
    )
