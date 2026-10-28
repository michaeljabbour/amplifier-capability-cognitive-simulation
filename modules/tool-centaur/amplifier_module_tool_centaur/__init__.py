"""
Centaur human cognition prediction tool module.

Registers the ``centaur_predict`` tool with Amplifier.  The tool accepts
either structured experiment descriptions or raw Psych-101 prompts, routes
them through a configurable backend, and returns structured predictions
with probabilities.

Mount-time configuration
------------------------
backend : dict
    Backend selection and options.  Must include a ``"type"`` key
    (``"rest_api"``, ``"cli"``, or ``"local_gpu"``).  See
    :mod:`backends` for per-type options.
max_tokens : int
    Default maximum tokens to generate per prediction (default: 16).

Example behaviour YAML::

    tools:
      - module: tool-centaur
        config:
          backend:
            type: rest_api
            api_url: http://localhost:8000
            api_key_env: CENTAUR_API_KEY
          max_tokens: 16
"""

from __future__ import annotations

__amplifier_module_type__ = "tool"

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input / output JSON schemas (constants)
# ---------------------------------------------------------------------------

TOOL_INPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "experiment_description": {
            "type": "string",
            "description": (
                "Natural-language instructions a participant would read at "
                "the start of the experiment.  Required when using structured "
                "mode (omit when providing raw_prompt)."
            ),
        },
        "prior_trials": {
            "type": "array",
            "description": (
                "Completed trials providing learning context.  Each object "
                "has 'stimulus' (str, required), 'response' (str, required), "
                "and 'feedback' (str, optional)."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "stimulus": {
                        "type": "string",
                        "description": "What the participant observed.",
                    },
                    "response": {
                        "type": "string",
                        "description": "The participant's choice token.",
                    },
                    "feedback": {
                        "type": "string",
                        "description": "Feedback shown after the choice (optional).",
                    },
                },
                "required": ["stimulus", "response"],
            },
        },
        "pending_trial": {
            "type": "string",
            "description": (
                "Stimulus for the trial to predict (without the response).  "
                "Required in structured mode."
            ),
        },
        "raw_prompt": {
            "type": "string",
            "description": (
                "A pre-formatted Psych-101 prompt containing << >> markers.  "
                "Use INSTEAD of the structured fields above."
            ),
        },
        "max_tokens": {
            "type": "integer",
            "description": "Override the default max tokens to generate.",
        },
    },
    "required": [],
}


# ---------------------------------------------------------------------------
# Mount entry point
# ---------------------------------------------------------------------------


async def mount(
    coordinator: Any,
    config: dict[str, Any] | None = None,
) -> Any:
    """Amplifier module entry point.

    Called by the kernel's module loader.  Creates a backend from *config*,
    wraps it in a :class:`CentaurPredictTool`, and registers the tool.

    Parameters
    ----------
    coordinator:
        The ``ModuleCoordinator`` provided by the Amplifier runtime.
        Used to register the tool via ``coordinator.mount("tools", ...)``.
    config:
        Merged configuration from the behaviour YAML and the user's
        ``settings.yaml``.

    Returns
    -------
    Callable[[], Awaitable[None]]
        An async cleanup function that releases backend resources.
    """
    from .backends import create_backend

    config = config or {}

    backend_config: dict[str, object] = config.get("backend", {"type": "rest_api"})  # type: ignore[assignment]
    max_tokens: int = int(config.get("max_tokens", 16))

    backend = create_backend(backend_config)
    tool = CentaurPredictTool(backend=backend, default_max_tokens=max_tokens)

    await coordinator.mount("tools", tool, name=tool.name)
    backend_type = backend_config.get("type", "rest_api")
    logger.info("Mounted %s (backend=%s)", tool.name, backend_type)

    async def cleanup() -> None:
        await backend.close()

    return cleanup


# ---------------------------------------------------------------------------
# Tool class
# ---------------------------------------------------------------------------


class CentaurPredictTool:
    """Amplifier tool that predicts human behaviour via the Centaur model.

    Implements the Amplifier **Tool protocol** (structural — no base class):

    =========  ==============  ==========================================
    Member     Type            Purpose
    =========  ==============  ==========================================
    name       ``str``         Unique tool identifier
    description ``str``        Shown to the LLM for tool selection
    input_schema ``dict``      JSON Schema for the tool's parameters
    execute    async method    Run inference and return a ``ToolResult``
    =========  ==============  ==========================================

    Parameters
    ----------
    backend:
        A :class:`~backends.CentaurBackend` instance.
    default_max_tokens:
        Fallback ``max_tokens`` when the caller does not specify one.
    """

    name: str = "centaur_predict"

    def __init__(self, backend: Any, default_max_tokens: int = 16) -> None:
        self._backend = backend
        self._default_max_tokens = default_max_tokens

    @property
    def description(self) -> str:
        return (
            "Predict human behaviour in psychological experiments using the "
            "Centaur foundation model (Llama-3.1-70B fine-tuned on 60,000+ "
            "participants across 160 experiments from the Psych-101 dataset).\n"
            "\n"
            "Two input modes:\n"
            "  structured -- provide experiment_description + prior_trials + "
            "pending_trial\n"
            "  raw        -- provide a pre-formatted Psych-101 prompt with "
            "<< >> choice markers\n"
            "\n"
            "Returns the predicted human choice, its probability, and "
            "alternative choices with their probabilities."
        )

    @property
    def input_schema(self) -> dict[str, object]:
        return TOOL_INPUT_SCHEMA

    async def execute(self, input: dict[str, Any]) -> Any:
        """Run a Centaur prediction.

        Parameters
        ----------
        input:
            Tool parameters matching :data:`TOOL_INPUT_SCHEMA`.

        Returns
        -------
        ToolResult
            On success, ``output`` contains::

                {
                    "predicted_choice": "Q",
                    "probability": 0.73,
                    "alternatives": {"Z": 0.27},
                    "raw_output": "Q>>.",
                    "prompt_used": "...",
                    "backend": "rest_api"
                }

            On failure, ``error`` contains ``{"message": "..."}``
        """
        from .backends import BackendResult
        from .prompt import (
            CentaurPrediction,
            build_prompt,
            build_prompt_raw,
            parse_completion,
        )

        # Late import: ToolResult comes from the host runtime.
        try:
            from amplifier_core import ToolResult  # type: ignore[import-not-found]
        except ImportError:
            # Fallback for testing without amplifier_core installed.
            ToolResult = _FallbackToolResult  # type: ignore[assignment,misc]

        try:
            # ---- build prompt ------------------------------------------------
            if "raw_prompt" in input and input["raw_prompt"]:
                prompt: str = build_prompt_raw(input["raw_prompt"])
            else:
                prompt = build_prompt(
                    experiment_description=input.get("experiment_description", ""),
                    prior_trials=input.get("prior_trials"),
                    pending_trial=input.get("pending_trial"),
                )

            max_tokens: int = int(input.get("max_tokens", self._default_max_tokens))

            # ---- run inference -----------------------------------------------
            result: BackendResult = await self._backend.predict(
                prompt, max_tokens=max_tokens
            )

            # ---- parse result ------------------------------------------------
            predictions: list[CentaurPrediction] = parse_completion(
                completion_text=result.completion,
                token_log_probs=result.token_log_probs,
            )

            top = predictions[0] if predictions else None

            return ToolResult(
                success=True,
                output={
                    "predicted_choice": top.choice if top else None,
                    "probability": round(top.probability, 4) if top and top.probability is not None else None,
                    "alternatives": (
                        {k: round(v, 4) for k, v in top.alternatives.items()}
                        if top
                        else {}
                    ),
                    "raw_output": result.completion,
                    "prompt_used": prompt,
                    "backend": result.metadata.get("backend", "unknown"),
                },
            )

        except Exception as e:
            logger.error("Centaur prediction failed: %s", e, exc_info=True)
            return ToolResult(
                success=False,
                error={"message": f"Prediction failed: {e}"},
            )


# ---------------------------------------------------------------------------
# Fallback for testing without amplifier_core
# ---------------------------------------------------------------------------


class _FallbackToolResult:
    """Minimal stand-in when ``amplifier_core.ToolResult`` is unavailable."""

    def __init__(
        self,
        success: bool,
        output: dict[str, object] | None = None,
        error: dict[str, object] | None = None,
    ) -> None:
        self.success = success
        self.output = output
        self.error = error

    def __repr__(self) -> str:
        if self.success:
            return f"ToolResult(success=True, output={self.output!r})"
        return f"ToolResult(success=False, error={self.error!r})"
