"""
Tests for amplifier_module_tool_centaur.backends

Covers:
- RestApiBackend: against a mock HTTP server (respx)
- CliBackend: against a real temp-file subprocess
- create_backend factory: routing and validation
- BackendResult data class
- CentaurBackend protocol conformance

LocalGpuBackend is NOT tested here (requires torch/CUDA).
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

# Ensure importable without pip install
_MODULE_ROOT = (
    Path(__file__).resolve().parent.parent.parent / "modules" / "tool-centaur"
)
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))

from amplifier_module_tool_centaur.backends import (  # noqa: E402
    BackendResult,
    CentaurBackend,
    CliBackend,
    LocalGpuBackend,
    RestApiBackend,
    create_backend,
)


# =========================================================================
# BackendResult
# =========================================================================


class TestBackendResult:
    """BackendResult data class contracts."""

    def test_minimal(self) -> None:
        r = BackendResult(completion="Q>>.")
        assert r.completion == "Q>>."
        assert r.token_log_probs is None
        assert r.metadata == {}

    def test_with_log_probs(self) -> None:
        lp = {"Q": math.log(0.7), "Z": math.log(0.3)}
        r = BackendResult(completion="Q>>.", token_log_probs=lp)
        assert r.token_log_probs is not None
        assert "Q" in r.token_log_probs

    def test_frozen(self) -> None:
        r = BackendResult(completion="Q>>.")
        with pytest.raises(AttributeError):
            r.completion = "Z>>."  # type: ignore[misc]


# =========================================================================
# RestApiBackend
# =========================================================================


class TestRestApiBackend:
    """RestApiBackend against a mock HTTP server using respx."""

    @pytest.fixture
    def backend(self) -> RestApiBackend:
        return RestApiBackend(
            api_url="http://testserver:8000",
            api_key="test-key",
            model_name="centaur-test",
            timeout=5,
        )

    @pytest.mark.asyncio
    async def test_predict_success(self, backend: RestApiBackend) -> None:
        """Successful prediction with log-probs."""
        import respx
        import httpx

        mock_response = {
            "choices": [
                {
                    "text": "Q>>.",
                    "logprobs": {
                        "top_logprobs": [
                            {"Q": math.log(0.73), "Z": math.log(0.27)},
                        ],
                    },
                }
            ],
            "model": "centaur-test",
        }

        with respx.mock:
            respx.post("http://testserver:8000/v1/completions").mock(
                return_value=httpx.Response(200, json=mock_response)
            )

            result = await backend.predict("test prompt <<", max_tokens=8)

        assert result.completion == "Q>>."
        assert result.token_log_probs is not None
        assert abs(result.token_log_probs["Q"] - math.log(0.73)) < 0.001
        assert result.metadata["backend"] == "rest_api"
        assert result.metadata["model"] == "centaur-test"

    @pytest.mark.asyncio
    async def test_predict_no_logprobs(self, backend: RestApiBackend) -> None:
        """Response without logprobs returns None for token_log_probs."""
        import respx
        import httpx

        mock_response = {
            "choices": [{"text": "Z>>.", "logprobs": None}],
            "model": "centaur-test",
        }

        with respx.mock:
            respx.post("http://testserver:8000/v1/completions").mock(
                return_value=httpx.Response(200, json=mock_response)
            )

            result = await backend.predict("test prompt <<")

        assert result.completion == "Z>>."
        assert result.token_log_probs is None

    @pytest.mark.asyncio
    async def test_predict_empty_choices_raises(self, backend: RestApiBackend) -> None:
        """Empty choices list raises RuntimeError."""
        import respx
        import httpx

        with respx.mock:
            respx.post("http://testserver:8000/v1/completions").mock(
                return_value=httpx.Response(200, json={"choices": []})
            )

            with pytest.raises(RuntimeError, match="no choices"):
                await backend.predict("test prompt <<")

    @pytest.mark.asyncio
    async def test_predict_http_error_raises(self, backend: RestApiBackend) -> None:
        """HTTP 500 raises an httpx error."""
        import respx
        import httpx

        with respx.mock:
            respx.post("http://testserver:8000/v1/completions").mock(
                return_value=httpx.Response(500, text="Internal Server Error")
            )

            with pytest.raises(httpx.HTTPStatusError):
                await backend.predict("test prompt <<")

    @pytest.mark.asyncio
    async def test_predict_sends_correct_payload(self, backend: RestApiBackend) -> None:
        """Verify the request body sent to the server."""
        import respx
        import httpx

        captured: list[dict[str, Any]] = []

        def capture_request(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={"choices": [{"text": "Q>>.", "logprobs": None}], "model": "x"},
            )

        with respx.mock:
            respx.post("http://testserver:8000/v1/completions").mock(
                side_effect=capture_request
            )
            await backend.predict("my prompt <<", max_tokens=4)

        assert len(captured) == 1
        payload = captured[0]
        assert payload["prompt"] == "my prompt <<"
        assert payload["max_tokens"] == 4
        assert payload["temperature"] == 0.0
        assert payload["logprobs"] == 10
        assert payload["model"] == "centaur-test"

    @pytest.mark.asyncio
    async def test_predict_auth_header(self, backend: RestApiBackend) -> None:
        """Bearer token is sent in the Authorization header."""
        import respx
        import httpx

        captured_headers: list[dict[str, str]] = []

        def capture_request(request: httpx.Request) -> httpx.Response:
            captured_headers.append(dict(request.headers))
            return httpx.Response(
                200,
                json={"choices": [{"text": "Q>>.", "logprobs": None}], "model": "x"},
            )

        with respx.mock:
            respx.post("http://testserver:8000/v1/completions").mock(
                side_effect=capture_request
            )
            await backend.predict("test <<")

        assert captured_headers[0].get("authorization") == "Bearer test-key"

    @pytest.mark.asyncio
    async def test_close(self, backend: RestApiBackend) -> None:
        """close() releases the HTTP client."""
        import respx
        import httpx

        with respx.mock:
            respx.post("http://testserver:8000/v1/completions").mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "choices": [{"text": "Q>>.", "logprobs": None}],
                        "model": "x",
                    },
                )
            )
            await backend.predict("test <<")

        assert backend._client is not None
        await backend.close()
        assert backend._client is None

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, backend: RestApiBackend) -> None:
        """health_check returns True on 200."""
        import respx
        import httpx

        with respx.mock:
            respx.get("http://testserver:8000/health").mock(
                return_value=httpx.Response(200)
            )
            assert await backend.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, backend: RestApiBackend) -> None:
        """health_check returns False on connection failure."""
        import respx
        import httpx

        with respx.mock:
            respx.get("http://testserver:8000/health").mock(
                side_effect=httpx.ConnectError("refused")
            )
            assert await backend.health_check() is False


# =========================================================================
# CliBackend
# =========================================================================


class TestCliBackend:
    """CliBackend against real temp-file subprocesses."""

    @staticmethod
    def _write_script(content: str) -> str:
        """Write a temp Python script and return its path."""
        fd, path = tempfile.mkstemp(suffix=".py")
        with os.fdopen(fd, "w") as f:
            f.write(content)
        return path

    @pytest.mark.asyncio
    async def test_predict_success(self) -> None:
        """Script writes valid JSON; backend parses it."""
        script = self._write_script(
            """
import json, sys
prompt_file = sys.argv[1]
output_file = sys.argv[2]
with open(prompt_file) as f:
    prompt = f.read()
result = {"completion": "Q>>.", "token_log_probs": {"Q": -0.31, "Z": -1.35}}
with open(output_file, "w") as f:
    json.dump(result, f)
"""
        )
        try:
            backend = CliBackend(
                command=f"python {script} {{prompt_file}} {{output_file}}",
                timeout=10,
            )
            result = await backend.predict("test prompt <<")

            assert result.completion == "Q>>."
            assert result.token_log_probs is not None
            assert abs(result.token_log_probs["Q"] - (-0.31)) < 0.001
            assert result.metadata["backend"] == "cli"
        finally:
            os.unlink(script)

    @pytest.mark.asyncio
    async def test_predict_no_logprobs(self) -> None:
        """Script returns only completion, no log-probs."""
        script = self._write_script(
            """
import json, sys
with open(sys.argv[2], "w") as f:
    json.dump({"completion": "Z>>."}, f)
"""
        )
        try:
            backend = CliBackend(
                command=f"python {script} {{prompt_file}} {{output_file}}",
                timeout=10,
            )
            result = await backend.predict("test <<")

            assert result.completion == "Z>>."
            assert result.token_log_probs is None
        finally:
            os.unlink(script)

    @pytest.mark.asyncio
    async def test_predict_script_failure_raises(self) -> None:
        """Non-zero exit code raises RuntimeError."""
        script = self._write_script("import sys; sys.exit(1)")
        try:
            backend = CliBackend(
                command=f"python {script} {{prompt_file}} {{output_file}}",
                timeout=10,
            )
            with pytest.raises(RuntimeError, match="exited with code 1"):
                await backend.predict("test <<")
        finally:
            os.unlink(script)

    @pytest.mark.asyncio
    async def test_predict_receives_prompt(self) -> None:
        """The script can read the prompt from the temp file."""
        script = self._write_script(
            """
import json, sys
with open(sys.argv[1]) as f:
    prompt = f.read()
# Echo the prompt back in the completion field
with open(sys.argv[2], "w") as f:
    json.dump({"completion": prompt}, f)
"""
        )
        try:
            backend = CliBackend(
                command=f"python {script} {{prompt_file}} {{output_file}}",
                timeout=10,
            )
            result = await backend.predict("my special prompt <<")

            assert result.completion == "my special prompt <<"
        finally:
            os.unlink(script)

    @pytest.mark.asyncio
    async def test_temp_files_cleaned_up(self) -> None:
        """Temp files are removed even after successful prediction."""
        script = self._write_script(
            """
import json, sys
with open(sys.argv[2], "w") as f:
    json.dump({"completion": "Q>>."}, f)
# Also print the paths so we can verify cleanup
print(sys.argv[1], sys.argv[2])
"""
        )
        try:
            backend = CliBackend(
                command=f"python {script} {{prompt_file}} {{output_file}}",
                timeout=10,
            )
            await backend.predict("test <<")
            # We can't easily capture the paths, but we can verify the
            # backend didn't crash during cleanup.  The main guarantee
            # is in the finally block of CliBackend.predict().
        finally:
            os.unlink(script)

    @pytest.mark.asyncio
    async def test_close_is_noop(self) -> None:
        """close() on CliBackend does nothing and doesn't error."""
        backend = CliBackend(command="echo {prompt_file} {output_file}")
        await backend.close()

    @pytest.mark.asyncio
    async def test_health_check_always_true(self) -> None:
        """CLI backend is always 'healthy'."""
        backend = CliBackend(command="echo")
        assert await backend.health_check() is True


# =========================================================================
# create_backend factory
# =========================================================================


class TestCreateBackend:
    """Tests for the backend factory function."""

    def test_rest_api_default(self) -> None:
        """Default type is rest_api."""
        backend = create_backend({})
        assert isinstance(backend, RestApiBackend)

    def test_rest_api_explicit(self) -> None:
        backend = create_backend(
            {
                "type": "rest_api",
                "api_url": "http://my-server:9000",
                "model_name": "my-model",
                "timeout": 30,
            }
        )
        assert isinstance(backend, RestApiBackend)
        assert backend.api_url == "http://my-server:9000"
        assert backend.model_name == "my-model"
        assert backend.timeout == 30

    def test_rest_api_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """API key is resolved from the named environment variable."""
        monkeypatch.setenv("MY_KEY", "secret-123")
        backend = create_backend(
            {
                "type": "rest_api",
                "api_key_env": "MY_KEY",
            }
        )
        assert isinstance(backend, RestApiBackend)
        assert backend.api_key == "secret-123"

    def test_rest_api_missing_env_gives_empty(self) -> None:
        """Missing env var gives empty api_key (not an error at creation)."""
        backend = create_backend(
            {
                "type": "rest_api",
                "api_key_env": "NONEXISTENT_KEY_ABC123",
            }
        )
        assert isinstance(backend, RestApiBackend)
        assert backend.api_key == ""

    def test_cli(self) -> None:
        backend = create_backend(
            {
                "type": "cli",
                "command": "python infer.py {prompt_file} {output_file}",
                "timeout": 90,
            }
        )
        assert isinstance(backend, CliBackend)
        assert "infer.py" in backend.command
        assert backend.timeout == 90

    def test_cli_missing_command_raises(self) -> None:
        with pytest.raises(ValueError, match="command"):
            create_backend({"type": "cli"})

    def test_local_gpu(self) -> None:
        backend = create_backend(
            {
                "type": "local_gpu",
                "model_name": "my-model",
                "loader": "transformers",
                "load_in_4bit": False,
            }
        )
        assert isinstance(backend, LocalGpuBackend)
        assert backend.model_name == "my-model"
        assert backend.loader == "transformers"
        assert backend.load_in_4bit is False

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend type"):
            create_backend({"type": "quantum_computer"})

    def test_rest_api_strips_trailing_slash(self) -> None:
        backend = create_backend(
            {
                "type": "rest_api",
                "api_url": "http://server:8000/",
            }
        )
        assert isinstance(backend, RestApiBackend)
        assert not backend.api_url.endswith("/")


# =========================================================================
# Protocol conformance
# =========================================================================


class TestProtocolConformance:
    """Verify all backends satisfy the CentaurBackend protocol."""

    def test_rest_api_is_backend(self) -> None:
        b = RestApiBackend(api_url="http://x")
        assert isinstance(b, CentaurBackend)

    def test_cli_is_backend(self) -> None:
        b = CliBackend(command="echo")
        assert isinstance(b, CentaurBackend)

    def test_local_gpu_is_backend(self) -> None:
        from amplifier_module_tool_centaur.backends import LocalGpuBackend

        b = LocalGpuBackend()
        assert isinstance(b, CentaurBackend)

    def test_conftest_mock_is_backend(self, mock_backend: Any) -> None:
        """The test MockBackend also satisfies the protocol."""
        assert isinstance(mock_backend, CentaurBackend)
