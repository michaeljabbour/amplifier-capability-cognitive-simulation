"""
Tests for amplifier_module_tool_centaur (tool integration)

Covers:
- CentaurPredictTool.execute: structured input, raw input, error handling
- mount(): coordinator registration and cleanup
- End-to-end flow: input → prompt → mock backend → parsed output
- TOOL_INPUT_SCHEMA structure
- _FallbackToolResult when amplifier_core is absent
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

# Ensure importable without pip install
_MODULE_ROOT = (
    Path(__file__).resolve().parent.parent.parent / "modules" / "tool-centaur"
)
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))

from amplifier_module_tool_centaur import (  # noqa: E402
    TOOL_INPUT_SCHEMA,
    CentaurPredictTool,
    _FallbackToolResult,
    mount,
)
from amplifier_module_tool_centaur.prompt import CHOICE_OPEN  # noqa: E402


# =========================================================================
# CentaurPredictTool.execute — structured mode
# =========================================================================


class TestToolStructuredMode:
    """End-to-end tests using structured input (experiment_description + trials)."""

    @pytest.mark.asyncio
    async def test_minimal_structured_input(
        self,
        mock_backend: Any,
        simple_experiment_description: str,
        pending_discounting_trial: str,
    ) -> None:
        """Minimal structured input produces a valid prediction."""
        tool = CentaurPredictTool(backend=mock_backend)
        result = await tool.execute(
            {
                "experiment_description": simple_experiment_description,
                "pending_trial": pending_discounting_trial,
            }
        )

        assert result.success is True
        assert result.output is not None
        assert result.output["predicted_choice"] == "Q"
        assert result.output["probability"] == 1.0
        assert result.output["backend"] == "mock"

    @pytest.mark.asyncio
    async def test_structured_with_prior_trials(
        self,
        mock_backend: Any,
        simple_experiment_description: str,
        temporal_discounting_trials: list[dict[str, str]],
        pending_discounting_trial: str,
    ) -> None:
        """Prior trials are included in the prompt sent to backend."""
        tool = CentaurPredictTool(backend=mock_backend)
        await tool.execute(
            {
                "experiment_description": simple_experiment_description,
                "prior_trials": temporal_discounting_trials,
                "pending_trial": pending_discounting_trial,
            }
        )

        # Verify the prompt sent to the backend contains prior trial markers
        assert len(mock_backend.calls) == 1
        prompt_sent = mock_backend.calls[0]["prompt"]
        assert "<<Q>>." in prompt_sent  # First prior trial response
        assert "<<Z>>." in prompt_sent  # Second prior trial response
        assert prompt_sent.endswith(CHOICE_OPEN)

    @pytest.mark.asyncio
    async def test_structured_with_feedback(
        self,
        mock_backend: Any,
        simple_experiment_description: str,
        categorization_trials: list[dict[str, str]],
    ) -> None:
        """Feedback from prior trials appears in the prompt."""
        tool = CentaurPredictTool(backend=mock_backend)
        await tool.execute(
            {
                "experiment_description": simple_experiment_description,
                "prior_trials": categorization_trials,
                "pending_trial": "You see a small white triangle. You press",
            }
        )

        prompt_sent = mock_backend.calls[0]["prompt"]
        assert "The correct category is K." in prompt_sent
        assert "The correct category is E." in prompt_sent

    @pytest.mark.asyncio
    async def test_structured_with_probabilities(
        self,
        mock_backend_with_probs: Any,
        simple_experiment_description: str,
        pending_discounting_trial: str,
    ) -> None:
        """When backend returns log-probs, predictions include probabilities."""
        tool = CentaurPredictTool(backend=mock_backend_with_probs)
        result = await tool.execute(
            {
                "experiment_description": simple_experiment_description,
                "pending_trial": pending_discounting_trial,
            }
        )

        assert result.success is True
        assert result.output is not None
        assert result.output["predicted_choice"] == "Q"
        assert abs(result.output["probability"] - 0.73) < 0.01
        assert "Z" in result.output["alternatives"]
        assert abs(result.output["alternatives"]["Z"] - 0.27) < 0.01

    @pytest.mark.asyncio
    async def test_max_tokens_forwarded(
        self,
        mock_backend: Any,
        simple_experiment_description: str,
        pending_discounting_trial: str,
    ) -> None:
        """max_tokens from input is forwarded to the backend."""
        tool = CentaurPredictTool(backend=mock_backend, default_max_tokens=16)
        await tool.execute(
            {
                "experiment_description": simple_experiment_description,
                "pending_trial": pending_discounting_trial,
                "max_tokens": 4,
            }
        )

        assert mock_backend.calls[0]["max_tokens"] == 4

    @pytest.mark.asyncio
    async def test_default_max_tokens_used(
        self,
        mock_backend: Any,
        simple_experiment_description: str,
        pending_discounting_trial: str,
    ) -> None:
        """When max_tokens is not in input, default_max_tokens is used."""
        tool = CentaurPredictTool(backend=mock_backend, default_max_tokens=32)
        await tool.execute(
            {
                "experiment_description": simple_experiment_description,
                "pending_trial": pending_discounting_trial,
            }
        )

        assert mock_backend.calls[0]["max_tokens"] == 32


# =========================================================================
# CentaurPredictTool.execute — raw mode
# =========================================================================


class TestToolRawMode:
    """Tests using raw Psych-101 format prompts."""

    @pytest.mark.asyncio
    async def test_raw_prompt_passthrough(
        self,
        mock_backend: Any,
        raw_psych101_prompt: str,
    ) -> None:
        """Raw prompt is sent to backend without modification."""
        tool = CentaurPredictTool(backend=mock_backend)
        result = await tool.execute({"raw_prompt": raw_psych101_prompt})

        assert result.success is True
        prompt_sent = mock_backend.calls[0]["prompt"]
        assert prompt_sent == raw_psych101_prompt.strip()

    @pytest.mark.asyncio
    async def test_raw_prompt_invalid_raises_gracefully(
        self,
        mock_backend: Any,
    ) -> None:
        """Raw prompt without << markers returns a failure ToolResult."""
        tool = CentaurPredictTool(backend=mock_backend)
        result = await tool.execute({"raw_prompt": "No markers here"})

        assert result.success is False
        assert result.error is not None
        assert "must contain" in str(result.error["message"])

    @pytest.mark.asyncio
    async def test_raw_prompt_preferred_over_structured(
        self,
        mock_backend: Any,
        simple_experiment_description: str,
    ) -> None:
        """When both raw_prompt and structured fields are present, raw wins."""
        raw = "Raw prompt with << markers >>"
        tool = CentaurPredictTool(backend=mock_backend)
        await tool.execute(
            {
                "raw_prompt": raw,
                "experiment_description": simple_experiment_description,
                "pending_trial": "Should be ignored. You press",
            }
        )

        prompt_sent = mock_backend.calls[0]["prompt"]
        assert prompt_sent == raw.strip()


# =========================================================================
# CentaurPredictTool.execute — error handling
# =========================================================================


class TestToolErrorHandling:
    """Tests for graceful failure modes."""

    @pytest.mark.asyncio
    async def test_backend_exception_returns_failure(
        self,
        simple_experiment_description: str,
        pending_discounting_trial: str,
    ) -> None:
        """When the backend raises, the tool returns a failure ToolResult."""

        class FailingBackend:
            async def predict(self, prompt: str, max_tokens: int = 16) -> None:
                raise ConnectionError("GPU on fire")

            async def close(self) -> None:
                pass

            async def health_check(self) -> bool:
                return False

        tool = CentaurPredictTool(backend=FailingBackend())
        result = await tool.execute(
            {
                "experiment_description": simple_experiment_description,
                "pending_trial": pending_discounting_trial,
            }
        )

        assert result.success is False
        assert result.error is not None
        assert "GPU on fire" in str(result.error["message"])

    @pytest.mark.asyncio
    async def test_empty_description_returns_failure(
        self,
        mock_backend: Any,
    ) -> None:
        """Empty experiment_description causes a validation error."""
        tool = CentaurPredictTool(backend=mock_backend)
        result = await tool.execute(
            {
                "experiment_description": "",
                "pending_trial": "You press",
            }
        )

        assert result.success is False
        assert "must not be empty" in str(result.error["message"])

    @pytest.mark.asyncio
    async def test_missing_both_modes_returns_failure(
        self,
        mock_backend: Any,
    ) -> None:
        """Input with neither raw_prompt nor structured fields fails."""
        tool = CentaurPredictTool(backend=mock_backend)
        result = await tool.execute({})

        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_empty_raw_prompt_falls_through_to_structured(
        self,
        mock_backend: Any,
    ) -> None:
        """Empty string raw_prompt falls through to structured mode."""
        tool = CentaurPredictTool(backend=mock_backend)
        result = await tool.execute(
            {
                "raw_prompt": "",
                "experiment_description": "",
                "pending_trial": "You press",
            }
        )

        # Falls through because raw_prompt is falsy, then structured
        # mode fails because experiment_description is empty
        assert result.success is False


# =========================================================================
# CentaurPredictTool — output shape
# =========================================================================


class TestToolOutputShape:
    """Verify the output dict structure matches the documented contract."""

    @pytest.mark.asyncio
    async def test_output_has_all_fields(
        self,
        mock_backend: Any,
        simple_experiment_description: str,
        pending_discounting_trial: str,
    ) -> None:
        """Successful output contains all documented fields."""
        tool = CentaurPredictTool(backend=mock_backend)
        result = await tool.execute(
            {
                "experiment_description": simple_experiment_description,
                "pending_trial": pending_discounting_trial,
            }
        )

        assert result.success is True
        output = result.output
        assert output is not None

        required_keys = {
            "predicted_choice",
            "probability",
            "alternatives",
            "raw_output",
            "prompt_used",
            "backend",
        }
        assert required_keys <= set(output.keys())

    @pytest.mark.asyncio
    async def test_probability_is_rounded(
        self,
        mock_backend_with_probs: Any,
        simple_experiment_description: str,
        pending_discounting_trial: str,
    ) -> None:
        """Probabilities are rounded to 4 decimal places."""
        tool = CentaurPredictTool(backend=mock_backend_with_probs)
        result = await tool.execute(
            {
                "experiment_description": simple_experiment_description,
                "pending_trial": pending_discounting_trial,
            }
        )

        assert result.output is not None
        prob = result.output["probability"]
        # Should be rounded to 4 decimal places
        assert prob == round(prob, 4)

    @pytest.mark.asyncio
    async def test_prompt_used_is_reproducible(
        self,
        mock_backend: Any,
        simple_experiment_description: str,
        pending_discounting_trial: str,
    ) -> None:
        """prompt_used in output matches what was sent to backend."""
        tool = CentaurPredictTool(backend=mock_backend)
        result = await tool.execute(
            {
                "experiment_description": simple_experiment_description,
                "pending_trial": pending_discounting_trial,
            }
        )

        assert result.output is not None
        assert result.output["prompt_used"] == mock_backend.calls[0]["prompt"]


# =========================================================================
# Tool metadata
# =========================================================================


class TestToolMetadata:
    """Verify tool name, description, and schema."""

    def test_name(self, mock_backend: Any) -> None:
        tool = CentaurPredictTool(backend=mock_backend)
        assert tool.name == "centaur_predict"

    def test_description_mentions_centaur(self, mock_backend: Any) -> None:
        tool = CentaurPredictTool(backend=mock_backend)
        desc = tool.description
        assert "Centaur" in desc
        assert "Psych-101" in desc

    def test_description_mentions_both_modes(self, mock_backend: Any) -> None:
        tool = CentaurPredictTool(backend=mock_backend)
        desc = tool.description
        assert "structured" in desc
        assert "raw" in desc

    def test_input_schema_is_valid_jsonschema(self) -> None:
        """TOOL_INPUT_SCHEMA has the expected top-level structure."""
        assert TOOL_INPUT_SCHEMA["type"] == "object"
        assert "properties" in TOOL_INPUT_SCHEMA
        assert "oneOf" in TOOL_INPUT_SCHEMA

        props = TOOL_INPUT_SCHEMA["properties"]
        assert isinstance(props, dict)
        assert "experiment_description" in props
        assert "prior_trials" in props
        assert "pending_trial" in props
        assert "raw_prompt" in props
        assert "max_tokens" in props

    def test_input_schema_oneof_modes(self) -> None:
        """oneOf defines exactly two valid input modes."""
        one_of = TOOL_INPUT_SCHEMA["oneOf"]
        assert isinstance(one_of, list)
        assert len(one_of) == 2

        # Mode 1: structured
        structured = one_of[0]
        assert "experiment_description" in structured["required"]
        assert "pending_trial" in structured["required"]

        # Mode 2: raw
        raw = one_of[1]
        assert "raw_prompt" in raw["required"]


# =========================================================================
# mount()
# =========================================================================


class TestMount:
    """Tests for the module mount entry point."""

    @pytest.mark.asyncio
    async def test_mount_registers_tool(self, mock_coordinator: Any) -> None:
        """mount() registers a tool named 'centaur_predict'."""
        cleanup = await mount(mock_coordinator, config={})

        assert len(mock_coordinator.mounted) == 1
        entry = mock_coordinator.mounted[0]
        assert entry["category"] == "tools"
        assert entry["name"] == "centaur_predict"
        assert isinstance(entry["tool"], CentaurPredictTool)

        # Cleanup should be callable
        assert cleanup is not None
        await cleanup()

    @pytest.mark.asyncio
    async def test_mount_with_config(self, mock_coordinator: Any) -> None:
        """mount() respects backend config."""
        cleanup = await mount(
            mock_coordinator,
            config={
                "backend": {
                    "type": "rest_api",
                    "api_url": "http://custom:9999",
                },
                "max_tokens": 32,
            },
        )

        tool = mock_coordinator.mounted[0]["tool"]
        assert tool._default_max_tokens == 32
        await cleanup()

    @pytest.mark.asyncio
    async def test_mount_none_config(self, mock_coordinator: Any) -> None:
        """mount() handles None config gracefully."""
        cleanup = await mount(mock_coordinator, config=None)

        assert len(mock_coordinator.mounted) == 1
        await cleanup()

    @pytest.mark.asyncio
    async def test_cleanup_closes_backend(self, mock_coordinator: Any) -> None:
        """The cleanup function returned by mount() closes the backend."""
        # We can't easily inspect the internal backend, but we can verify
        # cleanup doesn't raise.
        cleanup = await mount(mock_coordinator, config={})
        await cleanup()


# =========================================================================
# _FallbackToolResult
# =========================================================================


class TestFallbackToolResult:
    """Verify the fallback ToolResult used when amplifier_core is absent."""

    def test_success_result(self) -> None:
        r = _FallbackToolResult(success=True, output={"choice": "Q"})
        assert r.success is True
        assert r.output == {"choice": "Q"}
        assert r.error is None

    def test_failure_result(self) -> None:
        r = _FallbackToolResult(success=False, error={"message": "boom"})
        assert r.success is False
        assert r.error == {"message": "boom"}
        assert r.output is None

    def test_repr_success(self) -> None:
        r = _FallbackToolResult(success=True, output={"x": 1})
        assert "success=True" in repr(r)

    def test_repr_failure(self) -> None:
        r = _FallbackToolResult(success=False, error={"message": "fail"})
        assert "success=False" in repr(r)
