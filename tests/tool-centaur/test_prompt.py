"""
Tests for amplifier_module_tool_centaur.prompt

Covers:
- build_prompt: structured prompt construction
- build_prompt_raw: pass-through validation
- parse_completion: result parsing with and without log-probs
- Edge cases: empty inputs, missing feedback, multi-trial sequences
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# Ensure importable without pip install
_MODULE_ROOT = (
    Path(__file__).resolve().parent.parent.parent / "modules" / "tool-centaur"
)
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))

from amplifier_module_tool_centaur.prompt import (  # noqa: E402
    CHOICE_CLOSE,
    CHOICE_OPEN,
    CentaurPrediction,
    Trial,
    build_prompt,
    build_prompt_raw,
    parse_completion,
)


# =========================================================================
# build_prompt
# =========================================================================


class TestBuildPrompt:
    """Tests for structured prompt construction."""

    def test_minimal_prompt(self, simple_experiment_description: str) -> None:
        """Description + pending trial produces a valid prompt."""
        result = build_prompt(
            experiment_description=simple_experiment_description,
            pending_trial="Choose $100 now (Q) or $150 later (Z). You press",
        )

        assert result.startswith("You will be presented")
        assert result.endswith(CHOICE_OPEN)
        # No prior trials, so only 2 lines
        lines = result.strip().split("\n")
        assert len(lines) == 3  # 2-line description + 1 pending trial

    def test_with_prior_trials_dict(
        self,
        simple_experiment_description: str,
        temporal_discounting_trials: list[dict[str, str]],
        pending_discounting_trial: str,
    ) -> None:
        """Prior trials from dicts are formatted with << >> markers."""
        result = build_prompt(
            experiment_description=simple_experiment_description,
            prior_trials=temporal_discounting_trials,
            pending_trial=pending_discounting_trial,
        )

        # Prior trials should have closed markers
        assert "<<Q>>." in result
        assert "<<Z>>." in result
        # Pending trial should end with open marker
        assert result.endswith(CHOICE_OPEN)

    def test_with_prior_trials_objects(
        self, simple_experiment_description: str
    ) -> None:
        """Trial objects work the same as dicts."""
        trials = [
            Trial(stimulus="You see X. You press", response="A"),
            Trial(stimulus="You see Y. You press", response="B", feedback="Correct."),
        ]
        result = build_prompt(
            experiment_description=simple_experiment_description,
            prior_trials=trials,
            pending_trial="You see Z. You press",
        )

        assert "<<A>>." in result
        assert "<<B>>. Correct." in result
        assert result.endswith(CHOICE_OPEN)

    def test_with_feedback(
        self,
        simple_experiment_description: str,
        categorization_trials: list[dict[str, str]],
    ) -> None:
        """Feedback is appended after the closing >> marker."""
        result = build_prompt(
            experiment_description=simple_experiment_description,
            prior_trials=categorization_trials,
            pending_trial="You see a small white triangle. You press",
        )

        assert "<<K>>. The correct category is K." in result
        assert "<<K>>. The correct category is E." in result
        assert "<<E>>. The correct category is K." in result

    def test_pending_trial_already_has_marker(
        self, simple_experiment_description: str
    ) -> None:
        """If pending_trial already ends with <<, don't double it."""
        result = build_prompt(
            experiment_description=simple_experiment_description,
            pending_trial="You press <<",
        )

        assert result.endswith("<<")
        assert not result.endswith("<<" + " <<")

    def test_no_pending_trial(self, simple_experiment_description: str) -> None:
        """Omitting pending_trial produces context-only prompt (for eval)."""
        result = build_prompt(
            experiment_description=simple_experiment_description,
            prior_trials=[{"stimulus": "You press", "response": "Q"}],
            pending_trial=None,
        )

        assert "<<Q>>." in result
        assert not result.endswith(CHOICE_OPEN)

    def test_no_prior_trials(self, simple_experiment_description: str) -> None:
        """Empty prior_trials list is fine."""
        result = build_prompt(
            experiment_description=simple_experiment_description,
            prior_trials=[],
            pending_trial="You press",
        )

        assert result.endswith(CHOICE_OPEN)
        assert CHOICE_CLOSE not in result  # No completed trials

    def test_empty_description_raises(self) -> None:
        """Empty experiment_description is rejected."""
        with pytest.raises(ValueError, match="must not be empty"):
            build_prompt(
                experiment_description="",
                pending_trial="You press",
            )

    def test_whitespace_normalisation(self) -> None:
        """Leading/trailing whitespace is stripped from all parts."""
        result = build_prompt(
            experiment_description="  Instructions  ",
            prior_trials=[{"stimulus": "  You see X. You press  ", "response": " A "}],
            pending_trial="  You see Y. You press  ",
        )

        assert result.startswith("Instructions")
        assert "<<A>>." in result
        assert result.endswith(CHOICE_OPEN)

    def test_multi_line_description(self) -> None:
        """Multi-line descriptions are preserved."""
        desc = "Line one.\nLine two.\nLine three."
        result = build_prompt(
            experiment_description=desc,
            pending_trial="You press",
        )

        assert "Line one.\nLine two.\nLine three." in result


# =========================================================================
# build_prompt_raw
# =========================================================================


class TestBuildPromptRaw:
    """Tests for raw prompt pass-through."""

    def test_valid_raw_prompt(self, raw_psych101_prompt: str) -> None:
        """A prompt with << markers passes through."""
        result = build_prompt_raw(raw_psych101_prompt)
        assert CHOICE_OPEN in result
        assert result == raw_psych101_prompt.strip()

    def test_strips_whitespace(self) -> None:
        """Leading/trailing whitespace is removed."""
        result = build_prompt_raw("  You press <<Q>>  ")
        assert result == "You press <<Q>>"

    def test_missing_marker_raises(self) -> None:
        """A prompt without << is rejected."""
        with pytest.raises(ValueError, match="must contain"):
            build_prompt_raw("No markers here at all.")

    def test_empty_string_raises(self) -> None:
        """Empty string is rejected (no markers)."""
        with pytest.raises(ValueError, match="must contain"):
            build_prompt_raw("")


# =========================================================================
# parse_completion
# =========================================================================


class TestParseCompletion:
    """Tests for result parsing."""

    def test_simple_completion(self) -> None:
        """'Q>>.' extracts choice 'Q' without inventing a probability."""
        preds = parse_completion("Q>>.")
        assert len(preds) == 1
        assert preds[0].choice == "Q"
        assert preds[0].probability is None
        assert preds[0].alternatives == {}

    def test_completion_without_close(self) -> None:
        """A completion without >> still extracts the token."""
        preds = parse_completion("Z")
        assert preds[0].choice == "Z"

    def test_completion_with_whitespace(self) -> None:
        """Leading whitespace is stripped."""
        preds = parse_completion("  Q>>")
        assert preds[0].choice == "Q"

    def test_multi_char_choice(self) -> None:
        """Multi-character tokens like 'left' are supported."""
        preds = parse_completion("left>>.")
        assert preds[0].choice == "left"

    def test_with_log_probs(self) -> None:
        """Log-probs are converted to probabilities."""
        log_probs = {
            "Q": math.log(0.73),
            "Z": math.log(0.27),
        }
        preds = parse_completion("Q>>.", token_log_probs=log_probs)

        assert preds[0].choice == "Q"
        assert abs(preds[0].probability - 0.73) < 0.01
        assert "Z" in preds[0].alternatives
        assert abs(preds[0].alternatives["Z"] - 0.27) < 0.01

    def test_log_probs_sorted_descending(self) -> None:
        """Alternatives are sorted by probability (highest first)."""
        log_probs = {
            "Q": math.log(0.5),
            "Z": math.log(0.3),
            "X": math.log(0.2),
        }
        preds = parse_completion("Q>>.", token_log_probs=log_probs)

        alt_vals = list(preds[0].alternatives.values())
        assert alt_vals == sorted(alt_vals, reverse=True)

    def test_log_probs_with_whitespace_keys(self) -> None:
        """Token keys with whitespace are stripped."""
        log_probs = {
            " Q": math.log(0.7),
            " Z ": math.log(0.3),
        }
        preds = parse_completion("Q>>.", token_log_probs=log_probs)
        assert preds[0].choice == "Q"
        # Stripped keys should match
        all_keys = {preds[0].choice} | set(preds[0].alternatives.keys())
        assert "Q" in all_keys
        assert "Z" in all_keys

    def test_none_log_probs(self) -> None:
        """Explicitly passing None keeps probability unavailable."""
        preds = parse_completion("Q>>.", token_log_probs=None)
        assert preds[0].probability is None
        assert preds[0].alternatives == {}


# =========================================================================
# Data class contracts
# =========================================================================


class TestDataClasses:
    """Verify data class behaviour."""

    def test_trial_frozen(self) -> None:
        """Trial is immutable."""
        t = Trial(stimulus="X", response="Y")
        with pytest.raises(AttributeError):
            t.stimulus = "Z"  # type: ignore[misc]

    def test_prediction_frozen(self) -> None:
        """CentaurPrediction is immutable."""
        p = CentaurPrediction(choice="Q", probability=0.73)
        with pytest.raises(AttributeError):
            p.choice = "Z"  # type: ignore[misc]

    def test_trial_defaults(self) -> None:
        """Trial.feedback defaults to empty string."""
        t = Trial(stimulus="X", response="Y")
        assert t.feedback == ""

    def test_prediction_defaults(self) -> None:
        """CentaurPrediction.alternatives defaults to empty dict."""
        p = CentaurPrediction(choice="Q", probability=1.0)
        assert p.alternatives == {}


def test_missing_choice_in_partial_distribution_is_unknown():
    prediction = parse_completion("Q>>", {"Z": math.log(0.2)})[0]
    assert prediction.probability is None
    assert prediction.alternatives == {"Z": 0.2}
