"""
Centaur prompt formatting and result parsing.

Encapsulates the << >> token protocol used by the Psych-101 dataset and the
Centaur model.  Nothing outside this module should know about the raw token
format.  All other code works with structured Python objects.

Key concepts
------------
- **Choice markers** ``<<`` and ``>>`` delimit every human response in a
  Psych-101 transcript.  The model is trained to produce text *inside* these
  markers as next-token predictions.
- A **trial** is one stimulus-response pair, optionally followed by feedback.
- A **pending trial** ends with an *open* ``<<`` so the model completes the
  choice.

Public API
----------
build_prompt          Build a prompt from structured parts.
build_prompt_raw      Validate and pass through a pre-formatted prompt.
parse_completion      Turn raw model output into ``CentaurPrediction`` objects.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHOICE_OPEN: str = "<<"
CHOICE_CLOSE: str = ">>"

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Trial:
    """A single completed trial in an experiment transcript.

    Attributes
    ----------
    stimulus:
        What the participant observed, in natural language.
        Example: ``"You have the choice between receiving $500 immediately
        (press Q) or receiving $550 in one year (press Z). You press"``
    response:
        The participant's choice token.
        Example: ``"Q"``
    feedback:
        Optional feedback shown after the choice.
        Example: ``"The correct category is K."``
    """

    stimulus: str
    response: str
    feedback: str = ""


@dataclass(frozen=True)
class CentaurPrediction:
    """A single predicted human choice with its probability.

    Attributes
    ----------
    choice:
        The predicted response token (e.g. ``"Q"``, ``"K"``, ``"left"``).
    probability:
        The model's probability for *this* token, in [0, 1].
        ``None`` when the backend did not return a probability for the choice.
    alternatives:
        Mapping of other candidate tokens to their probabilities, sorted
        descending.  Empty when log-probs are unavailable.
    """

    choice: str
    probability: float | None
    alternatives: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CentaurResult:
    """Complete result envelope returned by the tool.

    Attributes
    ----------
    prompt_used:
        The exact text sent to the model (for reproducibility).
    predictions:
        One ``CentaurPrediction`` per choice point resolved.  In the
        common single-prediction case the list has length 1.
    raw_output:
        The verbatim text the model generated (before parsing).
    metadata:
        Backend-specific information (model name, latency, etc.).
    """

    prompt_used: str
    predictions: list[CentaurPrediction]
    raw_output: str
    metadata: dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_prompt(
    experiment_description: str,
    prior_trials: Sequence[Trial | dict[str, str]] | None = None,
    pending_trial: str | None = None,
) -> str:
    """Build a Centaur-compatible prompt from structured parts.

    Parameters
    ----------
    experiment_description:
        Natural-language instructions a participant would read at the
        start of the experiment.  Whitespace is normalised but the text
        is otherwise preserved verbatim.
    prior_trials:
        Zero or more completed trials.  Each element is either a
        :class:`Trial` or a plain ``dict`` with keys ``"stimulus"``,
        ``"response"``, and optionally ``"feedback"``.
    pending_trial:
        The stimulus for the trial whose response the model should
        predict.  The function ensures the string ends with an open
        ``<<`` marker.  Pass ``None`` to build a prompt that contains
        only context (useful for log-likelihood evaluation).

    Returns
    -------
    str
        A single string ready to send to the Centaur model.

    Raises
    ------
    ValueError
        If *experiment_description* is empty.

    Examples
    --------
    >>> build_prompt(
    ...     experiment_description="Choose between Q and Z.",
    ...     prior_trials=[
    ...         {"stimulus": "Option A (Q) or B (Z). You press",
    ...          "response": "Q"},
    ...     ],
    ...     pending_trial="Option C (Q) or D (Z). You press",
    ... )  # doctest: +NORMALIZE_WHITESPACE
    'Choose between Q and Z.\\nOption A (Q) or B (Z). You press <<Q>>.\\nOption C (Q) or D (Z). You press <<'
    """
    experiment_description = experiment_description.strip()
    if not experiment_description:
        raise ValueError("experiment_description must not be empty")

    parts: list[str] = [experiment_description]

    for raw_trial in prior_trials or []:
        trial = _to_trial(raw_trial)
        line = (
            f"{trial.stimulus.strip()} "
            f"{CHOICE_OPEN}{trial.response.strip()}{CHOICE_CLOSE}."
        )
        if trial.feedback:
            line += f" {trial.feedback.strip()}"
        parts.append(line)

    if pending_trial is not None:
        pending = pending_trial.strip()
        if not pending.endswith(CHOICE_OPEN):
            pending += f" {CHOICE_OPEN}"
        parts.append(pending)

    return "\n".join(parts)


def build_prompt_raw(raw_text: str) -> str:
    """Validate and return a pre-formatted Centaur prompt.

    Use this when the caller already has a Psych-101-style transcript
    and does not need automatic formatting.

    Parameters
    ----------
    raw_text:
        Prompt text that must contain at least one ``<<`` token.

    Returns
    -------
    str
        The input text, stripped of leading/trailing whitespace.

    Raises
    ------
    ValueError
        If no ``<<`` marker is found.
    """
    text = raw_text.strip()
    if CHOICE_OPEN not in text:
        raise ValueError(
            f"Raw prompt must contain at least one '{CHOICE_OPEN}' marker. "
            "Use build_prompt() for automatic formatting."
        )
    return text


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------


def parse_completion(
    completion_text: str,
    token_log_probs: dict[str, float] | None = None,
) -> list[CentaurPrediction]:
    """Parse raw model output into structured predictions.

    Parameters
    ----------
    completion_text:
        The text the model generated *after* the last ``<<`` marker.
        Typically very short (one or two tokens), e.g. ``"Q>>"`` or
        ``"left>>."``.
    token_log_probs:
        Optional mapping of candidate tokens to their **log-probabilities**
        (natural log, as returned by most serving frameworks).  When
        provided these are converted to probabilities and attached to
        the prediction.  Pass ``None`` when the backend does not
        support log-probs.

    Returns
    -------
    list[CentaurPrediction]
        A list containing one prediction.  (The list wrapper future-proofs
        for multi-step prediction without changing the return type.)

    Examples
    --------
    >>> parse_completion("Q>>.")
    [CentaurPrediction(choice='Q', probability=None, alternatives={})]

    >>> import math
    >>> preds = parse_completion(
    ...     "Q>>.",
    ...     token_log_probs={"Q": math.log(0.73), "Z": math.log(0.27)},
    ... )
    >>> preds[0].choice
    'Q'
    >>> round(preds[0].probability, 2)
    0.73
    """
    # Extract the chosen token: everything before ">>" or end-of-string,
    # stripping any trailing punctuation.
    match = re.match(r"\s*([^>\s.]+)", completion_text)
    choice = match.group(1).strip() if match else completion_text.strip()

    if token_log_probs is not None:
        probs = _logprobs_to_probs(token_log_probs)
        top_prob = probs.get(choice)
        alternatives = {
            k: v
            for k, v in sorted(probs.items(), key=lambda x: x[1], reverse=True)
            if k != choice
        }
    else:
        top_prob = None
        alternatives = {}

    return [
        CentaurPrediction(
            choice=choice,
            probability=top_prob,
            alternatives=alternatives,
        )
    ]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _to_trial(raw: Trial | dict[str, str]) -> Trial:
    """Coerce a dict to a Trial, or pass through if already one."""
    if isinstance(raw, Trial):
        return raw
    return Trial(
        stimulus=raw["stimulus"],
        response=raw["response"],
        feedback=raw.get("feedback", ""),
    )


def _logprobs_to_probs(log_probs: dict[str, float]) -> dict[str, float]:
    """Convert a {token: log_prob} mapping to {token: probability}.

    Handles both natural-log and already-normalised values gracefully.
    Strips whitespace from token keys.
    """
    probs: dict[str, float] = {}
    for token, lp in log_probs.items():
        key = token.strip()
        if not key:
            continue
        probs[key] = (
            math.exp(lp) if lp <= 0 else lp
        )  # Guard: if already a prob, pass through
    return probs
