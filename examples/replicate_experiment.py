#!/usr/bin/env python3
"""
Replicate Experiment: Replicate a full Psych-101 experiment trial-by-trial.

Cognitive science background
----------------------------
A core use case for the Centaur model is *in-silico replication* of published
behavioural experiments.  Instead of recruiting participants and running a
study (weeks, thousands of dollars), you can feed the experiment's transcript
format to Centaur and get predicted human responses in seconds.

This is valuable for:
- **Power analysis**: Estimate effect sizes before committing to a full study.
- **Stimulus pre-testing**: Identify floor/ceiling effects in your materials.
- **Replication prediction**: Before attempting a replication, check whether
  the model predicts the original effect will hold.
- **Teaching**: Demonstrate experimental paradigms without live participants.

The Centaur paper (Binz et al., 2025) shows that the model matches or
exceeds the predictive accuracy of 180 cognitive models across the Psych-101
benchmark.

This example
------------
We replicate a simplified temporal-discounting experiment with 8 trials.
For each trial we:
1. Build the prompt with all prior trials as context.
2. Call the mock backend to get a prediction.
3. Record the prediction and move to the next trial.

At the end we print the full trajectory and a summary of the implied
discount rate.

Usage:
    python examples/replicate_experiment.py
"""

from __future__ import annotations

import asyncio
import math
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make the module importable without pip install.
# ---------------------------------------------------------------------------
_MODULE_ROOT = Path(__file__).resolve().parent.parent / "modules" / "tool-centaur"
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))

from amplifier_module_tool_centaur import CentaurPredictTool  # noqa: E402
from amplifier_module_tool_centaur.backends import BackendResult  # noqa: E402


# ---------------------------------------------------------------------------
# Experiment definition
# ---------------------------------------------------------------------------

DESCRIPTION = (
    "You will be presented with choices between two monetary options.\n"
    "Option Q is always available immediately.\n"
    "Option Z is available after a delay.\n"
    "Press Q or Z to indicate your preference."
)

# Each trial: (immediate_amount, delayed_amount, delay_months)
TRIALS: list[tuple[int, int, int]] = [
    (100, 110, 1),
    (100, 120, 1),
    (100, 150, 3),
    (100, 150, 6),
    (100, 200, 6),
    (100, 200, 12),
    (100, 300, 12),
    (100, 500, 12),
]


def make_stimulus(immediate: int, delayed: int, delay_months: int) -> str:
    """Format a temporal-discounting stimulus in Psych-101 style."""
    delay_str = f"{delay_months} month{'s' if delay_months != 1 else ''}"
    return (
        f"You have the choice between receiving ${immediate} immediately "
        f"(press Q) or receiving ${delayed} in {delay_str} (press Z). "
        f"You press"
    )


# ---------------------------------------------------------------------------
# MockBackend with trial-aware responses
# ---------------------------------------------------------------------------


class SequentialMockBackend:
    """Returns pre-defined responses for each trial to simulate a realistic
    discounting curve.

    A typical participant takes the immediate reward when the delayed
    option is only slightly better, but switches to the delayed option
    as the ratio improves.
    """

    def __init__(self) -> None:
        # Pre-defined choices and probabilities for each trial.
        # This mimics a participant with moderate discounting.
        self._responses = [
            ("Q", 0.82),  # $100 now vs $110 in 1m  -> immediate
            ("Q", 0.68),  # $100 now vs $120 in 1m  -> immediate
            ("Q", 0.55),  # $100 now vs $150 in 3m  -> immediate (barely)
            ("Z", 0.52),  # $100 now vs $150 in 6m  -> delayed (barely)
            ("Z", 0.71),  # $100 now vs $200 in 6m  -> delayed
            ("Q", 0.58),  # $100 now vs $200 in 12m -> immediate (long delay)
            ("Z", 0.65),  # $100 now vs $300 in 12m -> delayed
            ("Z", 0.88),  # $100 now vs $500 in 12m -> delayed (obvious)
        ]
        self._call_count = 0

    async def predict(self, prompt: str, max_tokens: int = 16) -> BackendResult:
        idx = min(self._call_count, len(self._responses) - 1)
        choice, prob = self._responses[idx]
        alt = "Z" if choice == "Q" else "Q"
        self._call_count += 1

        return BackendResult(
            completion=f"{choice}>>.",
            token_log_probs={
                choice: math.log(prob),
                alt: math.log(1 - prob),
            },
            metadata={"backend": "mock", "trial": idx + 1},
        )

    async def close(self) -> None:
        pass

    async def health_check(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    backend = SequentialMockBackend()
    tool = CentaurPredictTool(backend=backend)

    completed_trials: list[dict[str, str]] = []
    results: list[dict[str, object]] = []

    print("=== Replicating Temporal Discounting Experiment ===")
    print(
        f"{'Trial':>5}  {'Immediate':>10}  {'Delayed':>10}  {'Delay':>8}  "
        f"{'Choice':>8}  {'P(choice)':>10}"
    )
    print("-" * 65)

    for i, (immediate, delayed, delay_months) in enumerate(TRIALS, 1):
        stimulus = make_stimulus(immediate, delayed, delay_months)

        result = await tool.execute(
            {
                "experiment_description": DESCRIPTION,
                "prior_trials": completed_trials,
                "pending_trial": stimulus,
            }
        )

        choice = result.output["predicted_choice"]
        prob = result.output["probability"]

        delay_str = f"{delay_months}m"
        choice_label = f"${immediate}" if choice == "Q" else f"${delayed}"
        print(
            f"{i:>5}  ${immediate:>9}  ${delayed:>9}  {delay_str:>8}  "
            f"{choice_label:>8}  {format(prob, '>9.1%') if prob is not None else 'unavailable'}"
        )

        # Record this trial for context in the next prediction.
        completed_trials.append({"stimulus": stimulus, "response": choice})
        results.append(
            {
                "trial": i,
                "immediate": immediate,
                "delayed": delayed,
                "delay_months": delay_months,
                "choice": choice,
                "probability": prob,
            }
        )

    # Summary
    immediate_count = sum(1 for r in results if r["choice"] == "Q")
    delayed_count = sum(1 for r in results if r["choice"] == "Z")
    print("-" * 65)
    print(f"Summary: {immediate_count} immediate, {delayed_count} delayed choices")
    print()
    print("Interpretation: The simulated participant shows a classic hyperbolic")
    print("discounting pattern -- preferring immediate rewards when the delayed")
    print("option is only slightly better or far in the future, but switching")
    print("to the delayed option as the ratio improves.  Note the 'reversal'")
    print("at trial 6 ($200 in 12 months) where the long delay tips the")
    print("balance back to immediate despite good value.")
    print()
    print("With a real backend, each prediction would reflect the aggregate")
    print("behaviour of 60,000+ participants from the Psych-101 dataset.")

    await backend.close()


if __name__ == "__main__":
    asyncio.run(main())
