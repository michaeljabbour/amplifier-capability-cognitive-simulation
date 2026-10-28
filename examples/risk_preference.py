#!/usr/bin/env python3
"""
Risk Preference: Predict risky vs. safe gambling choices.

Cognitive science background
----------------------------
Risk preference -- the tendency to seek or avoid uncertainty -- is one of the
most-studied topics in behavioural decision-making.  Prospect Theory (Kahneman
& Tversky, 1979) showed that people are typically risk-averse for gains and
risk-seeking for losses, and that they overweight small probabilities.

The Psych-101 dataset includes several risky-choice experiments:
- frey2017risk: Description-based gambles with stated probabilities
- wulff2018description: Description vs. experience-based risk paradigms

In these experiments, participants see a lottery (e.g. "50% chance of $100")
alongside a sure thing (e.g. "$40 for sure") and choose one.  The Centaur
model learns the population-level patterns of risk preference from these
transcripts.

This example
------------
We present a sequence of risky-choice trials with increasing expected value
of the gamble relative to the sure thing.  The prior trials show the
participant switching from safe to risky as the gamble becomes more
attractive.  We predict the next choice at the switching boundary.

Usage:
    python examples/risk_preference.py
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
# MockBackend
# ---------------------------------------------------------------------------


class MockBackend:
    """Simulates a Centaur prediction for a risky-choice boundary trial."""

    async def predict(self, prompt: str, max_tokens: int = 16) -> BackendResult:
        # At the switching boundary, the model is near-indifferent.
        return BackendResult(
            completion="Z>>.",
            token_log_probs={
                "Z": math.log(0.54),  # Sure thing (risk-averse lean)
                "Q": math.log(0.46),  # Gamble
            },
            metadata={"backend": "mock"},
        )

    async def close(self) -> None:
        pass

    async def health_check(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    tool = CentaurPredictTool(backend=MockBackend())

    result = await tool.execute(
        {
            "experiment_description": (
                "In each round you will see two options. One is a sure amount "
                "and the other is a gamble with a stated probability. Press Q "
                "to choose the gamble or Z to choose the sure thing."
            ),
            "prior_trials": [
                # Low EV gamble -> participant chooses safe
                {
                    "stimulus": (
                        "Option Q: 50% chance of $100, 50% chance of $0. "
                        "Option Z: $40 for sure. You press"
                    ),
                    "response": "Z",
                },
                # Moderate EV gamble -> still safe
                {
                    "stimulus": (
                        "Option Q: 50% chance of $100, 50% chance of $0. "
                        "Option Z: $60 for sure. You press"
                    ),
                    "response": "Z",
                },
                # High EV gamble -> switches to risky
                {
                    "stimulus": (
                        "Option Q: 80% chance of $200, 20% chance of $0. "
                        "Option Z: $120 for sure. You press"
                    ),
                    "response": "Q",
                },
                # Very attractive gamble -> risky again
                {
                    "stimulus": (
                        "Option Q: 90% chance of $300, 10% chance of $0. "
                        "Option Z: $150 for sure. You press"
                    ),
                    "response": "Q",
                },
            ],
            # Boundary trial: gamble EV = $150, sure = $100
            "pending_trial": (
                "Option Q: 30% chance of $500, 70% chance of $0. "
                "Option Z: $100 for sure. You press"
            ),
        }
    )

    print("=== Risk Preference Prediction ===")
    print()
    print("Pending trial:")
    print("  Gamble (Q) : 30% chance of $500  (EV = $150)")
    print("  Sure   (Z) : $100 for sure")
    print()
    print(
        f"Predicted choice : {'Gamble (Q)' if result.output['predicted_choice'] == 'Q' else 'Sure thing (Z)'}"
    )
    print(f"Probability      : {format(result.output['probability'], '.2%') if result.output['probability'] is not None else 'unavailable'}")
    print(f"Alternatives     : {result.output['alternatives']}")
    print()
    print("Interpretation: Despite the gamble having higher expected value")
    print("($150 vs $100), the model predicts slight preference for the")
    print("sure thing (54% vs 46%).  This is consistent with Prospect Theory:")
    print("the low probability (30%) of the gamble is underweighted relative")
    print("to its stated value, and the certainty of $100 is overweighted.")
    print()
    print("This 'certainty effect' is one of the most robust findings in")
    print("behavioural economics, and the Centaur model captures it from")
    print("the statistical patterns in real participant data.")


if __name__ == "__main__":
    asyncio.run(main())
