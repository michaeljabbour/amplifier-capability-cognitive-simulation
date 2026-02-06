#!/usr/bin/env python3
"""
UX A/B Test: Predict which UI design users would prefer.

Cognitive science background
----------------------------
A/B testing in UX design is fundamentally a preference-elicitation task --
the same class of problem that the Psych-101 dataset covers extensively
through risky-choice, temporal-discounting, and similarity-judgement
experiments.  Centaur learns general patterns of human preference from
60,000+ participants, making it useful for *pre-screening* design options
before running a live A/B test.

This is NOT a replacement for real user testing.  It is a fast, cheap way
to generate hypotheses: "Centaur predicts 70% of users will prefer Design A
over Design B.  Let's validate that with a live test focused on these two
finalists instead of testing all five candidates."

This example
------------
We frame a UX preference task using the same two-alternative forced-choice
format that Psych-101 uses for risky-choice experiments.  Three prior
"participants" have already stated preferences, and we predict what the
fourth participant will choose.

Usage:
    python examples/ux_ab_test.py
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
    """Simulates a Centaur prediction for the UX scenario."""

    async def predict(self, prompt: str, max_tokens: int = 16) -> BackendResult:
        return BackendResult(
            completion="Q>>.",
            token_log_probs={
                "Q": math.log(0.67),  # Design A: clean/minimal
                "Z": math.log(0.33),  # Design B: feature-rich
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

    # Frame the UX A/B test as a Psych-101-style two-alternative choice.
    # Each "trial" is one participant's preference judgement.
    result = await tool.execute(
        {
            "experiment_description": (
                "You are evaluating two user interface designs for a mobile app.\n"
                "Design Q has a clean, minimal layout with large touch targets "
                "and a single primary action per screen.\n"
                "Design Z has a feature-rich layout showing more information "
                "per screen with smaller controls.\n"
                "Press Q if you prefer Design Q, or Z if you prefer Design Z."
            ),
            "prior_trials": [
                {
                    "stimulus": (
                        "Participant views both designs on a phone. "
                        "Which design do you prefer for everyday use? You press"
                    ),
                    "response": "Q",
                },
                {
                    "stimulus": (
                        "Participant views both designs on a phone. "
                        "Which design do you prefer for everyday use? You press"
                    ),
                    "response": "Q",
                },
                {
                    "stimulus": (
                        "Participant views both designs on a phone. "
                        "Which design do you prefer for everyday use? You press"
                    ),
                    "response": "Z",
                },
            ],
            "pending_trial": (
                "Participant views both designs on a phone. "
                "Which design do you prefer for everyday use? You press"
            ),
        }
    )

    print("=== UX A/B Test Prediction ===")
    print(
        f"Predicted preference : Design {'A (minimal)' if result.output['predicted_choice'] == 'Q' else 'B (feature-rich)'}"
    )
    print(f"Probability          : {result.output['probability']:.2%}")
    print(f"Alternative          : {result.output['alternatives']}")
    print()
    print("Interpretation: Given the prior preference pattern (2:1 for the")
    print("minimal design), the model predicts the next participant is ~67%")
    print("likely to also prefer the clean, minimal layout.  This aligns")
    print("with well-known UX research on cognitive load and Hick's Law --")
    print("fewer choices per screen generally leads to higher satisfaction.")
    print()
    print("Next step: Run a live A/B test with real users to validate this")
    print("prediction before committing to Design A.")


if __name__ == "__main__":
    asyncio.run(main())
