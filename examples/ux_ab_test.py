#!/usr/bin/env python3
"""
Fictional UX choice task: inspect a mocked next-token distribution.

This example exercises prompt construction and output parsing without a model,
endpoint or real participant. The three prior responses are invented fixture
history. MockBackend supplies the completion and probabilities; they were not
estimated from people or produced by a deployed Centaur model.

The task uses two alternative interface descriptions and arbitrary Q/Z labels.
Its output can help inspect the software contract. It cannot establish design
preference, user satisfaction or a population percentage. A behavioral hypothesis
would need a configured model, label/wording sensitivity checks and a declared
comparison with held-out observations from actual participants.

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
    """Returns fixed synthetic outputs for a software-only plumbing check."""

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
    # Each prior response is invented fixture history, not participant data.
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

    print("=== Fictional UX choice task: mock backend ===")
    print(f"Backend             : {result.output['backend']}")
    print(f"Mock completion     : {result.output['predicted_choice']}")
    probability = result.output['probability']
    probability_text = format(probability, ".4f") if probability is not None else "unavailable"
    print(f"Mock token probability: {probability_text}")
    print(f"Mock alternatives   : {result.output['alternatives']}")
    print()
    print("Interpretation: the mock supplies a Q completion and a next-token")
    print("distribution for this exact fictional prompt. No model ran and the")
    print("prior responses are invented. This is a software check, not a")
    print("calibrated probability about a person or a population preference rate.")
    print()
    print("Next step: configure and identify the actual model, counterbalance")
    print("labels and wording, then compare its predictions with held-out")
    print("observations under a declared participant study before making a")
    print("claim about interface preference.")


if __name__ == "__main__":
    asyncio.run(main())
