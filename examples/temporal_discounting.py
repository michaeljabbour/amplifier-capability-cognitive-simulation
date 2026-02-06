#!/usr/bin/env python3
"""
Temporal Discounting: Predict preference between immediate vs. delayed rewards.

Cognitive science background
----------------------------
Temporal discounting (also called delay discounting) is the tendency for people
to prefer smaller, sooner rewards over larger, later ones.  It is one of the
most robust findings in behavioural economics and is linked to real-world
outcomes like savings behaviour, addiction, and health decisions.

The Psych-101 dataset includes temporal-discounting experiments such as
ruggeri2022globalizability, where participants repeatedly choose between an
immediate payout and a delayed alternative.  The Centaur model, trained on
these transcripts, learns the *typical human* discounting curve and can
predict which option a participant is likely to choose given the amounts and
delay.

This example
------------
We build a short experiment transcript where a simulated participant has
already made two choices, then ask the model to predict the third.  The
MockBackend returns a canned response so this runs without a GPU.

Usage:
    python examples/temporal_discounting.py
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
# MockBackend -- stands in for a real GPU backend
# ---------------------------------------------------------------------------


class MockBackend:
    """Returns a configurable canned response.

    In production you would use RestApiBackend (pointing at a vLLM server)
    or LocalGpuBackend (with an 80 GB+ GPU).
    """

    async def predict(self, prompt: str, max_tokens: int = 16) -> BackendResult:
        # Simulate: the model predicts the participant takes the immediate
        # reward (Q) with ~62% probability.
        return BackendResult(
            completion="Q>>.",
            token_log_probs={
                "Q": math.log(0.62),  # immediate $500
                "Z": math.log(0.38),  # delayed $600
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

    # The participant has seen two trials and made choices:
    #   Trial 1: $500 now vs $550 in 1 year -> chose immediate (Q)
    #   Trial 2: $500 now vs $700 in 1 year -> chose delayed  (Z)
    # Now we predict trial 3: $500 now vs $600 in 1 year.
    result = await tool.execute(
        {
            "experiment_description": (
                "You will be presented with choices between two options Q and Z.\n"
                "Please name which option you would prefer by pressing the "
                "corresponding key."
            ),
            "prior_trials": [
                {
                    "stimulus": (
                        "You have the choice between receiving 500$ immediately "
                        "(press Q) or receiving 550$ in one year (press Z). You press"
                    ),
                    "response": "Q",
                },
                {
                    "stimulus": (
                        "You have the choice between receiving 500$ immediately "
                        "(press Q) or receiving 700$ in one year (press Z). You press"
                    ),
                    "response": "Z",
                },
            ],
            "pending_trial": (
                "You have the choice between receiving 500$ immediately "
                "(press Q) or receiving 600$ in one year (press Z). You press"
            ),
        }
    )

    print("=== Temporal Discounting Prediction ===")
    print(f"Predicted choice : {result.output['predicted_choice']}")
    print(f"Probability      : {result.output['probability']:.2%}")
    print(f"Alternatives     : {result.output['alternatives']}")
    print()
    print("Interpretation: The model predicts the participant will choose")
    print("the immediate $500 over the delayed $600, consistent with typical")
    print("human temporal discounting at moderate delay/amount ratios.")
    print()
    print("--- Prompt sent to model ---")
    print(result.output["prompt_used"])


if __name__ == "__main__":
    asyncio.run(main())
