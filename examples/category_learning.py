#!/usr/bin/env python3
"""
Category Learning: Simulate a participant learning to classify visual stimuli.

Cognitive science background
----------------------------
Category learning is a core cognitive ability studied since the 1960s.
Experiments like badham2017deficits present participants with stimuli that
vary on multiple dimensions (size, color, shape) and ask them to learn,
through trial-and-error with feedback, which features predict category
membership.

Key findings from this literature:
- Humans use a mix of rule-based and similarity-based strategies.
- Working-memory capacity predicts rule-learning ability.
- Feedback timing matters: immediate feedback helps rule-learning while
  delayed feedback may favour implicit similarity-based learning.

The Centaur model, trained on thousands of category-learning transcripts,
captures these dynamics.  Given a sequence of trials with feedback, it
predicts what a typical participant would choose next -- including realistic
early-stage errors and gradual improvement.

This example
------------
We simulate a category-learning experiment where stimuli are described by
three binary features (size, color, shape) and the true rule is "big = K,
small = E".  The participant has seen 5 trials with feedback and we predict
trial 6.

Usage:
    python examples/category_learning.py
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
    """Simulates a Centaur prediction for category learning.

    After 5 trials with feedback, the model should show partial learning
    of the "big = K" rule but still make some errors -- realistic for
    early-stage human performance.
    """

    async def predict(self, prompt: str, max_tokens: int = 16) -> BackendResult:
        return BackendResult(
            completion="K>>.",
            token_log_probs={
                "K": math.log(0.72),  # Learned the "big = K" pattern
                "E": math.log(0.28),  # Some residual uncertainty
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

    # Build a learning trajectory: participant gradually discovers the rule.
    # True rule: big -> K, small -> E (shape and color are irrelevant).
    result = await tool.execute(
        {
            "experiment_description": (
                "In this experiment you will learn to classify shapes into "
                "two categories: K and E.  On each trial you will see a shape "
                "described by its size (big or small), color (black or white), "
                "and form (square, triangle, or circle).  Press K or E to "
                "classify it.  After each response you will see the correct "
                "answer."
            ),
            "prior_trials": [
                {
                    "stimulus": "You see a big black square. You press",
                    "response": "K",
                    "feedback": "The correct category is K.",
                },
                {
                    "stimulus": "You see a small black triangle. You press",
                    "response": "K",
                    "feedback": "The correct category is E.",
                },
                {
                    "stimulus": "You see a big white triangle. You press",
                    "response": "E",
                    "feedback": "The correct category is K.",
                },
                {
                    "stimulus": "You see a small white circle. You press",
                    "response": "E",
                    "feedback": "The correct category is E.",
                },
                {
                    "stimulus": "You see a big black circle. You press",
                    "response": "K",
                    "feedback": "The correct category is K.",
                },
            ],
            "pending_trial": "You see a big white square. You press",
        }
    )

    print("=== Category Learning Prediction ===")
    print("Stimulus         : big white square")
    print("True category    : K (rule: big = K)")
    print(f"Predicted choice : {result.output['predicted_choice']}")
    print(f"Probability      : {result.output['probability']:.2%}")
    print(f"Alternatives     : {result.output['alternatives']}")
    print()
    print("Learning trajectory:")
    print("  Trial 1: big black square   -> K (correct)  -- lucky guess")
    print("  Trial 2: small black triangle -> K (wrong)  -- hasn't learned yet")
    print("  Trial 3: big white triangle -> E (wrong)    -- trying color rule?")
    print("  Trial 4: small white circle -> E (correct)  -- starting to learn")
    print("  Trial 5: big black circle   -> K (correct)  -- size rule emerging")
    print("  Trial 6: big white square   -> K (predicted) -- 72% confident")
    print()
    print("The model captures realistic human learning: after 5 trials the")
    print("participant has partially learned the 'big = K' rule but is not")
    print("yet fully confident, consistent with cognitive science findings")
    print("on the speed of rule extraction in category learning.")


if __name__ == "__main__":
    asyncio.run(main())
