"""
Tests for Psych-101 experiment types against the Centaur tool module.

Validates that the tool correctly handles the prompt formats used across the
major experiment categories in the Psych-101 dataset (160+ experiments,
60,000+ participants).  Each experiment class tests three layers:

1. **Structured prompt** -- build_prompt() produces the correct << >> format
2. **Raw prompt passthrough** -- build_prompt_raw() accepts real dataset text
3. **Full pipeline** -- structured input -> prompt -> MockBackend -> parsed output

Experiment types covered (with representative dataset references):
- Category Learning         (badham2017deficits)
- Temporal Discounting      (ruggeri2022globalizability)
- Risky Choice              (frey2017risk, wulff2018description)
- Multi-Armed Bandits       (wilson2014humans)
- Odd-One-Out Similarity    (hebart2023things)
- Columbia Card Task         (frey2017cct)
- Working Memory / N-back   (enkavi2019adaptivenback, enkavi2019digitspan)
- Go/No-Go                  (enkavi2019gonogo)
- Dunning-Kruger Self-Assessment (jansen2021dunningkruger)
- Reinforcement Learning    (gershman2018deconstructing, kool2016when)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Ensure the module package is importable without installing it.
# ---------------------------------------------------------------------------

_MODULE_ROOT = (
    Path(__file__).resolve().parent.parent.parent / "modules" / "tool-centaur"
)
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))

from amplifier_module_tool_centaur import CentaurPredictTool  # noqa: E402
from amplifier_module_tool_centaur.prompt import (  # noqa: E402
    CHOICE_CLOSE,
    CHOICE_OPEN,
    Trial,
    build_prompt,
    build_prompt_raw,
    parse_completion,
)


# =========================================================================
# Category Learning (badham2017deficits)
# =========================================================================


class TestCategoryLearning:
    """Category-learning experiments present visual stimuli and ask the
    participant to classify them.  Feedback reveals the correct category
    after each trial, enabling learning over time.

    Dataset format (badham2017deficits):
        "You see a big black square. You press <<K>>.
         The correct category is K."
    """

    DESCRIPTION = (
        "In each trial you will see a shape described by its size, color, "
        "and form. You must classify it into category K or category E by "
        "pressing the corresponding key."
    )

    PRIOR_TRIALS = [
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
    ]

    PENDING = "You see a small white square. You press"

    RAW_PROMPT = (
        "In each trial you will see a shape described by its size, color, "
        "and form. You must classify it into category K or category E by "
        "pressing the corresponding key.\n"
        "You see a big black square. You press <<K>>. "
        "The correct category is K.\n"
        "You see a small black triangle. You press <<K>>. "
        "The correct category is E.\n"
        "You see a big white triangle. You press <<E>>. "
        "The correct category is K.\n"
        "You see a small white square. You press <<"
    )

    def test_structured_prompt_format(self) -> None:
        """build_prompt produces correct << >> markers with feedback."""
        prompt = build_prompt(
            experiment_description=self.DESCRIPTION,
            prior_trials=self.PRIOR_TRIALS,
            pending_trial=self.PENDING,
        )

        assert prompt.startswith("In each trial")
        assert "<<K>>. The correct category is K." in prompt
        assert "<<K>>. The correct category is E." in prompt
        assert "<<E>>. The correct category is K." in prompt
        assert prompt.endswith(CHOICE_OPEN)

    def test_raw_prompt_passthrough(self) -> None:
        """A real badham2017-style prompt passes validation."""
        result = build_prompt_raw(self.RAW_PROMPT)

        assert CHOICE_OPEN in result
        assert "<<K>>." in result
        assert result.endswith(CHOICE_OPEN)

    @pytest.mark.asyncio
    async def test_full_pipeline(self, mock_backend: Any) -> None:
        """Structured input flows through the tool to a prediction."""
        mock_backend.completion = "K>>."
        mock_backend.token_log_probs = {
            "K": math.log(0.65),
            "E": math.log(0.35),
        }
        tool = CentaurPredictTool(backend=mock_backend)
        result = await tool.execute(
            {
                "experiment_description": self.DESCRIPTION,
                "prior_trials": self.PRIOR_TRIALS,
                "pending_trial": self.PENDING,
            }
        )

        assert result.success is True
        assert result.output["predicted_choice"] == "K"
        assert abs(result.output["probability"] - 0.65) < 0.01
        assert "E" in result.output["alternatives"]

        prompt_sent = mock_backend.calls[0]["prompt"]
        assert "<<K>>." in prompt_sent
        assert prompt_sent.endswith(CHOICE_OPEN)


# =========================================================================
# Temporal Discounting (ruggeri2022globalizability)
# =========================================================================


class TestTemporalDiscounting:
    """Temporal discounting measures how much a participant devalues future
    rewards.  Each trial offers an immediate vs. delayed monetary option.

    Dataset format (ruggeri2022globalizability):
        "You have the choice between receiving 500$ immediately (press Q)
         or receiving 550$ in one year (press Z). You press <<Q>>."
    """

    DESCRIPTION = (
        "You will be presented with choices between two options Q and Z.\n"
        "Please name which option you would prefer by pressing the "
        "corresponding key."
    )

    PRIOR_TRIALS = [
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
    ]

    PENDING = (
        "You have the choice between receiving 500$ immediately "
        "(press Q) or receiving 600$ in one year (press Z). You press"
    )

    RAW_PROMPT = (
        "You will be presented with choices between two options Q and Z.\n"
        "Please name which option you would prefer by pressing the "
        "corresponding key.\n"
        "You have the choice between receiving 500$ immediately "
        "(press Q) or receiving 550$ in one year (press Z). "
        "You press <<Q>>.\n"
        "You have the choice between receiving 500$ immediately "
        "(press Q) or receiving 700$ in one year (press Z). "
        "You press <<Z>>.\n"
        "You have the choice between receiving 500$ immediately "
        "(press Q) or receiving 600$ in one year (press Z). "
        "You press <<"
    )

    def test_structured_prompt_format(self) -> None:
        """build_prompt produces immediate-vs-delayed choice markers."""
        prompt = build_prompt(
            experiment_description=self.DESCRIPTION,
            prior_trials=self.PRIOR_TRIALS,
            pending_trial=self.PENDING,
        )

        assert "<<Q>>." in prompt
        assert "<<Z>>." in prompt
        assert "500$" in prompt
        assert prompt.endswith(CHOICE_OPEN)

    def test_raw_prompt_passthrough(self) -> None:
        """A real ruggeri2022-style prompt passes validation."""
        result = build_prompt_raw(self.RAW_PROMPT)

        assert "<<Q>>." in result
        assert "<<Z>>." in result
        assert result.endswith(CHOICE_OPEN)

    @pytest.mark.asyncio
    async def test_full_pipeline(self, mock_backend: Any) -> None:
        """Model predicts preference for immediate vs. delayed reward."""
        mock_backend.completion = "Q>>."
        mock_backend.token_log_probs = {
            "Q": math.log(0.62),
            "Z": math.log(0.38),
        }
        tool = CentaurPredictTool(backend=mock_backend)
        result = await tool.execute(
            {
                "experiment_description": self.DESCRIPTION,
                "prior_trials": self.PRIOR_TRIALS,
                "pending_trial": self.PENDING,
            }
        )

        assert result.success is True
        assert result.output["predicted_choice"] == "Q"
        assert abs(result.output["probability"] - 0.62) < 0.01
        assert "Z" in result.output["alternatives"]


# =========================================================================
# Risky Choice (frey2017risk, wulff2018description)
# =========================================================================


class TestRiskyChoice:
    """Risky-choice experiments present gambles with known probabilities.
    Participants choose between a safe option and a risky lottery.

    Dataset format (frey2017risk / wulff2018description):
        "Option Q: 50% chance of $100, 50% chance of $0.
         Option Z: $40 for sure.
         You press <<Z>>."
    """

    DESCRIPTION = (
        "In each round you will see two options. One is a sure amount and "
        "the other is a gamble. Press Q for the gamble or Z for the sure "
        "thing."
    )

    PRIOR_TRIALS = [
        {
            "stimulus": (
                "Option Q: 50% chance of $100, 50% chance of $0. "
                "Option Z: $40 for sure. You press"
            ),
            "response": "Z",
        },
        {
            "stimulus": (
                "Option Q: 80% chance of $200, 20% chance of $0. "
                "Option Z: $120 for sure. You press"
            ),
            "response": "Q",
        },
    ]

    PENDING = (
        "Option Q: 30% chance of $500, 70% chance of $0. "
        "Option Z: $100 for sure. You press"
    )

    RAW_PROMPT = (
        "In each round you will see two options. One is a sure amount and "
        "the other is a gamble. Press Q for the gamble or Z for the sure "
        "thing.\n"
        "Option Q: 50% chance of $100, 50% chance of $0. "
        "Option Z: $40 for sure. You press <<Z>>.\n"
        "Option Q: 80% chance of $200, 20% chance of $0. "
        "Option Z: $120 for sure. You press <<Q>>.\n"
        "Option Q: 30% chance of $500, 70% chance of $0. "
        "Option Z: $100 for sure. You press <<"
    )

    def test_structured_prompt_format(self) -> None:
        """build_prompt encodes gamble parameters and choices."""
        prompt = build_prompt(
            experiment_description=self.DESCRIPTION,
            prior_trials=self.PRIOR_TRIALS,
            pending_trial=self.PENDING,
        )

        assert "<<Z>>." in prompt
        assert "<<Q>>." in prompt
        assert "50% chance" in prompt
        assert prompt.endswith(CHOICE_OPEN)

    def test_raw_prompt_passthrough(self) -> None:
        """A real frey2017-style risky-choice prompt passes validation."""
        result = build_prompt_raw(self.RAW_PROMPT)

        assert "<<Z>>." in result
        assert "<<Q>>." in result
        assert result.endswith(CHOICE_OPEN)

    @pytest.mark.asyncio
    async def test_full_pipeline(self, mock_backend: Any) -> None:
        """Model predicts safe vs. risky choice."""
        mock_backend.completion = "Z>>."
        mock_backend.token_log_probs = {
            "Z": math.log(0.71),
            "Q": math.log(0.29),
        }
        tool = CentaurPredictTool(backend=mock_backend)
        result = await tool.execute(
            {
                "experiment_description": self.DESCRIPTION,
                "prior_trials": self.PRIOR_TRIALS,
                "pending_trial": self.PENDING,
            }
        )

        assert result.success is True
        assert result.output["predicted_choice"] == "Z"
        assert abs(result.output["probability"] - 0.71) < 0.01
        assert "Q" in result.output["alternatives"]


# =========================================================================
# Multi-Armed Bandits (wilson2014humans)
# =========================================================================


class TestMultiArmedBandits:
    """Bandit tasks study explore-exploit trade-offs.  Participants choose
    between slot machines with unknown reward distributions and receive
    point feedback.

    Dataset format (wilson2014humans):
        "You chose machine <<Q>>. Machine Q gave 62 points.
         Machine Z gave 45 points."
    """

    DESCRIPTION = (
        "You will play a slot machine game. In each round, choose machine Q "
        "or machine Z by pressing the corresponding key. Each machine gives "
        "a random number of points. Try to earn as many points as possible."
    )

    PRIOR_TRIALS = [
        {
            "stimulus": "You chose machine",
            "response": "Q",
            "feedback": "Machine Q gave 62 points. Machine Z gave 45 points.",
        },
        {
            "stimulus": "You chose machine",
            "response": "Q",
            "feedback": "Machine Q gave 58 points. Machine Z gave 51 points.",
        },
        {
            "stimulus": "You chose machine",
            "response": "Z",
            "feedback": "Machine Q gave 55 points. Machine Z gave 70 points.",
        },
    ]

    PENDING = "You chose machine"

    RAW_PROMPT = (
        "You will play a slot machine game. In each round, choose machine Q "
        "or machine Z by pressing the corresponding key. Each machine gives "
        "a random number of points. Try to earn as many points as possible.\n"
        "You chose machine <<Q>>. "
        "Machine Q gave 62 points. Machine Z gave 45 points.\n"
        "You chose machine <<Q>>. "
        "Machine Q gave 58 points. Machine Z gave 51 points.\n"
        "You chose machine <<Z>>. "
        "Machine Q gave 55 points. Machine Z gave 70 points.\n"
        "You chose machine <<"
    )

    def test_structured_prompt_format(self) -> None:
        """build_prompt encodes bandit choice + reward feedback."""
        prompt = build_prompt(
            experiment_description=self.DESCRIPTION,
            prior_trials=self.PRIOR_TRIALS,
            pending_trial=self.PENDING,
        )

        assert "<<Q>>." in prompt
        assert "<<Z>>." in prompt
        assert "62 points" in prompt
        assert "70 points" in prompt
        assert prompt.endswith(CHOICE_OPEN)

    def test_raw_prompt_passthrough(self) -> None:
        """A real wilson2014-style bandit prompt passes validation."""
        result = build_prompt_raw(self.RAW_PROMPT)

        assert "<<Q>>." in result
        assert "<<Z>>." in result
        assert "Machine Q gave 62 points" in result
        assert result.endswith(CHOICE_OPEN)

    @pytest.mark.asyncio
    async def test_full_pipeline(self, mock_backend: Any) -> None:
        """Model predicts explore vs. exploit in a bandit task."""
        mock_backend.completion = "Z>>."
        mock_backend.token_log_probs = {
            "Z": math.log(0.55),
            "Q": math.log(0.45),
        }
        tool = CentaurPredictTool(backend=mock_backend)
        result = await tool.execute(
            {
                "experiment_description": self.DESCRIPTION,
                "prior_trials": self.PRIOR_TRIALS,
                "pending_trial": self.PENDING,
            }
        )

        assert result.success is True
        assert result.output["predicted_choice"] == "Z"
        assert abs(result.output["probability"] - 0.55) < 0.01

        prompt_sent = mock_backend.calls[0]["prompt"]
        assert "Machine Q gave 62 points" in prompt_sent


# =========================================================================
# Odd-One-Out Similarity (hebart2023things)
# =========================================================================


class TestOddOneOutSimilarity:
    """The odd-one-out task presents three objects and asks which is least
    similar.  This probes the participant's internal similarity space.

    Dataset format (hebart2023things):
        "D: piecrust, P: game, and H: bracelet. You press <<D>>."
    """

    DESCRIPTION = (
        "In each trial you will see three objects labelled with single "
        "letters. Press the letter of the object that is least similar to "
        "the other two."
    )

    PRIOR_TRIALS = [
        {
            "stimulus": "D: piecrust, P: game, and H: bracelet. You press",
            "response": "D",
        },
        {
            "stimulus": "D: guitar, P: piano, and H: hammer. You press",
            "response": "H",
        },
        {
            "stimulus": "D: car, P: bicycle, and H: airplane. You press",
            "response": "H",
        },
    ]

    PENDING = "D: sunflower, P: cactus, and H: telephone. You press"

    RAW_PROMPT = (
        "In each trial you will see three objects labelled with single "
        "letters. Press the letter of the object that is least similar to "
        "the other two.\n"
        "D: piecrust, P: game, and H: bracelet. You press <<D>>.\n"
        "D: guitar, P: piano, and H: hammer. You press <<H>>.\n"
        "D: car, P: bicycle, and H: airplane. You press <<H>>.\n"
        "D: sunflower, P: cactus, and H: telephone. You press <<"
    )

    def test_structured_prompt_format(self) -> None:
        """build_prompt encodes triplet items with letter labels."""
        prompt = build_prompt(
            experiment_description=self.DESCRIPTION,
            prior_trials=self.PRIOR_TRIALS,
            pending_trial=self.PENDING,
        )

        assert "<<D>>." in prompt
        assert "<<H>>." in prompt
        assert "piecrust" in prompt
        assert "telephone" in prompt
        assert prompt.endswith(CHOICE_OPEN)

    def test_raw_prompt_passthrough(self) -> None:
        """A real hebart2023-style similarity prompt passes validation."""
        result = build_prompt_raw(self.RAW_PROMPT)

        assert "<<D>>." in result
        assert "<<H>>." in result
        assert result.endswith(CHOICE_OPEN)

    @pytest.mark.asyncio
    async def test_full_pipeline(self, mock_backend: Any) -> None:
        """Model predicts the odd-one-out from a semantic triplet."""
        mock_backend.completion = "H>>."
        mock_backend.token_log_probs = {
            "H": math.log(0.78),
            "D": math.log(0.14),
            "P": math.log(0.08),
        }
        tool = CentaurPredictTool(backend=mock_backend)
        result = await tool.execute(
            {
                "experiment_description": self.DESCRIPTION,
                "prior_trials": self.PRIOR_TRIALS,
                "pending_trial": self.PENDING,
            }
        )

        assert result.success is True
        assert result.output["predicted_choice"] == "H"
        assert abs(result.output["probability"] - 0.78) < 0.01
        assert "D" in result.output["alternatives"]
        assert "P" in result.output["alternatives"]


# =========================================================================
# Columbia Card Task (frey2017cct)
# =========================================================================


class TestColumbiaCardTask:
    """The Columbia Card Task (CCT) is a sequential risk-taking measure.
    Participants turn over cards that may contain gains or losses,
    deciding after each card whether to continue or stop.

    Dataset format (frey2017cct):
        "There are 32 cards: 3 loss cards and 29 gain cards.
         Gain amount: 10.  Loss amount: 750.
         You have turned over 5 cards so far. You press <<turn>>."
    """

    DESCRIPTION = (
        "You will play a card game. On each round there is a deck of cards. "
        "Some cards give you points (gain cards) and some take points away "
        "(loss cards). You can press 'turn' to turn over another card or "
        "'stop' to collect your earnings."
    )

    PRIOR_TRIALS = [
        {
            "stimulus": (
                "There are 32 cards: 3 loss cards and 29 gain cards. "
                "Gain amount: 10. Loss amount: 750. "
                "You have turned over 0 cards so far. You press"
            ),
            "response": "turn",
        },
        {
            "stimulus": (
                "There are 32 cards: 3 loss cards and 29 gain cards. "
                "Gain amount: 10. Loss amount: 750. "
                "You have turned over 1 cards so far. You press"
            ),
            "response": "turn",
        },
        {
            "stimulus": (
                "There are 32 cards: 3 loss cards and 29 gain cards. "
                "Gain amount: 10. Loss amount: 750. "
                "You have turned over 2 cards so far. You press"
            ),
            "response": "stop",
        },
    ]

    PENDING = (
        "There are 32 cards: 3 loss cards and 29 gain cards. "
        "Gain amount: 10. Loss amount: 750. "
        "You have turned over 5 cards so far. You press"
    )

    RAW_PROMPT = (
        "You will play a card game. On each round there is a deck of cards. "
        "Some cards give you points (gain cards) and some take points away "
        "(loss cards). You can press 'turn' to turn over another card or "
        "'stop' to collect your earnings.\n"
        "There are 32 cards: 3 loss cards and 29 gain cards. "
        "Gain amount: 10. Loss amount: 750. "
        "You have turned over 0 cards so far. You press <<turn>>.\n"
        "There are 32 cards: 3 loss cards and 29 gain cards. "
        "Gain amount: 10. Loss amount: 750. "
        "You have turned over 1 cards so far. You press <<turn>>.\n"
        "There are 32 cards: 3 loss cards and 29 gain cards. "
        "Gain amount: 10. Loss amount: 750. "
        "You have turned over 2 cards so far. You press <<stop>>.\n"
        "There are 32 cards: 3 loss cards and 29 gain cards. "
        "Gain amount: 10. Loss amount: 750. "
        "You have turned over 5 cards so far. You press <<"
    )

    def test_structured_prompt_format(self) -> None:
        """build_prompt encodes card-turning decisions."""
        prompt = build_prompt(
            experiment_description=self.DESCRIPTION,
            prior_trials=self.PRIOR_TRIALS,
            pending_trial=self.PENDING,
        )

        assert "<<turn>>." in prompt
        assert "<<stop>>." in prompt
        assert "loss cards" in prompt
        assert "gain cards" in prompt
        assert prompt.endswith(CHOICE_OPEN)

    def test_raw_prompt_passthrough(self) -> None:
        """A real frey2017cct-style prompt passes validation."""
        result = build_prompt_raw(self.RAW_PROMPT)

        assert "<<turn>>." in result
        assert "<<stop>>." in result
        assert result.endswith(CHOICE_OPEN)

    @pytest.mark.asyncio
    async def test_full_pipeline(self, mock_backend: Any) -> None:
        """Model predicts stop vs. continue in a card task."""
        mock_backend.completion = "stop>>."
        tool = CentaurPredictTool(backend=mock_backend)
        result = await tool.execute(
            {
                "experiment_description": self.DESCRIPTION,
                "prior_trials": self.PRIOR_TRIALS,
                "pending_trial": self.PENDING,
            }
        )

        assert result.success is True
        assert result.output["predicted_choice"] == "stop"

        prompt_sent = mock_backend.calls[0]["prompt"]
        assert "3 loss cards" in prompt_sent
        assert prompt_sent.endswith(CHOICE_OPEN)


# =========================================================================
# Working Memory / N-back (enkavi2019adaptivenback, enkavi2019digitspan)
# =========================================================================


class TestWorkingMemoryNBack:
    """N-back tasks test working memory.  Participants see a sequence of
    stimuli and press 'match' if the current item is the same as the one
    N steps back, or 'no_match' otherwise.

    Dataset format (enkavi2019adaptivenback):
        "The current letter is G.  Two letters ago was G.
         You press <<match>>."
    Digit-span (enkavi2019digitspan):
        "Remember these digits: 7 3 9. You type <<739>>."
    """

    DESCRIPTION = (
        "This is a 2-back task. You will see letters one at a time. "
        "Press 'match' if the current letter is the same as the letter "
        "shown 2 steps ago, otherwise press 'no_match'."
    )

    PRIOR_TRIALS = [
        {
            "stimulus": ("The current letter is G. Two letters ago was X. You press"),
            "response": "no_match",
        },
        {
            "stimulus": ("The current letter is T. Two letters ago was G. You press"),
            "response": "no_match",
        },
        {
            "stimulus": ("The current letter is G. Two letters ago was G. You press"),
            "response": "match",
        },
    ]

    PENDING = "The current letter is T. Two letters ago was T. You press"

    RAW_PROMPT = (
        "This is a 2-back task. You will see letters one at a time. "
        "Press 'match' if the current letter is the same as the letter "
        "shown 2 steps ago, otherwise press 'no_match'.\n"
        "The current letter is G. Two letters ago was X. "
        "You press <<no_match>>.\n"
        "The current letter is T. Two letters ago was G. "
        "You press <<no_match>>.\n"
        "The current letter is G. Two letters ago was G. "
        "You press <<match>>.\n"
        "The current letter is T. Two letters ago was T. "
        "You press <<"
    )

    def test_structured_prompt_format(self) -> None:
        """build_prompt encodes n-back match/no_match decisions."""
        prompt = build_prompt(
            experiment_description=self.DESCRIPTION,
            prior_trials=self.PRIOR_TRIALS,
            pending_trial=self.PENDING,
        )

        assert "<<no_match>>." in prompt
        assert "<<match>>." in prompt
        assert "Two letters ago" in prompt
        assert prompt.endswith(CHOICE_OPEN)

    def test_raw_prompt_passthrough(self) -> None:
        """A real enkavi2019-style n-back prompt passes validation."""
        result = build_prompt_raw(self.RAW_PROMPT)

        assert "<<match>>." in result
        assert "<<no_match>>." in result
        assert result.endswith(CHOICE_OPEN)

    @pytest.mark.asyncio
    async def test_full_pipeline(self, mock_backend: Any) -> None:
        """Model predicts match vs. no_match in an n-back task."""
        mock_backend.completion = "match>>."
        tool = CentaurPredictTool(backend=mock_backend)
        result = await tool.execute(
            {
                "experiment_description": self.DESCRIPTION,
                "prior_trials": self.PRIOR_TRIALS,
                "pending_trial": self.PENDING,
            }
        )

        assert result.success is True
        assert result.output["predicted_choice"] == "match"

        prompt_sent = mock_backend.calls[0]["prompt"]
        assert "<<no_match>>." in prompt_sent
        assert "<<match>>." in prompt_sent


class TestDigitSpan:
    """Digit-span tests present a sequence of digits and ask the
    participant to recall them, measuring working memory capacity.

    Dataset format (enkavi2019digitspan):
        "Remember these digits: 7 3 9. You type <<739>>."
    """

    DESCRIPTION = (
        "You will see a sequence of digits. After the sequence, type "
        "the digits you remember in order."
    )

    PRIOR_TRIALS = [
        {
            "stimulus": "Remember these digits: 7 3 9. You type",
            "response": "739",
        },
        {
            "stimulus": "Remember these digits: 4 1 8 2. You type",
            "response": "4182",
        },
    ]

    PENDING = "Remember these digits: 5 9 2 6 1. You type"

    RAW_PROMPT = (
        "You will see a sequence of digits. After the sequence, type "
        "the digits you remember in order.\n"
        "Remember these digits: 7 3 9. You type <<739>>.\n"
        "Remember these digits: 4 1 8 2. You type <<4182>>.\n"
        "Remember these digits: 5 9 2 6 1. You type <<"
    )

    def test_structured_prompt_format(self) -> None:
        """build_prompt encodes digit-span recall responses."""
        prompt = build_prompt(
            experiment_description=self.DESCRIPTION,
            prior_trials=self.PRIOR_TRIALS,
            pending_trial=self.PENDING,
        )

        assert "<<739>>." in prompt
        assert "<<4182>>." in prompt
        assert prompt.endswith(CHOICE_OPEN)

    def test_raw_prompt_passthrough(self) -> None:
        """A real digit-span prompt passes validation."""
        result = build_prompt_raw(self.RAW_PROMPT)

        assert "<<739>>." in result
        assert result.endswith(CHOICE_OPEN)

    @pytest.mark.asyncio
    async def test_full_pipeline(self, mock_backend: Any) -> None:
        """Model predicts recalled digit sequence."""
        mock_backend.completion = "59261>>."
        tool = CentaurPredictTool(backend=mock_backend)
        result = await tool.execute(
            {
                "experiment_description": self.DESCRIPTION,
                "prior_trials": self.PRIOR_TRIALS,
                "pending_trial": self.PENDING,
            }
        )

        assert result.success is True
        assert result.output["predicted_choice"] == "59261"


# =========================================================================
# Go/No-Go (enkavi2019gonogo)
# =========================================================================


class TestGoNoGo:
    """Go/No-Go tasks test impulse control.  Participants see a rapid
    stream of stimuli and must press 'go' for targets and withhold
    responses ('no_go') for non-targets.

    Dataset format (enkavi2019gonogo):
        "You see: OX. You press <<go>>."
        "You see: XO. You press <<no_go>>."
    """

    DESCRIPTION = (
        "You will see patterns of letters. Press 'go' when you see the "
        "target pattern OX, and press 'no_go' when you see any other "
        "pattern. Respond as quickly and accurately as possible."
    )

    PRIOR_TRIALS = [
        {
            "stimulus": "You see: OX. You press",
            "response": "go",
        },
        {
            "stimulus": "You see: XO. You press",
            "response": "no_go",
        },
        {
            "stimulus": "You see: OX. You press",
            "response": "go",
        },
        {
            "stimulus": "You see: XX. You press",
            "response": "no_go",
        },
    ]

    PENDING = "You see: OX. You press"

    RAW_PROMPT = (
        "You will see patterns of letters. Press 'go' when you see the "
        "target pattern OX, and press 'no_go' when you see any other "
        "pattern. Respond as quickly and accurately as possible.\n"
        "You see: OX. You press <<go>>.\n"
        "You see: XO. You press <<no_go>>.\n"
        "You see: OX. You press <<go>>.\n"
        "You see: XX. You press <<no_go>>.\n"
        "You see: OX. You press <<"
    )

    def test_structured_prompt_format(self) -> None:
        """build_prompt encodes go/no_go response tokens."""
        prompt = build_prompt(
            experiment_description=self.DESCRIPTION,
            prior_trials=self.PRIOR_TRIALS,
            pending_trial=self.PENDING,
        )

        assert "<<go>>." in prompt
        assert "<<no_go>>." in prompt
        assert prompt.endswith(CHOICE_OPEN)

    def test_raw_prompt_passthrough(self) -> None:
        """A real enkavi2019gonogo-style prompt passes validation."""
        result = build_prompt_raw(self.RAW_PROMPT)

        assert "<<go>>." in result
        assert "<<no_go>>." in result
        assert result.endswith(CHOICE_OPEN)

    @pytest.mark.asyncio
    async def test_full_pipeline(self, mock_backend: Any) -> None:
        """Model predicts go vs. no_go impulse response."""
        mock_backend.completion = "go>>."
        tool = CentaurPredictTool(backend=mock_backend)
        result = await tool.execute(
            {
                "experiment_description": self.DESCRIPTION,
                "prior_trials": self.PRIOR_TRIALS,
                "pending_trial": self.PENDING,
            }
        )

        assert result.success is True
        assert result.output["predicted_choice"] == "go"

        prompt_sent = mock_backend.calls[0]["prompt"]
        assert "<<go>>." in prompt_sent
        assert "<<no_go>>." in prompt_sent


# =========================================================================
# Dunning-Kruger Self-Assessment (jansen2021dunningkruger)
# =========================================================================


class TestDunningKrugerSelfAssessment:
    """Dunning-Kruger experiments ask participants to estimate their
    own performance, measuring metacognitive calibration.

    Dataset format (jansen2021dunningkruger):
        "How many of the 20 questions do you think you will answer
         correctly? You say <<15>>."
    """

    DESCRIPTION = (
        "You will answer 20 general knowledge questions. Before you start, "
        "estimate how many you think you will get right."
    )

    PRIOR_TRIALS = [
        {
            "stimulus": (
                "How many of the 20 questions do you think you will "
                "answer correctly? You say"
            ),
            "response": "15",
        },
    ]

    # After the quiz, ask for a revised estimate.
    PENDING = (
        "Now that you have finished, how many of the 20 questions do you "
        "think you answered correctly? You say"
    )

    RAW_PROMPT = (
        "You will answer 20 general knowledge questions. Before you start, "
        "estimate how many you think you will get right.\n"
        "How many of the 20 questions do you think you will "
        "answer correctly? You say <<15>>.\n"
        "Now that you have finished, how many of the 20 questions do you "
        "think you answered correctly? You say <<"
    )

    def test_structured_prompt_format(self) -> None:
        """build_prompt encodes numeric self-assessment responses."""
        prompt = build_prompt(
            experiment_description=self.DESCRIPTION,
            prior_trials=self.PRIOR_TRIALS,
            pending_trial=self.PENDING,
        )

        assert "<<15>>." in prompt
        assert "20 questions" in prompt
        assert prompt.endswith(CHOICE_OPEN)

    def test_raw_prompt_passthrough(self) -> None:
        """A real jansen2021-style self-assessment prompt passes."""
        result = build_prompt_raw(self.RAW_PROMPT)

        assert "<<15>>." in result
        assert result.endswith(CHOICE_OPEN)

    @pytest.mark.asyncio
    async def test_full_pipeline(self, mock_backend: Any) -> None:
        """Model predicts a numeric self-assessment value."""
        mock_backend.completion = "12>>."
        tool = CentaurPredictTool(backend=mock_backend)
        result = await tool.execute(
            {
                "experiment_description": self.DESCRIPTION,
                "prior_trials": self.PRIOR_TRIALS,
                "pending_trial": self.PENDING,
            }
        )

        assert result.success is True
        assert result.output["predicted_choice"] == "12"

        prompt_sent = mock_backend.calls[0]["prompt"]
        assert "<<15>>." in prompt_sent


# =========================================================================
# Reinforcement Learning (gershman2018deconstructing, kool2016when)
# =========================================================================


class TestReinforcementLearning:
    """Reinforcement learning experiments use two-stage Markov decision
    tasks.  Participants learn state-action-reward contingencies over
    repeated trials with probabilistic transitions.

    Dataset format (gershman2018deconstructing / kool2016when):
        "Stage 1: Choose spaceship Q or spaceship Z.
         You press <<Q>>.
         You arrive at the red planet.
         Choose alien A or alien B. You press <<A>>.
         You received 1 point."
    """

    DESCRIPTION = (
        "You are playing a two-stage decision game. In stage 1, choose a "
        "spaceship (Q or Z). You will then arrive at a planet and choose "
        "between two aliens (A or B). Each alien gives a reward that changes "
        "slowly over time. Try to earn as many points as possible."
    )

    PRIOR_TRIALS = [
        {
            "stimulus": "Stage 1: Choose spaceship Q or spaceship Z. You press",
            "response": "Q",
            "feedback": (
                "You arrive at the red planet. "
                "Choose alien A or alien B. You press A. "
                "You received 1 point."
            ),
        },
        {
            "stimulus": "Stage 1: Choose spaceship Q or spaceship Z. You press",
            "response": "Z",
            "feedback": (
                "You arrive at the blue planet. "
                "Choose alien A or alien B. You press B. "
                "You received 0 points."
            ),
        },
    ]

    PENDING = "Stage 1: Choose spaceship Q or spaceship Z. You press"

    RAW_PROMPT = (
        "You are playing a two-stage decision game. In stage 1, choose a "
        "spaceship (Q or Z). You will then arrive at a planet and choose "
        "between two aliens (A or B). Each alien gives a reward that changes "
        "slowly over time. Try to earn as many points as possible.\n"
        "Stage 1: Choose spaceship Q or spaceship Z. You press <<Q>>. "
        "You arrive at the red planet. "
        "Choose alien A or alien B. You press A. "
        "You received 1 point.\n"
        "Stage 1: Choose spaceship Q or spaceship Z. You press <<Z>>. "
        "You arrive at the blue planet. "
        "Choose alien A or alien B. You press B. "
        "You received 0 points.\n"
        "Stage 1: Choose spaceship Q or spaceship Z. You press <<"
    )

    def test_structured_prompt_format(self) -> None:
        """build_prompt encodes two-stage RL task with reward feedback."""
        prompt = build_prompt(
            experiment_description=self.DESCRIPTION,
            prior_trials=self.PRIOR_TRIALS,
            pending_trial=self.PENDING,
        )

        assert "<<Q>>." in prompt
        assert "<<Z>>." in prompt
        assert "red planet" in prompt
        assert "1 point" in prompt
        assert prompt.endswith(CHOICE_OPEN)

    def test_raw_prompt_passthrough(self) -> None:
        """A real gershman2018-style RL prompt passes validation."""
        result = build_prompt_raw(self.RAW_PROMPT)

        assert "<<Q>>." in result
        assert "<<Z>>." in result
        assert "red planet" in result
        assert result.endswith(CHOICE_OPEN)

    @pytest.mark.asyncio
    async def test_full_pipeline(self, mock_backend: Any) -> None:
        """Model predicts model-based vs. model-free RL behaviour."""
        mock_backend.completion = "Q>>."
        mock_backend.token_log_probs = {
            "Q": math.log(0.68),
            "Z": math.log(0.32),
        }
        tool = CentaurPredictTool(backend=mock_backend)
        result = await tool.execute(
            {
                "experiment_description": self.DESCRIPTION,
                "prior_trials": self.PRIOR_TRIALS,
                "pending_trial": self.PENDING,
            }
        )

        assert result.success is True
        assert result.output["predicted_choice"] == "Q"
        assert abs(result.output["probability"] - 0.68) < 0.01
        assert "Z" in result.output["alternatives"]


# =========================================================================
# Cross-experiment structural tests
# =========================================================================


class TestCrossExperimentProperties:
    """Properties that must hold across all experiment types."""

    @pytest.mark.parametrize(
        "prior_trials,pending",
        [
            # Category learning
            (
                [
                    Trial(
                        "You see a big black square. You press",
                        "K",
                        "The correct category is K.",
                    )
                ],
                "You see a small white circle. You press",
            ),
            # Temporal discounting
            (
                [Trial("Receive 500$ now (Q) or 550$ in 1 year (Z). You press", "Q")],
                "Receive 500$ now (Q) or 800$ in 1 year (Z). You press",
            ),
            # Risky choice
            (
                [Trial("Option Q: 50% of $100. Option Z: $40 sure. You press", "Z")],
                "Option Q: 90% of $50. Option Z: $30 sure. You press",
            ),
            # Bandit
            (
                [Trial("You chose machine", "Q", "Machine Q gave 62 points.")],
                "You chose machine",
            ),
            # Similarity
            (
                [Trial("D: piecrust, P: game, and H: bracelet. You press", "D")],
                "D: dog, P: cat, and H: wrench. You press",
            ),
            # Columbia Card Task
            (
                [
                    Trial(
                        "32 cards: 3 loss, 29 gain. Cards turned: 0. You press", "turn"
                    )
                ],
                "32 cards: 3 loss, 29 gain. Cards turned: 5. You press",
            ),
            # N-back
            (
                [Trial("Current letter: G. Two ago: X. You press", "no_match")],
                "Current letter: T. Two ago: T. You press",
            ),
            # Go/No-Go
            (
                [Trial("You see: OX. You press", "go")],
                "You see: XO. You press",
            ),
            # Dunning-Kruger
            (
                [Trial("How many of 20 questions correct? You say", "15")],
                "Revised estimate? You say",
            ),
            # Reinforcement learning
            (
                [
                    Trial(
                        "Stage 1: spaceship Q or Z. You press",
                        "Q",
                        "Red planet. A. 1 point.",
                    )
                ],
                "Stage 1: spaceship Q or Z. You press",
            ),
        ],
        ids=[
            "category_learning",
            "temporal_discounting",
            "risky_choice",
            "bandit",
            "similarity",
            "columbia_card",
            "nback",
            "go_nogo",
            "dunning_kruger",
            "reinforcement_learning",
        ],
    )
    def test_prompt_always_ends_with_open_marker(
        self, prior_trials: list[Trial], pending: str
    ) -> None:
        """Every experiment type's pending trial ends with <<."""
        prompt = build_prompt(
            experiment_description="Experiment instructions.",
            prior_trials=prior_trials,
            pending_trial=pending,
        )
        assert prompt.endswith(CHOICE_OPEN)

    @pytest.mark.parametrize(
        "prior_trials",
        [
            [Trial("You press", "K", "Correct: K.")],
            [Trial("You press", "Q")],
            [Trial("You press", "turn")],
            [Trial("You press", "go")],
            [Trial("You press", "match")],
            [Trial("You press", "no_match")],
            [Trial("You say", "15")],
            [Trial("You type", "739")],
        ],
        ids=["letter", "key", "turn", "go", "match", "no_match", "numeric", "digits"],
    )
    def test_prior_trial_always_has_closed_markers(
        self, prior_trials: list[Trial]
    ) -> None:
        """Every completed trial has both << and >> around the response."""
        prompt = build_prompt(
            experiment_description="Instructions.",
            prior_trials=prior_trials,
        )
        response = prior_trials[0].response
        assert f"{CHOICE_OPEN}{response}{CHOICE_CLOSE}." in prompt

    @pytest.mark.parametrize(
        "completion,expected_choice",
        [
            ("Q>>.", "Q"),
            ("Z>>.", "Z"),
            ("K>>.", "K"),
            ("turn>>.", "turn"),
            ("stop>>.", "stop"),
            ("go>>.", "go"),
            ("no_go>>.", "no_go"),
            ("match>>.", "match"),
            ("no_match>>.", "no_match"),
            ("15>>.", "15"),
            ("739>>.", "739"),
            ("D>>.", "D"),
            ("H>>.", "H"),
        ],
        ids=[
            "Q",
            "Z",
            "K",
            "turn",
            "stop",
            "go",
            "no_go",
            "match",
            "no_match",
            "numeric",
            "digits",
            "D",
            "H",
        ],
    )
    def test_parse_completion_extracts_all_token_types(
        self, completion: str, expected_choice: str
    ) -> None:
        """parse_completion handles all response token types."""
        preds = parse_completion(completion)
        assert len(preds) == 1
        assert preds[0].choice == expected_choice
        assert preds[0].probability == 1.0

    @pytest.mark.asyncio
    async def test_raw_and_structured_produce_equivalent_prompts(self) -> None:
        """Raw and structured modes for the same data yield matching prompts."""
        structured = build_prompt(
            experiment_description="Choose Q or Z.",
            prior_trials=[
                {"stimulus": "Option A. You press", "response": "Q"},
            ],
            pending_trial="Option B. You press",
        )
        raw = build_prompt_raw(
            "Choose Q or Z.\nOption A. You press <<Q>>.\nOption B. You press <<"
        )
        assert structured == raw


# =========================================================================
# Pipeline tests with configurable mock backend
# =========================================================================


class TestPipelineWithLogProbs:
    """Verify log-prob handling across different experiment response types."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "completion,choice,prob,alt_key,alt_prob",
        [
            ("Q>>.", "Q", 0.73, "Z", 0.27),
            ("K>>.", "K", 0.65, "E", 0.35),
            ("turn>>.", "turn", 0.80, "stop", 0.20),
            ("go>>.", "go", 0.90, "no_go", 0.10),
        ],
        ids=["discounting", "category", "cct", "gonogo"],
    )
    async def test_log_probs_parsed_for_various_tokens(
        self,
        mock_backend: Any,
        completion: str,
        choice: str,
        prob: float,
        alt_key: str,
        alt_prob: float,
    ) -> None:
        """Log-probs are correctly converted for each experiment type."""
        mock_backend.completion = completion
        mock_backend.token_log_probs = {
            choice: math.log(prob),
            alt_key: math.log(alt_prob),
        }

        tool = CentaurPredictTool(backend=mock_backend)
        result = await tool.execute(
            {
                "experiment_description": "Instructions.",
                "pending_trial": "You press",
            }
        )

        assert result.success is True
        assert result.output["predicted_choice"] == choice
        assert abs(result.output["probability"] - prob) < 0.01
        assert alt_key in result.output["alternatives"]
        assert abs(result.output["alternatives"][alt_key] - alt_prob) < 0.01

    @pytest.mark.asyncio
    async def test_three_way_log_probs(self, mock_backend: Any) -> None:
        """Three-alternative similarity task has all alternatives."""
        mock_backend.completion = "H>>."
        mock_backend.token_log_probs = {
            "H": math.log(0.60),
            "D": math.log(0.25),
            "P": math.log(0.15),
        }

        tool = CentaurPredictTool(backend=mock_backend)
        result = await tool.execute(
            {
                "experiment_description": "Pick the odd one out.",
                "pending_trial": "D: cat, P: dog, H: wrench. You press",
            }
        )

        assert result.success is True
        assert result.output["predicted_choice"] == "H"
        assert "D" in result.output["alternatives"]
        assert "P" in result.output["alternatives"]


# =========================================================================
# Edge cases for experiment prompts
# =========================================================================


class TestExperimentEdgeCases:
    """Edge cases specific to experiment format handling."""

    def test_long_trial_history(self) -> None:
        """A prompt with many prior trials (like a full experiment) works."""
        trials = [
            Trial(
                stimulus=f"Trial {i}: You see stimulus {i}. You press",
                response="Q" if i % 2 == 0 else "Z",
                feedback=f"Correct answer was {'Q' if i % 3 == 0 else 'Z'}.",
            )
            for i in range(50)
        ]
        prompt = build_prompt(
            experiment_description="A long experiment.",
            prior_trials=trials,
            pending_trial="Trial 50: You see stimulus 50. You press",
        )

        assert prompt.count(CHOICE_OPEN) == 51  # 50 prior + 1 pending
        assert prompt.count(CHOICE_CLOSE) == 50  # 50 prior completed
        assert prompt.endswith(CHOICE_OPEN)

    def test_numeric_response_tokens(self) -> None:
        """Multi-digit numeric responses (like Dunning-Kruger) are preserved."""
        prompt = build_prompt(
            experiment_description="Estimate your score.",
            prior_trials=[
                {"stimulus": "How many correct? You say", "response": "15"},
                {"stimulus": "Revised estimate? You say", "response": "12"},
            ],
            pending_trial="Final estimate? You say",
        )

        assert "<<15>>." in prompt
        assert "<<12>>." in prompt

    def test_multi_word_response_tokens(self) -> None:
        """Multi-word tokens like 'no_match' and 'no_go' are preserved."""
        prompt = build_prompt(
            experiment_description="Instructions.",
            prior_trials=[
                {"stimulus": "You press", "response": "no_match"},
                {"stimulus": "You press", "response": "no_go"},
            ],
            pending_trial="You press",
        )

        assert "<<no_match>>." in prompt
        assert "<<no_go>>." in prompt

    def test_trial_objects_and_dicts_mixed(self) -> None:
        """Mixing Trial objects and dicts in the same prompt works."""
        prompt = build_prompt(
            experiment_description="Mixed trial sources.",
            prior_trials=[
                Trial("You press", "Q"),
                {"stimulus": "You press", "response": "Z"},
                Trial("You press", "K", "Feedback."),
            ],
            pending_trial="You press",
        )

        assert "<<Q>>." in prompt
        assert "<<Z>>." in prompt
        assert "<<K>>. Feedback." in prompt

    @pytest.mark.asyncio
    async def test_pipeline_with_no_prior_trials(self, mock_backend: Any) -> None:
        """First trial of an experiment (no context) flows through."""
        mock_backend.completion = "Q>>."
        tool = CentaurPredictTool(backend=mock_backend)
        result = await tool.execute(
            {
                "experiment_description": "Choose Q or Z.",
                "pending_trial": "Which do you prefer? You press",
            }
        )

        assert result.success is True
        assert result.output["predicted_choice"] == "Q"

        prompt_sent = mock_backend.calls[0]["prompt"]
        assert CHOICE_CLOSE not in prompt_sent  # No completed trials
        assert prompt_sent.endswith(CHOICE_OPEN)

    @pytest.mark.asyncio
    async def test_pipeline_raw_prompt_preserves_exact_text(
        self, mock_backend: Any
    ) -> None:
        """Raw prompt mode sends exact text (whitespace-stripped) to backend."""
        raw = (
            "Custom experiment.\n"
            "Trial 1: stimulus. You press <<Q>>.\n"
            "Trial 2: stimulus. You press <<"
        )
        mock_backend.completion = "Z>>."
        tool = CentaurPredictTool(backend=mock_backend)
        result = await tool.execute({"raw_prompt": raw})

        assert result.success is True
        assert mock_backend.calls[0]["prompt"] == raw.strip()
