"""
Shared fixtures for tool-centaur tests.

Provides mock backends, sample prompts, and helper factories so individual
test modules stay focused on their own assertions.
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


# ---------------------------------------------------------------------------
# Re-export key types for convenience
# ---------------------------------------------------------------------------

from amplifier_module_tool_centaur.backends import BackendResult  # noqa: E402


# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------


class MockBackend:
    """A controllable fake backend for testing the tool without a real model.

    Set ``completion`` and ``token_log_probs`` before calling ``predict``
    to control what the tool sees.  All calls are recorded in ``calls``.
    """

    def __init__(
        self,
        completion: str = "Q>>.",
        token_log_probs: dict[str, float] | None = None,
    ) -> None:
        self.completion = completion
        self.token_log_probs = token_log_probs
        self.calls: list[dict[str, Any]] = []
        self.closed: bool = False
        self.healthy: bool = True

    async def predict(self, prompt: str, max_tokens: int = 16) -> BackendResult:
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens})
        return BackendResult(
            completion=self.completion,
            token_log_probs=self.token_log_probs,
            metadata={"backend": "mock"},
        )

    async def close(self) -> None:
        self.closed = True

    async def health_check(self) -> bool:
        return self.healthy


@pytest.fixture
def mock_backend() -> MockBackend:
    """A fresh MockBackend with default completion "Q>>"."""
    return MockBackend()


@pytest.fixture
def mock_backend_with_probs() -> MockBackend:
    """A MockBackend that returns log-probs for Q and Z."""
    return MockBackend(
        completion="Q>>.",
        token_log_probs={
            "Q": math.log(0.73),
            "Z": math.log(0.27),
        },
    )


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_experiment_description() -> str:
    return (
        "You will be presented with choices between two options Q and Z.\n"
        "Please name which option you would prefer by pressing the "
        "corresponding key."
    )


@pytest.fixture
def temporal_discounting_trials() -> list[dict[str, str]]:
    """Two completed temporal-discounting trials."""
    return [
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


@pytest.fixture
def pending_discounting_trial() -> str:
    return (
        "You have the choice between receiving 500$ immediately "
        "(press Q) or receiving 600$ in one year (press Z). You press"
    )


@pytest.fixture
def categorization_trials() -> list[dict[str, str]]:
    """Category-learning trials with feedback."""
    return [
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


@pytest.fixture
def raw_psych101_prompt() -> str:
    """A minimal pre-formatted Psych-101 prompt."""
    return (
        "You will be presented with choices between two options Q and Z.\n"
        "Please name which option you would prefer by pressing the "
        "corresponding key.\n"
        "\n"
        "You have the choice between receiving 500$ immediately "
        "(press Q) or receiving 550$ in one year (press Z). "
        "You press <<Q>>.\n"
        "You have the choice between receiving 500$ immediately "
        "(press Q) or receiving 600$ in one year (press Z). "
        "You press <<"
    )


# ---------------------------------------------------------------------------
# Mock coordinator for mount() tests
# ---------------------------------------------------------------------------


class MockCoordinator:
    """Records what gets mounted without needing the real Amplifier runtime."""

    def __init__(self) -> None:
        self.mounted: list[dict[str, Any]] = []

    async def mount(self, category: str, tool: Any, name: str = "") -> None:
        self.mounted.append({"category": category, "tool": tool, "name": name})


@pytest.fixture
def mock_coordinator() -> MockCoordinator:
    return MockCoordinator()
