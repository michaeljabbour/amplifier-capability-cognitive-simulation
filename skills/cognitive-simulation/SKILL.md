---
name: cognitive-simulation
description: Use when framing a behavioral simulation, interpreting Centaur output, or planning a comparison against real observations.
license: MIT
---

# Cognitive Simulation

Compare a behavioral hypothesis with a model prediction—without calling it a real participant.

## Frame a behavioral question

Identify the decision, population, task, meaningful alternatives, outcome, and consequence. Translate the question into an explicit choice, learning, or sequential-decision task. Explain the fit to the model's training tasks and what social, emotional, organizational, or longitudinal context the task leaves out. If the intended inference needs actual people, keep that study in the plan; simulation does not replace it.

## Construct the experiment

Preserve instructions, choice-key mapping, prior trials, responses, feedback, and the pending stimulus. Keep treatment differences isolated. Counterbalance option order and arbitrary response labels; label sensitivity is a model result, not human variation. Distinguish actual prior observations from invented/synthetic histories. Freeze the exact prompts and analysis rule before comparing variants.

[Prompt and result contract](references/prompt-contract.md) documents the retained engine. [Original experiment framing](references/cognition-simulator.md) and [Centaur reference](references/CENTAUR_REFERENCE.md) preserve the domain patterns. Historical population or capability statements in those documents need verification against the actual model/version and cited study; they are not evaluation of this wrapper.

## Simulate only with an available engine

The optional repository engine retains REST, CLI, and local-GPU backends. Inspect its configuration, the deployed model identity/revision, endpoint protocol, generation settings, and cost limits. Never silently substitute a general chat model and label the result Centaur. Record the exact prompt, completion, alternatives, and backend metadata. A mocked backend is a software check only.

If no endpoint is configured, return the experiment and blocked inference step. Do not invent a prediction. If log-probabilities are absent, report probability as unavailable; never convert a missing distribution into `1.0`. Available next-token probabilities still are not calibrated population choice rates.

## Compare and interpret

Report the model choice and available distribution as predictions. Evaluate sensitivity to wording/labels and task fit. For a claim about human behavior, compare held-out observed participant data under a declared sampling and analysis plan. Do not average repeated deterministic completions into a participant count. Separate model calibration, task-generalization evidence, and this adapter's software tests.

## Output

Return scenario; task fit and omitted context; exact experiment/prompt; backend identity; prediction or blocked status; probability availability; sensitivity checks; what actual observations support; limitations; and the smallest participant-based check that could change the decision. Never say a synthetic preference establishes what a specific person or customer segment will do.
