# Cognitive Simulation

Explore a behavioral hypothesis before the study.

Watch the [74-second film](https://michaeljabbour.github.io/amplifier-skill-cognitive-simulation/#film): a model answer mistaken for a customer, and how the skill keeps a prediction from standing in for a study. The example in it is fictional.

Frame a choice or learning task, record what a model predicts, and identify the observations needed to trust that prediction. Useful for preparing research questions and comparing explicit alternatives.

[Explore the site](https://michaeljabbour.github.io/amplifier-skill-cognitive-simulation/) · [Read the method](skills/cognitive-simulation/SKILL.md) · [Install guide](docs/INSTALL.md)

## What you get

- **Make the question testable.** Name the population, task, alternatives and outcome, including the context the simulation leaves out.
- **Keep the prediction reproducible.** Freeze instructions, labels, prior trials, backend identity and analysis before comparing variants.
- **Know what remains to learn.** Separate model choice, available probabilities and actual participant evidence so a synthetic answer stays interpretable.

## Install and try it

Requires Git and Python 3.11+. Run these commands from a terminal:

```sh
git clone https://github.com/michaeljabbour/amplifier-skill-cognitive-simulation.git
cd amplifier-skill-cognitive-simulation
python3 scripts/capability.py doctor
python3 scripts/capability.py install --dest ~/.agents/skills
```

The installer copies `skills/cognitive-simulation` as a complete folder into your chosen skills parent. Use the discovery path supported by your agent. It starts no service and configures no provider. Restart or refresh skill discovery in your host, then ask:

```text
Use Cognitive Simulation to frame a fictional kiosk confirmation study. Compare a single confirm button with a review-then-confirm flow. State task fit, counterbalance response labels, prepare exact prompts and define the observed-user check. Do not run an endpoint.
```

If your agent does not discover installed skills, point it directly at the complete folder and ask it to read `SKILL.md`. Reading the method needs no Python runtime. [Installation details](docs/INSTALL.md) cover conflict handling and optional native execution.

## How the work proceeds

1. **Frame a bounded task.** Translate the decision into an explicit choice or sequence and assess its fit to the available model.
2. **Run or leave a prepared experiment.** Use an authorized compatible endpoint; without one, deliver the exact prompt and blocked inference step.
3. **Compare against observations.** Check label sensitivity and held-out human data before making a population claim.

## Example output

Fictional worked example. It illustrates the intended output; it is not a measured result or a real customer story.

```text
Fictional experiment: kiosk confirmation
Alternatives: confirm / review then confirm
Labels: counterbalanced across variants
Prediction: pending compatible endpoint
Probability: unavailable until returned
Human check: observed errors and task completion under a declared study
```

## What is checked

164 mocked engine regression tests cover prompt/result and backend behavior, including missing probabilities. No model deployment or new inference is part of this release, and no human-behavior accuracy claim is established. See [validation](docs/VALIDATION.md) for the exact checks and limits.

## Retained implementation and references

[Migration map](docs/MIGRATION.md) records the original source paths and current roles. Primary Git history, native engines, tests, examples, and evaluation assets remain in this checkout. [Native guide](docs/NATIVE-README.md) preserves the original project documentation; old install URLs and runtime claims there are historical and need current verification. [Validation](docs/VALIDATION.md) records actual checks and limits.

## Build the site

```sh
python3 scripts/capability.py build
python3 scripts/capability.py check
```

The static site includes a ZIP built from the canonical method, references and license. Runtime and provider setup remain separate from installing the portable method.

## Repository migration

This repository was renamed from `amplifier-skill-cognitive-simulation` to `amplifier-capability-cognitive-simulation`. Existing installed method names, bundle aliases and native interfaces stay the same. See the [migration guide](https://michaeljabbour.github.io/migration/) and [repository map](https://michaeljabbour.github.io/migration/mapping.json) for source URL updates.
