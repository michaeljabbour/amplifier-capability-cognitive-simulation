# Cognitive Simulation

Compare a behavioral hypothesis with a model prediction—without calling it a real participant.

Researchers and product teams exploring explicit choice and learning hypotheses. Use when framing a behavioral simulation, interpreting Centaur output, or planning a comparison against real observations.

## Install the portable method

```sh
python3 scripts/capability.py doctor
python3 scripts/capability.py install --dest ~/.agents/skills
```

Keep the complete `skills/cognitive-simulation` folder together. Installation copies only that method, references, and license; it starts no service and installs no runtime. Use a different skills-parent path for your host.

## Choose the mode

- **Frame** — A behavioral question needs a falsifiable experiment. A choice task, task-domain fit assessment, label counterbalance, and a human-validation gap.
- **Simulate** — An authorized model endpoint and exact prompt are available. An attributed model prediction with actual probability availability and a reproducible prompt.
- **Compare** — Observed participant data exist. A preregistered held-out comparison with calibration and domain limits.

## What was consolidated

[Migration map](docs/MIGRATION.md) records the original source paths and current roles. Primary Git history, native engines, tests, examples, and evaluation assets remain in this checkout. [Native guide](docs/NATIVE-README.md) preserves the original project documentation; old install URLs and runtime claims there are historical and need current verification. [Validation](docs/VALIDATION.md) records actual checks and limits.

## Build the local site

```sh
python3 scripts/capability.py build
python3 scripts/capability.py check
```

Open `_site/index.html` or serve `_site` with a local static server. The downloadable ZIP is built from canonical method bytes. The fictional before/after example illustrates intended judgment; it is not a measured improvement. Nothing has been published or renamed remotely.
