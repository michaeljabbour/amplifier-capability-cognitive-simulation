# Centaur Model Reference

Technical reference for the Centaur foundation model of human cognition.
This document is loaded ONLY inside the cognition-simulator agent session
(context sink pattern -- keeps the root session lean).

## Model Identity

- **Full name**: Llama-3.1-Centaur-70B
- **Base model**: Meta Llama-3.1-70B
- **Fine-tuning**: LoRA adapter on the Psych-101 dataset
- **Paper**: Binz et al. (2025) "A foundation model to predict and capture human cognition" (Nature)
- **HuggingFace**: `marcelbinz/Llama-3.1-Centaur-70B-adapter` (adapter) / `marcelbinz/Llama-3.1-Centaur-70B` (merged)
- **License**: Llama 3.1 Community License (Apache-2.0 for dataset)

## Psych-101 Dataset

The training data that defines what Centaur can predict.

| Stat | Value |
|------|-------|
| Participants | 60,092 |
| Experiments | 160 |
| Total choices | 10,681,650 |
| Format | Natural language transcripts with `<<` `>>` choice markers |
| Language | English |
| Source | Published psychology experiments |

### Dataset schema

Each row in Psych-101 is a complete participant transcript:

```json
{
  "text": "Full natural language transcript with <<choice>> markers",
  "experiment": "author_year_task_name",
  "participant": "unique_id"
}
```

### Loading the dataset

```python
from datasets import load_dataset
data = load_dataset("marcelbinz/Psych-101")
# data["train"] contains 60,092 participant transcripts
```

## Token Protocol

Centaur's core interface is the `<<` `>>` token pair.

- Human choices are ALWAYS wrapped: `You press <<Q>>.`
- The model predicts what goes INSIDE the markers
- For prediction: end the prompt with an OPEN marker: `You press <<`
- The model completes: `Q>>` (with the choice and closing marker)

### Critical formatting rules

1. Space before `<<`: `You press <<Q>>` (not `You press<<Q>>`)
2. Period after `>>`: `You press <<Q>>.` (matches training data)
3. Newline between trials: each trial is on its own line
4. Instructions come first, then trials in sequence

## Experiment Domains in Psych-101

The 160 experiments span these cognitive domains:

### Decision-Making Under Risk
- Risky choice (gambles, lotteries)
- Description-experience gap
- Iowa Gambling Task variants
- Columbia Card Task

### Temporal Discounting
- Immediate vs delayed rewards
- Hyperbolic discounting paradigms
- Inter-temporal choice

### Reinforcement Learning
- Two-armed bandits
- Multi-armed bandits
- Restless bandits
- Model-based vs model-free tasks
- Explore-exploit tradeoffs

### Categorization
- Rule-based categorization
- Similarity-based categorization
- Category learning with feedback
- Prototype vs exemplar tasks

### Working Memory
- N-back tasks
- Digit span
- Adaptive working memory tasks

### Similarity and Judgment
- Odd-one-out tasks (THINGS dataset)
- Similarity ratings
- Generalization judgments

### Cognitive Control
- Go/No-Go tasks
- Recent probes
- Task switching

## Hardware Requirements

| Method | Model ID | VRAM | Notes |
|--------|----------|------|-------|
| unsloth (recommended) | `marcelbinz/Llama-3.1-Centaur-70B-adapter` | 80 GB | 4-bit quantized adapter |
| transformers | `marcelbinz/Llama-3.1-Centaur-70B` | 160 GB+ | Full merged model, bfloat16 |
| vLLM server | Either | 80-160 GB | Depends on quantization |

## Example Experiments from Psych-101

### Temporal Discounting

```
You will be presented with multiple choices between two options Q and Z.
Please name which option you would prefer by pressing the corresponding key.

You have the choice between receiving 500$ immediately (press Q) or receiving 550$ in one year (press Z). You press <<Q>>.
You have the choice between receiving 500$ immediately (press Q) or receiving 600$ in one year (press Z). You press <<
```

### Category Learning

```
You will be shown several examples of geometric objects.
Your task is to learn a rule that allows you to tell whether an object belongs to the E or K category.
For each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.

You see a big black square. You press <<K>>. The correct category is K.
You see a small black triangle. You press <<K>>. The correct category is E.
You see a big white triangle. You press <<E>>. The correct category is K.
You see a small white triangle. You press <<
```

### Odd-One-Out (Similarity)

```
You will be presented with triplets of objects, which will be assigned to the keys D, P, and H.
In each trial, please indicate which object you think is the odd one out by pressing the corresponding key.

D: piecrust, P: game, and H: bracelet. You press <<D>>.
D: tuning fork, P: rocket, and H: waffle iron. You press <<
```

### Two-Armed Bandit

```
In the following task, you will be repeatedly asked to select between two slot machines, each denoted by a key.
After you select a slot machine, you will see how many points you earned from that machine.

You chose machine <<Q>>. Machine Q gave 62 points and machine Z gave unknown points.
You chose machine <<Q>>. Machine Q gave 58 points and machine Z gave unknown points.
You chose machine <<Z>>. Machine Q gave unknown points and machine Z gave 71 points.
You chose machine <<
```

## Known Experiment Names

A sample of the 160 experiment identifiers in Psych-101 (useful for
filtering the test set or referencing specific paradigms):

- `badham2017deficits` -- Category learning (geometric objects)
- `frey2017risk` -- Risky choice
- `frey2017cct` -- Columbia Card Task
- `gershman2018deconstructing` -- Reinforcement learning
- `hebart2023things` -- Odd-one-out similarity (THINGS)
- `kool2016when` -- Model-based vs model-free RL
- `peterson2021using` -- Risky choice
- `plonsky2018when` -- Description-experience gap
- `schulz2020finding` -- Exploration in function learning
- `steingroever2015data` -- Iowa Gambling Task
- `wilson2014humans` -- Explore-exploit (Horizon Task)
- `wulff2018description` -- Description-based risky choice
- `wulff2018sampling` -- Sampling-based risky choice

## Citation

```bibtex
@article{binz2025foundation,
  title={A foundation model to predict and capture human cognition},
  author={Binz, Marcel and Akata, Elif and Bethge, Matthias and others},
  journal={Nature},
  year={2025},
  doi={10.1038/s41586-025-09215-4}
}
```
