> Original native guide, preserved at source commit `a01fbafef7dfeea7d57d55d45ae8b76a48b6bb74`. Install locations, model names, service availability, and claims below are historical and must be verified before use. Current portable installation is in INSTALL.md.

# Cognitive Research Bundle for Amplifier

**Human behavior prediction via the Centaur foundation model**

This is an [Amplifier](https://github.com/microsoft/amplifier) bundle that integrates the **Centaur** foundation model of human cognition ([Binz et al., Nature 2025](https://www.nature.com/articles/s41586-025-09215-4)). Centaur is Llama-3.1-70B fine-tuned on the **Psych-101** dataset — 60,000+ participants, 160 experiments, 10.6 million choices. It predicts how humans would behave in any experiment expressible in natural language.

---

## What This Adds to Amplifier

Without this bundle, Amplifier can only *theorize* about human behavior using LLM reasoning. With it, Amplifier can make **empirically-grounded predictions** calibrated on real behavioral data.

**Key capabilities:**

- **`centaur_predict` tool** — Sends experiments to the Centaur model and returns choice probabilities
- **`cognition-simulator` agent** — Translates natural language questions into Centaur experiments and interprets results
- **Supports 10+ cognitive domains:** temporal discounting, risky choice, reinforcement learning, category learning, working memory, similarity judgments, and more

---

## Real-World Example: Game Loot Box Pricing

A user asked Amplifier:

> *"I'm designing loot boxes for a mobile game. Compare three designs and tell me which one will maximize repeat purchases."*

The `cognition-simulator` agent decomposed this into three parallel Centaur experiments — each framing one loot box design as a risky-choice paradigm calibrated against Psych-101 gambling tasks. Results:

| Design | Cost | Mechanic | Buy Again? | Probability |
|--------|------|----------|-----------|-------------|
| **A** | 100 gems | Guaranteed rare | Yes | **77.7%** |
| **B** | 50 gems | 40% rare / 60% common | Yes | **81.8%** |
| **C** | 75 gems | 25% legendary / 75% junk | **No** | **62.2% skip** |

**Finding:** Design B (cheap + variable reward) wins. Design C, despite having the biggest potential jackpot, is the only design where players stop buying — loss aversion from 75% junk outcomes overrides the allure of the legendary item.

This is a counterintuitive result that pure LLM reasoning would likely get wrong. An LLM would tend to overweight the "legendary" outcome; Centaur, calibrated on tens of thousands of real human gambles, correctly captures that most people are loss-averse and will walk away from a bet that fails 75% of the time.

**This is what Centaur adds** — empirical behavioral prediction calibrated on real human data, not LLM theorizing. Each prediction took ~500ms against a live 70B parameter model.

---

## Architecture

```
User (natural language)
  → Amplifier (Opus/Claude orchestrates)
    → cognition-simulator agent (frames experiment)
      → centaur_predict tool (structured or raw mode)
        → Centaur model (via REST API / CLI / Local GPU)
          → Returns: predicted choice + probability distribution
```

The bundle has three components:

1. **`centaur_predict` tool** — An Amplifier tool module (`modules/tool-centaur/`) with three pluggable backends: REST API, CLI subprocess, and local GPU via transformers. The tool accepts either structured experiment definitions or raw Psych-101-format prompts.

2. **`cognition-simulator` agent** — A context-sink agent (`agents/cognition-simulator.yaml`) that knows how to translate natural language questions into well-formed Centaur experiments. It selects the right cognitive domain, formats the stimulus, and interprets the probability distributions that come back.

3. **Routing context** — Instruction context (`context/cognitive-research-instructions.md`) that teaches the Amplifier orchestrator when to delegate to the `cognition-simulator` — any time a conversation touches human behavior prediction, decision-making, or cognitive modeling.

---

## Prerequisites: Deploying Centaur

Centaur is a **70B parameter model** that requires GPU infrastructure. You must deploy it yourself before using this bundle.

> **Important:** Centaur uses **`/v1/completions`** (text completion), NOT `/v1/chat/completions`. It has no chat template — it is a completion model trained on raw Psych-101 transcripts with `<<`/`>>` delimiter tokens.

### Option 1: HuggingFace Inference Endpoints (Recommended for getting started)

1. Go to [marcelbinz/Llama-3.1-Centaur-70B](https://huggingface.co/marcelbinz/Llama-3.1-Centaur-70B)
2. Click **"Deploy"** → **"Inference Endpoints"**
3. Select GPU (minimum **A100 80GB**)
4. The endpoint URL goes in your config

### Option 2: vLLM (Recommended for production)

```bash
pip install vllm
vllm serve marcelbinz/Llama-3.1-Centaur-70B --port 8000
```

### Option 3: Text Generation Inference (TGI)

```bash
docker run --gpus all -p 8000:80 \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id marcelbinz/Llama-3.1-Centaur-70B
```

### Option 4: Unsloth (4-bit quantized, 80GB VRAM)

Use the adapter: [`marcelbinz/Llama-3.1-Centaur-70B-adapter`](https://huggingface.co/marcelbinz/Llama-3.1-Centaur-70B-adapter)

---

## Installation

```bash
# 1. Install the tool module into Amplifier's environment
uv pip install amplifier-centaur \
  --python $(head -1 $(which amplifier) | sed 's/#!//')

# 2. Add to your Amplifier settings (~/.amplifier/settings.yaml)
# Under bundle.app, add:
bundle:
  app:
    - git+https://github.com/michaeljabbour/amplifier-centaur@main
```

### Local Development

```bash
# Clone the repo
git clone https://github.com/michaeljabbour/amplifier-centaur.git
cd amplifier-centaur

# Install the tool module
uv pip install -e modules/tool-centaur/ \
  --python $(head -1 $(which amplifier) | sed 's/#!//')

# Add to settings (~/.amplifier/settings.yaml)
bundle:
  app:
    - /path/to/amplifier-centaur
```

---

## Configuration

The behavior file at `behaviors/cognitive-research.yaml` configures the tool:

```yaml
tools:
  - module: tool-centaur
    config:
      backend:
        type: rest_api                    # rest_api | cli | local_gpu
        api_url: http://localhost:8000    # Your Centaur endpoint
        api_key_env: HF_TOKEN            # Env var with your API key
        model_name: marcelbinz/Llama-3.1-Centaur-70B
        timeout: 120
      max_tokens: 16
```

| Field | Description |
|-------|-------------|
| `type` | Backend type. `rest_api` for hosted endpoints, `cli` for subprocess calls, `local_gpu` for in-process transformers |
| `api_url` | URL of your deployed Centaur instance |
| `api_key_env` | Name of the environment variable containing your API key |
| `model_name` | HuggingFace model identifier |
| `timeout` | Request timeout in seconds (Centaur can take a few seconds on complex experiments) |
| `max_tokens` | Maximum tokens to generate (16 is usually sufficient — responses are short choice tokens) |

---

## Supported Experiment Domains

| Domain | Example | Psych-101 Experiments |
|--------|---------|----------------------|
| Temporal Discounting | $100 now vs $200 later | ruggeri2022, somerville2017 |
| Risky Choice | Gamble vs sure thing | frey2017risk, wulff2018 |
| Reinforcement Learning | Multi-armed bandits | wilson2014, gershman2018 |
| Category Learning | Shape classification | badham2017, flesch2018 |
| Working Memory | N-back, digit span | enkavi2019 |
| Similarity Judgments | Odd-one-out | hebart2023 |
| Cognitive Control | Go/No-Go | enkavi2019 |
| Self-Assessment | Dunning-Kruger | jansen2021 |

---

## Try It

Once installed and configured, try these prompts in Amplifier:

```
How would people choose between $100 now and $200 in 6 months?
```

```
Would users prefer a simple checkout or a multi-step wizard?
```

```
Simulate the Iowa Gambling Task for 20 trials
```

---

## Testing

```bash
cd amplifier-centaur
uv run --with httpx --with respx --with pytest --with pytest-asyncio \
  pytest tests/tool-centaur/ -v
# 163 tests, ~0.4 seconds
```

---

## How It Works: The `<<` `>>` Token Protocol

Centaur uses a specialized prompt format inherited from Psych-101. Experiments are presented as sequences of trials where participant responses are delimited by `<<` and `>>` tokens:

```
You will make a series of choices between a smaller, sooner reward
and a larger, later reward.

Do you prefer $10 today or $20 in 30 days? <<$10>>
Do you prefer $5 today or $40 in 90 days? <<$40>>
Do you prefer $100 today or $200 in 180 days? <<
```

The model completes the final `<<` with its predicted human response. The `centaur_predict` tool handles this formatting automatically — you describe experiments in natural language, and the `cognition-simulator` agent translates them into the `<<`/`>>` protocol.

---

## References

- **Paper:** Binz, M., Alaniz, S., Roskies, A. et al. (2025). "A foundation model to predict and capture human cognition." *Nature*. [doi:10.1038/s41586-025-09215-4](https://www.nature.com/articles/s41586-025-09215-4)
- **Model:** [marcelbinz/Llama-3.1-Centaur-70B-adapter](https://huggingface.co/marcelbinz/Llama-3.1-Centaur-70B-adapter)
- **Dataset:** [marcelbinz/Psych-101](https://huggingface.co/datasets/marcelbinz/Psych-101)
- **Amplifier:** [github.com/microsoft/amplifier](https://github.com/microsoft/amplifier)

---

## License

[MIT](https://github.com/michaeljabbour/amplifier-bundle-centaur/blob/a01fbafef7dfeea7d57d55d45ae8b76a48b6bb74/LICENSE)
