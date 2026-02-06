# Cognitive Research Bundle -- Design Document

## Overview

The `cognitive-research` bundle integrates the Centaur foundation model of human
cognition into the Amplifier ecosystem. Centaur is Llama-3.1-70B fine-tuned on
the Psych-101 dataset (60,092 participants, 160 experiments, 10.6M choices) and
can predict human behavior in any experiment expressible in natural language.

Published: Binz et al. (2025) "A foundation model to predict and capture human
cognition" -- Nature.

---

## Architecture

```
                    Amplifier Root Session
                           |
                    [routing context]
                    context/cognitive-research-instructions.md
                           |
                    delegates to
                           |
                           v
                  cognition-simulator agent
                  agents/cognition-simulator.md
                  (carries: docs/CENTAUR_REFERENCE.md)
                           |
                    calls tool
                           |
                           v
                  centaur_predict tool
                  modules/tool-centaur/
                           |
              +------------+------------+
              |            |            |
              v            v            v
         RestApiBackend  CliBackend  LocalGpuBackend
         (vLLM/TGI)     (scripts)  (unsloth/transformers)
```

### Key Design Decisions

**1. Single tool, pluggable backends**

The `centaur_predict` tool is the stable interface. Three backends sit behind a
`CentaurBackend` protocol. Swapping backends is a config change -- zero code
changes. This follows Amplifier's "mechanism, not policy" principle.

**2. Prompt module as the critical boundary**

`prompt.py` encapsulates the `<<`/`>>` token protocol completely. Nothing else
in the system knows about Centaur's raw format. All other code works with
structured Python objects (Trial, CentaurPrediction, CentaurResult).

**3. Agent as context sink**

The `cognition-simulator` agent carries the heavy documentation (~400 lines of
prompt patterns, domain knowledge, output contract). The root session sees only
~30 lines of routing instructions. This keeps root sessions lean per the
Amplifier context sink pattern.

**4. Two input modes**

- **Structured**: `experiment_description` + `prior_trials` + `pending_trial`
  -- the agent translates natural language into Centaur format
- **Raw**: `raw_prompt` -- power users pass Psych-101 formatted text directly

**5. Heavy dependencies stay out of pyproject.toml**

`torch`, `transformers`, `unsloth` are NOT package dependencies. They are
environment-specific. The default `rest_api` backend only needs `httpx`. The
bundle installs cleanly everywhere -- you only need a GPU environment if you
choose `local_gpu`.

---

## File Inventory

```
amplifier-centaur/
  bundle.md                                    # Thin root bundle
  pyrightconfig.json                           # Type checker config
  behaviors/
    cognitive-research.yaml                    # Behavior: tool + agent + context
  agents/
    cognition-simulator.md                     # Expert agent (context sink)
  context/
    cognitive-research-instructions.md         # Routing pointer (~30 lines)
  docs/
    CENTAUR_REFERENCE.md                       # Heavy reference (agent-only)
    ORCHESTRATION_EXAMPLES.md                  # Routing patterns and examples
    DESIGN.md                                  # This document
  modules/
    tool-centaur/
      pyproject.toml                           # Package config + entry point
      amplifier_module_tool_centaur/
        __init__.py                            # mount() + CentaurPredictTool
        prompt.py                              # Prompt formatting + parsing
        backends.py                            # Protocol + 3 implementations
  examples/
    temporal_discounting.py                    # Delay aversion prediction
    ux_ab_test.py                              # UX preference prediction
    category_learning.py                       # Learning simulation
    risk_preference.py                         # Gamble vs sure-thing
    replicate_experiment.py                    # Full experiment replication
  tests/
    tool-centaur/
      conftest.py                              # MockBackend, fixtures
      test_prompt.py                           # 18 tests
      test_backends.py                         # 32 tests
      test_tool.py                             # 37 tests
      test_psych101_experiments.py             # 76 tests
  .research/                                   # Raw research materials
```

**163 tests. All passing. All Python clean on ruff + pyright.**

---

## Centaur Model: I/O Specification

### Input Format (Psych-101 Token Protocol)

Centaur uses a specific prompt format derived from the Psych-101 dataset:

1. **Instructions block**: Natural language description of the experiment
2. **Trial sequence**: Each trial on its own line
3. **Choice markers**: Human responses wrapped in `<<` and `>>`
4. **Pending choice**: Final trial ends with open `<<` for model completion

```
[Instructions]

[Trial 1 stimulus] <<[response]>>. [optional feedback]
[Trial 2 stimulus] <<[response]>>. [optional feedback]
...
[Pending trial stimulus] <<
```

**Critical formatting rules:**
- Space before `<<`: `You press <<Q>>` not `You press<<Q>>`
- Period after `>>`: `You press <<Q>>.` (matches training data)
- Newline between trials
- Instructions come first, then trials in sequence

### Output Format

The model completes the open `<<` marker:
- Typical output: `Q>>.` (choice token + closing marker + period)
- With log-probs: probability distribution over candidate tokens

### Token Types Across Experiments

| Domain | Example Tokens | Notes |
|--------|---------------|-------|
| Forced choice | `Q`, `Z`, `A`, `B` | Single letter keys |
| Category learning | `K`, `E` + feedback | With "The correct category is X." |
| Similarity | `D`, `P`, `H` | 3-way choice |
| Card task | `turn`, `stop` | Multi-character tokens |
| Go/No-Go | `go`, `no_go` | Multi-character with underscore |
| Working memory | `match`, `no_match` | Multi-character |
| Self-assessment | `15`, `60` | Numeric tokens |
| Digit span | `739`, `4182` | Multi-digit numeric |

---

## Backend Abstraction

### Protocol

```python
@runtime_checkable
class CentaurBackend(Protocol):
    async def predict(self, prompt: str, max_tokens: int = 16) -> BackendResult: ...
    async def close(self) -> None: ...
    async def health_check(self) -> bool: ...
```

### Backend Comparison

| Backend | Latency | VRAM | Dependencies | Best For |
|---------|---------|------|-------------|----------|
| `rest_api` | 1-5s | 0 (remote) | `httpx` only | Production, shared servers |
| `cli` | 5-30s | varies | None (subprocess) | Existing scripts, HPC |
| `local_gpu` | 2-10s | 80-160 GB | `torch`, `transformers`/`unsloth` | Development, research |

### Adding a New Backend

To support a new inference provider (e.g., Ollama, AWS SageMaker, GGUF):

1. Create a class with `predict`, `close`, and `health_check` methods
2. Add a branch in `create_backend()` for the new `type` string
3. The class does NOT need to inherit from `CentaurBackend` (structural protocol)

Example:

```python
class OllamaBackend:
    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self.base_url = base_url

    async def predict(self, prompt: str, max_tokens: int = 16) -> BackendResult:
        # POST to Ollama's /api/generate
        ...

    async def close(self) -> None:
        pass

    async def health_check(self) -> bool:
        # GET /api/tags
        ...
```

### Configuration

Backend selection is a config-time decision:

```yaml
# In settings.yaml or behavior YAML
tools:
  - module: tool-centaur
    config:
      backend:
        type: rest_api          # or "cli" or "local_gpu"
        api_url: http://gpu-box:8000
        api_key_env: CENTAUR_API_KEY
```

API keys are ALWAYS resolved from environment variables, never stored in config.

---

## Limitations

### Model Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|-----------|
| **Population bias** | Training data is predominantly Western, English-speaking adults (Psych-101 sources are mostly US/European university studies) | Always note population context in predictions; do not generalize to other demographics |
| **Aggregate predictions** | Centaur predicts average behavior, not individual differences | Frame predictions as "typical human" not "this specific person" |
| **No emotional modeling** | Stress, fatigue, social pressure not captured | Note when scenarios involve strong emotional components |
| **Task structure dependence** | Novel structures far from Psych-101 may reduce accuracy | Stay close to established paradigms; note uncertainty for novel tasks |
| **Context window** | Long trial histories (100+ trials) may hit the 32K token limit | Truncate or summarize early trials; use representative subsets |
| **Temperature sensitivity** | Model runs at temperature ~0 for deterministic predictions | Log-probs capture uncertainty; single-token predictions may miss multimodal distributions |

### Infrastructure Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|-----------|
| **GPU requirements** | 80 GB minimum (adapter) or 160 GB (full model) for local inference | Use REST API backend with a shared GPU server |
| **Latency** | 2-30 seconds per prediction depending on backend and prompt length | Batch predictions when possible; cache results |
| **No streaming** | Centaur predictions are 1-3 tokens; streaming has no benefit | Single-shot inference is appropriate |
| **Model availability** | HuggingFace hosting; no official cloud API | Self-host via vLLM or use local GPU |

### Scope Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|-----------|
| **Not clinical** | Cannot and should not be used for clinical diagnosis or therapy | Explicit guardrails in agent description |
| **Not individual** | Cannot predict specific user behavior from their history | Frame as population-level behavioral tendency |
| **Not causal** | Predictions are correlational from training data, not causal models | Note in agent output; do not claim "because" |
| **English only** | Psych-101 is English-only | Do not use for cross-linguistic predictions |

---

## Blockers and Open Questions

### Current Blockers

1. **No public Centaur API endpoint**: Users must self-host the model. This
   requires significant GPU infrastructure (at minimum an A100-80GB or
   equivalent). The `rest_api` backend is ready when a server is available.

2. **Quantization accuracy**: The 4-bit quantized adapter (unsloth) is the
   practical option for 80 GB cards, but quantization may affect prediction
   accuracy vs the full bfloat16 model. This has not been independently
   validated outside the original paper.

3. **No GGUF/llama.cpp support**: Centaur is distributed as a HuggingFace model
   with a LoRA adapter. Running it via llama.cpp would require conversion to
   GGUF format, which is not officially supported. A `GGUFBackend` could be
   added if community conversion becomes available.

### Open Design Questions

1. **Batch prediction API**: Should the tool support predicting multiple trials
   in a single call? Currently each call predicts one choice point. For
   replicating full experiments (100+ trials), a batch mode would reduce
   overhead.

2. **Caching**: Should predictions be cached by prompt hash? Centaur is
   deterministic at temperature 0, so identical prompts always produce identical
   results. Caching would eliminate redundant GPU work.

3. **Multi-model support**: Should the backend abstraction support non-Centaur
   models that follow similar prompt formats? E.g., if a Centaur-8B or
   domain-specific variant is released, the same tool could serve them.

4. **Psych-101 dataset integration**: Should the bundle include utilities for
   loading and filtering the Psych-101 dataset directly? Currently users must
   use the `datasets` library separately.

---

## Experiment Domains Tested

The test harness covers all major Psych-101 experiment types:

| Domain | Dataset Reference | Tests | Token Type |
|--------|------------------|-------|-----------|
| Category Learning | badham2017deficits | 3 | Letter + feedback |
| Temporal Discounting | ruggeri2022globalizability | 3 | Letter (Q/Z) |
| Risky Choice | frey2017risk, wulff2018description | 3 | Letter (Q/Z) |
| Multi-Armed Bandits | wilson2014humans | 3 | Letter + points |
| Odd-One-Out Similarity | hebart2023things | 3 | Letter (D/P/H) |
| Columbia Card Task | frey2017cct | 3 | Multi-char (turn/stop) |
| Working Memory (N-back) | enkavi2019adaptivenback | 3 | Multi-char (match/no_match) |
| Digit Span | enkavi2019digitspan | 3 | Numeric (739, 4182) |
| Go/No-Go | enkavi2019gonogo | 3 | Multi-char (go/no_go) |
| Dunning-Kruger | jansen2021dunningkruger | 3 | Numeric (15, 60) |
| Reinforcement Learning | gershman2018deconstructing | 3 | Letter + stages |
| Cross-experiment | All | 21 | Parametrized |
| Edge cases | Various | 6 | Mixed |

**Total: 163 tests, all passing.**

---

## Future Work

### Near-term (ready to implement)

- **Batch prediction mode**: Accept an array of pending trials, return array of
  predictions. Reduces HTTP/subprocess overhead for experiment replication.
- **Result caching**: Content-addressed cache keyed by SHA256 of prompt text.
  Configurable TTL.
- **Ollama backend**: Many researchers have Ollama installed. If Centaur gets
  GGUF support, this backend would democratize access.

### Medium-term (requires investigation)

- **Fine-tuning integration**: Allow users to fine-tune Centaur on their own
  behavioral data. Would require a separate `centaur_finetune` tool.
- **Multi-model registry**: Support different model variants (Centaur-70B,
  hypothetical Centaur-8B, custom fine-tunes) via a model registry.
- **Visualization**: Generate plots of prediction distributions, learning
  curves, discount functions. Requires matplotlib or plotly integration.

### Long-term (research directions)

- **Individual difference modeling**: Extend Centaur to predict individual
  behavior from a brief calibration sequence (few-shot personalization).
- **Cross-cultural predictions**: Train on non-English behavioral data.
- **Real-time UX integration**: Stream predictions during live user testing.

---

## How to Use This Bundle

### Quick start (with a Centaur server running)

```yaml
# In your Amplifier settings.yaml
tools:
  - module: tool-centaur
    config:
      backend:
        type: rest_api
        api_url: http://your-gpu-server:8000
```

Then ask Amplifier:
> "How would people choose between $100 now and $150 in 6 months?"

The routing context will automatically delegate to `cognition-simulator`, which
frames the question as a temporal discounting experiment and runs it through
Centaur.

### Quick start (without GPU, using examples)

```bash
cd amplifier-centaur
uv run --with httpx python examples/temporal_discounting.py
uv run --with httpx python examples/risk_preference.py
uv run --with httpx python examples/replicate_experiment.py
```

These use MockBackend with realistic fake probabilities to demonstrate the full
pipeline without requiring a GPU.

### Running tests

```bash
cd amplifier-centaur
uv run --with httpx --with respx --with pytest --with pytest-asyncio \
  pytest tests/tool-centaur/ -v
```

163 tests, ~0.4 seconds.
