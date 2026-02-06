# Orchestration Examples

How to route user requests to the `cognition-simulator` agent and the
`centaur_predict` tool within the Amplifier ecosystem.

> **Audience**: Bundle authors, root-session context designers, and anyone
> writing routing logic that touches human-behavior prediction.

---

## 1. When to Route to Centaur (Decision Matrix)

The root session NEVER calls `centaur_predict` directly. It delegates to
`cognitive-research:cognition-simulator`, which owns the full
understand-frame-predict-interpret cycle.

### Route YES -- delegate to cognition-simulator

| User intent pattern | Why it matches |
|---|---|
| "What would users do if..." | Direct behavior prediction request |
| "How would people choose between..." | Forced-choice paradigm |
| "Would users prefer X now or Y later?" | Temporal discounting |
| "Will users pick the safe option or gamble?" | Risky choice / expected utility |
| "If I show users examples, will they learn the rule?" | Category learning with feedback |
| "Simulate participants in [experiment name]" | Psych experiment replication |
| "Which design would users click more?" | UX preference prediction |
| "Predict how people explore vs exploit" | Multi-armed bandit / RL |
| "How would different groups respond to..." | Multi-scenario comparison |
| "Run this Psych-101 prompt through Centaur" | Raw prompt passthrough |
| "What's the probability users choose option A?" | Choice probability estimation |

### Route NO -- handle directly or delegate elsewhere

| User intent pattern | Why it does NOT match | What to do instead |
|---|---|---|
| "What's the psychology behind loss aversion?" | General knowledge question, no prediction needed | Answer directly from LLM knowledge |
| "Predict how GPT-4 would answer this prompt" | Centaur predicts HUMAN cognition, not AI | Answer directly or use appropriate AI eval tools |
| "Is my user depressed based on their behavior?" | Clinical diagnosis -- inappropriate use | Decline; recommend professional resources |
| "Explain the Stroop effect" | Educational question, no simulation requested | Answer directly |
| "What neurotransmitter causes risk-seeking?" | Neuroscience mechanism, not behavioral prediction | Answer directly |
| "Help me design a psychology experiment" | Experiment *design* help, not *prediction* | Use zen-architect or answer directly |
| "Summarize the Centaur paper" | Meta-question about the tool, not a prediction | Answer directly from CENTAUR_REFERENCE.md |

### Edge Cases

| Ambiguous request | Resolution |
|---|---|
| "Tell me about temporal discounting" | NO -- educational question. But if followed by "...and predict what a user would choose," YES. |
| "How do people usually decide under uncertainty?" | NO -- general knowledge. Route only if the user provides a *specific* scenario to predict. |
| "Which of these two options is better?" | DEPENDS -- if asking about objective quality, NO. If asking which one *users would pick*, YES. |
| "Replicate Kahneman's prospect theory" | YES -- if the user wants simulated participant choices. NO -- if they want a summary of the theory. |
| "Would a 5-year-old choose differently?" | YES with caveat -- route to cognition-simulator, but note that Centaur's training population is predominantly adults. The agent will include this limitation. |

---

## 2. Prompt Routing Examples

Each example shows: the user's natural-language request, how the
`cognition-simulator` agent frames it, the exact `centaur_predict` call,
and how to interpret the result.

### a) UX A/B Test -- Checkout Flow Preference

**User says:**
> "Which checkout flow would users prefer -- one-page or multi-step?"

**Step 1 -- Agent frames as forced-choice experiment**

The agent maps "one-page" and "multi-step" to key labels Q and Z (the
standard Centaur choice tokens) and writes experiment instructions that a
participant would read.

**Step 2 -- centaur_predict call (structured mode)**

```json
{
  "experiment_description": "You will be presented with a choice between two online checkout experiences. Please indicate which option you would prefer by pressing the corresponding key.\n\nOption Q: A single-page checkout where shipping address, payment, and order review are all visible on one screen.\nOption Z: A multi-step checkout that guides you through shipping, then payment, then review on separate screens.",
  "prior_trials": [],
  "pending_trial": "You are buying a pair of shoes online. You can use checkout Q (single page) or checkout Z (multi-step guided). You press"
}
```

**What happens internally:**

The tool's `build_prompt()` function assembles this into the Centaur
token format:

```
You will be presented with a choice between two online checkout experiences.
Please indicate which option you would prefer by pressing the corresponding key.

Option Q: A single-page checkout where shipping address, payment, and order
review are all visible on one screen.
Option Z: A multi-step checkout that guides you through shipping, then payment,
then review on separate screens.
You are buying a pair of shoes online. You can use checkout Q (single page) or checkout Z (multi-step guided). You press <<
```

The model completes after `<<` with something like `Q>>.` and returns
log-probabilities for all candidate tokens.

**Step 3 -- Interpret the result**

```json
{
  "predicted_choice": "Q",
  "probability": 0.6142,
  "alternatives": { "Z": 0.3858 },
  "raw_output": "Q>>.",
  "prompt_used": "...",
  "backend": "rest_api"
}
```

**Agent's interpretation:** The model leans toward the single-page checkout
(61%) but with meaningful uncertainty. This is a moderate-confidence
prediction -- not a slam dunk for either design. Consider running a real
A/B test; Centaur suggests the single-page flow has an edge but the
preference is not overwhelming.

---

### b) Temporal Discounting -- Discount Now vs Later

**User says:**
> "Would users prefer a $10 discount now or $15 off next month?"

**Agent frames as temporal discounting paradigm**

This maps directly to Centaur's core strength: immediate vs delayed
reward trade-offs.

**centaur_predict call:**

```json
{
  "experiment_description": "You will be presented with multiple choices between two options Q and Z.\nPlease name which option you would prefer by pressing the corresponding key.",
  "prior_trials": [],
  "pending_trial": "You have the choice between receiving a $10 discount on your purchase today (press Q) or receiving a $15 discount on a purchase next month (press Z). You press"
}
```

**Assembled prompt (what Centaur sees):**

```
You will be presented with multiple choices between two options Q and Z.
Please name which option you would prefer by pressing the corresponding key.
You have the choice between receiving a $10 discount on your purchase today (press Q) or receiving a $15 discount on a purchase next month (press Z). You press <<
```

**Example result:**

```json
{
  "predicted_choice": "Q",
  "probability": 0.7291,
  "alternatives": { "Z": 0.2709 }
}
```

**Agent's interpretation:** Strong preference for the immediate $10 (73%).
This aligns with well-documented hyperbolic discounting -- humans
systematically overweight immediate rewards. The 50% premium ($10 to $15)
is not enough to overcome the one-month delay for most people.

**Extending the analysis:** The agent might suggest running a second
prediction with a larger delay gap (e.g., $10 now vs $25 next month) to
find the indifference point.

---

### c) Risk Preference -- Guaranteed vs Gamble

**User says:**
> "Will users opt for the guaranteed $50 reward or the 50% chance at $120?"

**Agent frames as risky choice paradigm**

This is a classic expected-value problem. The gamble has higher expected
value ($60) but carries risk. Centaur is trained extensively on exactly
this type of task.

**centaur_predict call with prior trials for calibration:**

Including a prior trial lets the model calibrate risk attitude from
context. This is optional but improves ecological validity.

```json
{
  "experiment_description": "You will be presented with multiple choices between two options Q and Z.\nPlease name which option you would prefer by pressing the corresponding key.\nOne option is always a sure payment. The other option is a gamble with some probability of a larger payment.",
  "prior_trials": [
    {
      "stimulus": "You have the choice between receiving $20 for sure (press Q) or a 50% chance of receiving $45 and a 50% chance of receiving nothing (press Z). You press",
      "response": "Q"
    }
  ],
  "pending_trial": "You have the choice between receiving $50 for sure (press Q) or a 50% chance of receiving $120 and a 50% chance of receiving nothing (press Z). You press"
}
```

**Assembled prompt:**

```
You will be presented with multiple choices between two options Q and Z.
Please name which option you would prefer by pressing the corresponding key.
One option is always a sure payment. The other option is a gamble with some probability of a larger payment.
You have the choice between receiving $20 for sure (press Q) or a 50% chance of receiving $45 and a 50% chance of receiving nothing (press Z). You press <<Q>>.
You have the choice between receiving $50 for sure (press Q) or a 50% chance of receiving $120 and a 50% chance of receiving nothing (press Z). You press <<
```

**Example result:**

```json
{
  "predicted_choice": "Q",
  "probability": 0.6834,
  "alternatives": { "Z": 0.3166 }
}
```

**Agent's interpretation:** The model predicts most people would take the
sure $50 (68%), consistent with loss aversion and risk aversion in the
gains domain. The prior trial (where the participant also chose safe)
reinforces a risk-averse profile. Without the prior trial, the
probability might shift slightly toward the gamble.

---

### d) Learning Simulation -- Pattern Acquisition

**User says:**
> "If I show users 10 training examples, will they learn the pattern?"

**Agent frames as category learning with feedback**

This is a supervised learning paradigm. The agent constructs a trial
sequence where participants see stimuli, make guesses, and receive
corrective feedback. The pending trial tests whether learning has
occurred.

**centaur_predict call:**

```json
{
  "experiment_description": "You will be shown several examples of geometric objects.\nYour task is to learn a rule that allows you to tell whether an object belongs to the A or B category.\nFor each presented object, you will be asked to make a category judgment by pressing the corresponding key and then you will receive feedback.",
  "prior_trials": [
    { "stimulus": "You see a large red circle. You press", "response": "A", "feedback": "The correct category is A." },
    { "stimulus": "You see a small blue square. You press", "response": "B", "feedback": "The correct category is B." },
    { "stimulus": "You see a large red square. You press", "response": "B", "feedback": "The correct category is A." },
    { "stimulus": "You see a small blue circle. You press", "response": "A", "feedback": "The correct category is B." },
    { "stimulus": "You see a large green circle. You press", "response": "A", "feedback": "The correct category is A." },
    { "stimulus": "You see a small green square. You press", "response": "B", "feedback": "The correct category is B." },
    { "stimulus": "You see a large red triangle. You press", "response": "A", "feedback": "The correct category is A." },
    { "stimulus": "You see a small blue triangle. You press", "response": "A", "feedback": "The correct category is B." },
    { "stimulus": "You see a large green square. You press", "response": "A", "feedback": "The correct category is A." },
    { "stimulus": "You see a small red circle. You press", "response": "B", "feedback": "The correct category is B." }
  ],
  "pending_trial": "You see a large blue triangle. You press"
}
```

Note the pattern: the rule is **large = A, small = B** (color and shape
are irrelevant). Some prior trials show the simulated participant making
errors early and improving over time -- this is realistic and matches
Psych-101 training data.

**Example result:**

```json
{
  "predicted_choice": "A",
  "probability": 0.8217,
  "alternatives": { "B": 0.1783 }
}
```

**Agent's interpretation:** After 10 trials with feedback, the model
predicts high confidence (82%) that a participant would correctly classify
the new object as category A. This suggests the rule (size determines
category) is learnable within 10 examples for most people, though some
residual uncertainty remains -- consistent with human learning curves.

---

### e) Multi-Scenario Comparison -- Age Group Differences

**User says:**
> "Compare how different age groups might respond to a risky vs safe
> investment choice."

**Agent approach:** Run multiple `centaur_predict` calls with varied
framing to approximate demographic differences. Centaur does not have an
explicit demographic input, so the agent varies the scenario context.

**Call 1 -- Young adult framing:**

```json
{
  "experiment_description": "You are a 25-year-old professional making investment decisions.\nYou will be presented with choices between two options Q and Z.\nPlease name which option you would prefer by pressing the corresponding key.",
  "pending_trial": "You have the choice between a guaranteed 5% annual return on your savings (press Q) or investing in a portfolio with a 50% chance of 12% return and a 50% chance of -2% return (press Z). You press"
}
```

**Call 2 -- Retiree framing:**

```json
{
  "experiment_description": "You are a 68-year-old retiree making investment decisions with your retirement savings.\nYou will be presented with choices between two options Q and Z.\nPlease name which option you would prefer by pressing the corresponding key.",
  "pending_trial": "You have the choice between a guaranteed 5% annual return on your savings (press Q) or investing in a portfolio with a 50% chance of 12% return and a 50% chance of -2% return (press Z). You press"
}
```

**Comparing results:**

| Scenario | Predicted choice | Probability | Alternative |
|---|---|---|---|
| 25-year-old | Z (risky) | 0.58 | Q: 0.42 |
| 68-year-old | Q (safe) | 0.71 | Z: 0.29 |

**Agent's interpretation:** The model predicts the younger framing shifts
preference toward the riskier investment, while the retiree framing
strongly favors the guaranteed return. The difference in probabilities
(~13 percentage points) suggests Centaur has captured age-related risk
context from its training data.

**Important caveat the agent MUST note:** Centaur was not trained with
explicit demographic variables. These results reflect how scenario framing
(mentioning age and context) shifts the predicted behavior. They should
NOT be cited as precise demographic predictions. Use as directional
hypotheses only.

---

### f) Raw Psych-101 Replication

**User says:**
> "Run this experiment prompt through Centaur exactly as-is:
>
> You will be presented with multiple choices between two options Q and Z.
> Please name which option you would prefer by pressing the corresponding key.
>
> You have the choice between receiving 500$ immediately (press Q) or receiving 550$ in one year (press Z). You press <<Q>>.
> You have the choice between receiving 500$ immediately (press Q) or receiving 600$ in one year (press Z). You press <<"

**Agent uses raw_prompt mode** -- no reformatting, no experiment framing.
The prompt is passed through verbatim.

**centaur_predict call:**

```json
{
  "raw_prompt": "You will be presented with multiple choices between two options Q and Z.\nPlease name which option you would prefer by pressing the corresponding key.\n\nYou have the choice between receiving 500$ immediately (press Q) or receiving 550$ in one year (press Z). You press <<Q>>.\nYou have the choice between receiving 500$ immediately (press Q) or receiving 600$ in one year (press Z). You press <<"
}
```

**What happens internally:**

`build_prompt_raw()` validates that the text contains at least one `<<`
marker and passes it through. No `experiment_description` / `prior_trials`
/ `pending_trial` fields are used.

**Example result:**

```json
{
  "predicted_choice": "Q",
  "probability": 0.6503,
  "alternatives": { "Z": 0.3497 },
  "raw_output": "Q>>."
}
```

**Agent's interpretation:** Even with the increased delayed reward ($600
vs $550 in the prior trial), the model still predicts a majority
preference for the immediate $500 (65%). The prior trial where the
participant chose immediate ($500 over $550) establishes a discounting
context that carries forward. This is a direct Psych-101 replication and
can be validated against the published dataset.

---

## 3. Integration Patterns

The `cognition-simulator` agent is most powerful when composed with other
Amplifier agents in a multi-step workflow.

### Explorer + Cognition Simulator: UX Audit

**Scenario:** Audit an existing application for likely user confusion.

```
1. User: "Audit our checkout flow for UX problems"

2. Root session delegates to foundation:explorer
   -> Explorer scans the codebase, identifies 3 checkout screens
   -> Reports: screen 1 has 12 form fields, screen 2 has a
      confusing "save for later" vs "add to wishlist" choice,
      screen 3 has a small "confirm" button

3. Root session delegates to cognitive-research:cognition-simulator
   -> Agent frames the confusing choice as a forced-choice experiment:
      "Would users press 'save for later' (Q) or 'add to wishlist' (Z)
       when they want to come back to this item?"
   -> centaur_predict returns: Q at 0.52, Z at 0.48
   -> Interpretation: Near-random choice (52/48) indicates the labels
      are genuinely confusing -- users cannot distinguish them

4. Root session synthesizes: "The 'save for later' vs 'add to wishlist'
   distinction is predicted to confuse users (near 50/50 split).
   Recommend consolidating into a single action."
```

### Zen-Architect + Cognition Simulator: Feature Validation

**Scenario:** Validate assumptions during feature design.

```
1. User: "Design a pricing page with three tiers"

2. Root session delegates to foundation:zen-architect
   -> Architect designs: Free / Pro ($12/mo) / Enterprise ($49/mo)
   -> Assumption: "Most users will upgrade from Free to Pro"

3. Root session delegates to cognitive-research:cognition-simulator
   -> Agent frames as a three-way forced choice:
      "You are choosing a software plan. Option D is free with basic
       features. Option P is $12/month with advanced features. Option
       H is $49/month with all features. You press"
   -> centaur_predict returns: D at 0.54, P at 0.31, H at 0.15
   -> Interpretation: Majority stays on Free. The jump to Pro is not
      compelling enough at the current feature/price split.

4. Architect revises based on prediction: adjusts Pro pricing or adds a
   trial period to lower the barrier.
```

### Bug-Hunter + Cognition Simulator: UX Bug Diagnosis

**Scenario:** Investigate why users keep hitting an error state.

```
1. User: "Users keep submitting the form without filling required fields"

2. Root session delegates to foundation:bug-hunter
   -> Bug-hunter confirms: no code bug. The submit button is always
      enabled. Required fields are marked with a small asterisk.
   -> Hypothesis: users don't notice the required-field indicators

3. Root session delegates to cognitive-research:cognition-simulator
   -> Agent frames as a visual attention / decision scenario:
      "You see a form with 8 fields. Some are marked with a small
       asterisk (*) next to the label. You have filled in 5 fields.
       Do you press 'Submit' (Q) or continue filling fields (Z)?"
   -> centaur_predict returns: Q at 0.67
   -> Interpretation: 67% predicted to hit Submit early -- the asterisk
      is not a strong enough signal

4. Root session recommends: "Add inline validation or disable the Submit
   button until required fields are complete. The asterisk convention
   alone is insufficient for most users."
```

---

## 4. Backend Selection Guide

The `centaur_predict` tool supports three inference backends. Choose based
on your deployment context.

### REST API (vLLM / TGI server)

**When to use:** Production deployments, shared team server, latency-sensitive
applications.

```yaml
# behaviors/cognitive-research.yaml
tools:
  - module: tool-centaur
    config:
      backend:
        type: rest_api
        api_url: http://localhost:8000
        api_key_env: CENTAUR_API_KEY
        model_name: ""
        timeout: 60
      max_tokens: 16
```

| Attribute | Detail |
|---|---|
| Latency | 1-5 seconds per prediction |
| Concurrency | Multiple users can share one server |
| Setup | Run vLLM or TGI with the Centaur model, point `api_url` at it |
| Auth | API key read from the `CENTAUR_API_KEY` environment variable |
| Log-probs | Supported (requests `logprobs: 10` from the OpenAI-compatible endpoint) |

**Typical vLLM launch command:**

```bash
vllm serve marcelbinz/Llama-3.1-Centaur-70B \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9
```

### CLI (subprocess)

**When to use:** Integration with existing inference scripts, batch
processing pipelines, HPC job schedulers.

```yaml
tools:
  - module: tool-centaur
    config:
      backend:
        type: cli
        command: "python /path/to/infer.py --prompt-file {prompt_file} --output-file {output_file} --max-tokens {max_tokens}"
        timeout: 120
      max_tokens: 16
```

| Attribute | Detail |
|---|---|
| Latency | Depends on the script (10s+ typical for cold start) |
| Concurrency | One prediction at a time (sequential subprocess calls) |
| Setup | Point `command` at any script that reads a prompt file and writes JSON output |
| Output format | Script must write `{"completion": "Q>>.", "token_log_probs": {"Q": -0.31, "Z": -1.34}}` to `{output_file}` |
| Log-probs | Supported if the script writes `token_log_probs` |

The CLI backend uses temp files: it writes the prompt to `{prompt_file}`,
invokes the command, and reads the result from `{output_file}`. Both
files are cleaned up automatically.

### Local GPU (unsloth / transformers)

**When to use:** Development, fine-tuning experiments, offline use,
maximum control over inference parameters.

```yaml
tools:
  - module: tool-centaur
    config:
      backend:
        type: local_gpu
        model_id: marcelbinz/Llama-3.1-Centaur-70B-adapter
        device: auto
      max_tokens: 16
```

| Attribute | Detail |
|---|---|
| Latency | 5-15 seconds (after model is loaded; first call loads the model) |
| Concurrency | Single-user (model holds GPU memory) |
| VRAM | 80 GB with unsloth (4-bit), 160 GB+ with transformers (bfloat16) |
| Setup | Install torch + unsloth (or transformers) in your Python environment |
| Log-probs | Extracted from model logits (top-10 tokens) |

**Important:** The heavy dependencies (torch, transformers, unsloth) are
NOT declared in `tool-centaur`'s `pyproject.toml`. They must be
pre-installed in the runtime environment. This is intentional -- it avoids
forcing GPU dependencies on REST API users.

### Decision Flowchart

```
Do you have a running vLLM/TGI server?
├── YES -> Use rest_api (fastest, shared, production-ready)
└── NO
    ├── Do you have a GPU with 80+ GB VRAM?
    │   ├── YES -> Use local_gpu (full control, no server needed)
    │   └── NO
    │       ├── Do you have an existing inference script?
    │       │   ├── YES -> Use cli (wraps any script)
    │       │   └── NO -> Set up a vLLM server (recommended path)
    │       └── Do you need batch processing on HPC?
    │           └── YES -> Use cli with your SLURM/PBS script
    └── Are you just developing/testing the bundle?
        └── YES -> Use rest_api with a mock server, or run tests
            (pytest uses MockBackend, no GPU needed)
```

---

## Appendix: Output Contract Reference

Every response from `cognition-simulator` includes these 7 sections:

| # | Section | Content |
|---|---------|---------|
| 1 | **Scenario** | How the agent understood the user's question |
| 2 | **Experiment Design** | The structured input sent to `centaur_predict` |
| 3 | **Prediction** | Predicted choice and its probability |
| 4 | **Alternatives** | Other choices the model considered with probabilities |
| 5 | **Interpretation** | Plain-language meaning for the user's context |
| 6 | **Caveats** | Relevant limitations (population bias, domain fit, etc.) |
| 7 | **Reproducibility** | Exact prompt used, so the user can re-run or modify |

### Confidence Thresholds

| Probability range | Label | Phrasing guidance |
|---|---|---|
| > 0.80 | High confidence | "The model strongly predicts humans would choose X" |
| 0.55 -- 0.80 | Moderate | "The model leans toward X but with meaningful uncertainty" |
| < 0.55 | Near-even | "The model predicts roughly equal preference -- no clear winner" |
