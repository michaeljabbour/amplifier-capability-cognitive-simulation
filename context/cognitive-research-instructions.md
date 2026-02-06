# Cognitive Research Capabilities

You have access to human cognition prediction via the Centaur foundation
model -- Llama-3.1-70B fine-tuned on 60,000+ participants across 160
psychology experiments (the Psych-101 dataset).

## When to Delegate

ALWAYS delegate to `cognitive-research:cognition-simulator` when the user:

- Asks about predicting human behavior or decision-making
- Wants to simulate participants in a psychology experiment
- Needs UX research predictions (which option would users prefer?)
- References cognitive biases, risk preferences, or temporal discounting
- Has Psych-101 format data or mentions the Centaur model
- Asks "what would people do if..." or "how would humans choose..."
- Wants to replicate or extend a published psychology experiment
- Needs choice probability estimates for human decisions

## Do NOT Delegate

- General psychology knowledge questions (answer directly)
- Requests about AI model behavior (Centaur predicts HUMAN cognition)
- Clinical psychology, diagnosis, or therapy (not appropriate use)

## Usage

Delegate ALL prediction tasks to `cognitive-research:cognition-simulator`.
The agent knows how to frame scenarios as experiments, call the
`centaur_predict` tool, and interpret the probabilistic results.
Do NOT call the `centaur_predict` tool directly from the root session.
