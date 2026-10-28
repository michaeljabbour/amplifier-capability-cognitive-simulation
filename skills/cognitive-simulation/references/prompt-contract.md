# Centaur prompt and result contract

The optional engine is `modules/tool-centaur/`. `build_prompt` accepts an experiment description, prior stimulus/response/feedback trials, and a pending trial. Completed choices use `<<Q>>`; the pending prompt ends with `<<`. Raw mode accepts the same protocol directly. `parse_completion` returns choice plus available alternatives and probability. Missing log-probabilities must produce `null`, never certainty. The exact prompt and raw completion remain in the result envelope.

REST serving uses the deployed completion endpoint, not an assumed chat template. Inspect the configured backend before inference. Software fixtures and mocked completions establish protocol behavior only. Use independent participant observations for a claim about people.
