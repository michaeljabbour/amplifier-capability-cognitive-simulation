---
bundle:
  name: cognitive-research
  version: 0.1.0
  description: Human cognition prediction via the Centaur foundation model (Llama-3.1-70B fine-tuned on Psych-101)

includes:
  - bundle: git+https://github.com/microsoft/amplifier-foundation@main
  - bundle: cognitive-research:behaviors/cognitive-research
---

# Cognitive Research

@cognitive-research:context/cognitive-research-instructions.md

---

@foundation:context/shared/common-system-base.md
