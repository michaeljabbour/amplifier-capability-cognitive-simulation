# Consolidation and compatibility map

Primary source: `amplifier-bundle-centaur` at `a01fbafef7dfeea7d57d55d45ae8b76a48b6bb74`. Its Git history, native code and evaluation assets remain. Original personal working copies were not edited. The public repository has not been renamed or archived. Eventual publication should rename the primary owner repository to `amplifier-skill-cognitive-simulation`, preserving its history and redirects; this local name is not a live URL.

## Canonical portable entry

`skills/cognitive-simulation/SKILL.md` is the installable front door. Older topic entrypoints become preserved rich references; consumers should load the canonical entry and select the named mode/reference. This is a staged compatibility migration, not deletion of the source method. Native namespace/engine identities stay as originally registered.

| Original path | Current path | Role |
|---|---|---|
| `agents/cognition-simulator.md` | `skills/cognitive-simulation/references/cognition-simulator.md` | Preserved source method; reached from canonical entry. |
| `docs/CENTAUR_REFERENCE.md` | `skills/cognitive-simulation/references/CENTAUR_REFERENCE.md` | Preserved source method; reached from canonical entry. |

Original examples/claims are retained as source evidence, not new benchmark claims. Distinct native methods and engines are not flattened into one generic checklist. Review all downstream consumers before retiring a secondary repository.

## Secondary source routes


Secondary ancestry is retained in merge commits using an ours strategy; selected procedures/contracts are imported deliberately, rather than overlaying unrelated engine trees. No original repository has been removed.

## Result schema repair

`CentaurPrediction.probability` and tool JSON probability now allow null when no probability is supplied for the chosen token. The former 1.0 sentinel falsely implied certainty. Consumers formatting percentages must handle null; retained examples and regression expectations are updated. Supplied probabilities retain their numeric values. Mocked tests do not establish model calibration.
