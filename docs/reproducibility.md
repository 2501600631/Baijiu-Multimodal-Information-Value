# Reproducibility and imported execution records

The repository contains three completed collections: 210 baseline tasks, 255 robustness tasks and fifteen structural tasks. All numerical inputs and task payloads are retained. All prediction directions and all eleven target outcomes remain available.

The directory layout and descriptive text were normalized during export. Recorded execution signatures identify the original numerical executions; their hex values are preserved. Path-bearing imported protocol contexts now use the portable layout, so their current representation digest is stored separately. It would be incorrect to claim that a new source representation produced the old executions. `provenance/public_representation.json` distinguishes the imported identities from the current file hashes explicitly.

The export-time design snapshot checks the current rules and code before a fresh rerun. It does not replace the original pre-fit design record or backdate the export. Fresh fitting captures a new source/data/runtime signature and writes to `runs/`, while imported completed outputs under `results/` remain unchanged.

Validation checks each original task signature, payload digest, ordered fitting/test bottle IDs, inner roles, target coverage and saved reference values. It also verifies the current representation's code, configuration, input and result file hashes. The numerical-preservation record compares source and exported inputs, all prediction tables and all task payloads without fitting models.

Rebuilding tables reads saved predictions and candidate diagnostics. Rebuilding figures reads retained baseline tables. These commands write to `work/`. A full baseline refit, the robustness extension and the structural extension use their declared fixed seeds and grids. The equal-budget and structural dependencies on retained baseline evidence are intentional design features, not unrecorded external inputs.

Runtime records retain package versions, single-thread BLAS settings and floating-point details. Historical machine-specific path strings use neutral environment markers; they are not executable dependencies. CPU and BLAS differences can change floating-point rounding, so a fresh fit should be assessed using its captured actual runtime rather than assumed bitwise identity.

Git line-ending conversion is disabled in `.gitattributes` so that saved data, code and protocol file hashes remain stable across checkouts.
