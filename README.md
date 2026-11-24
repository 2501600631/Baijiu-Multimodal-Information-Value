# Baijiu Multimodal Information Value

Version 1.0.0 distributes the spectral arrays, numerical methods, analysis code, diagnostics and derived statistical tables for an eleven-constituent study of 139 independent strong-aroma baijiu bottles. UV, Raman and NIR spectra are compared using seven fixed modality subsets, four original algorithm families, training-only acquisition rules and a per-block SO-PLS structural extension.

Reference concentrations, completed task caches, decomposed prediction tables and figures derived from them are withheld under the data-sharing agreement with the distillery, which does not permit their public release. What is distributed is stated section by section below; what was executed but withheld is recorded, with its execution identity, in `provenance/`.

All eleven constituents retain equal reporting status. The repository retains favorable, null and adverse results. No independent production-batch or external-cohort validation is claimed.

## Completed experiments

| Collection | Saved fitting tasks | Prediction records | Location |
|---|---:|---:|---|
| Baseline comparisons and point-prediction sensitivities | 210 | 2,990,724 | `results/baseline/` |
| Preprocessing, search-budget, NIR-gate and residual sensitivities | 255 | 376,134 | `results/robustness/` |
| Per-block SO-PLS structural extension | 15 | 9,174 | `results/structural/` |
| **Total** | **480** | **3,376,032** | All completed |

These are repeated prediction records, not additional independent samples. The independent sample count is 139. The structural completion file also reports 270 combined tasks, representing the 255 robustness tasks plus the 15 structural tasks; the baseline collection is separate.

The counts above record what was executed and are retained as an identity record. The corresponding task caches and prediction tables are not part of this distribution, so the counts cannot be reproduced by downloading files.

## Repository layout

| Folder | Contents |
|---|---|
| `data/` | English-schema spectral workbooks for UV, Raman and NIR, with value-preservation records. |
| `executed/data/` | Exact numerical spectral arrays used by the fitting code; the reference-concentration table is withheld. |
| `executed/` | Original numerical implementations and their synthetic tests. |
| `v8/` | Robustness experiment code, statistics and tests. |
| `v9/` | Per-block SO-PLS vector entry, greedy selection, joined statistical analysis and tests. |
| `configs/` | Declared grids, folds, seeds and selection rules. |
| `docs/` | Scientific methods, fixed selection rules, statistical definitions and output dictionaries. |
| `results/` | Derived statistical tables, selection diagnostics and fitting diagnostics; completed task payloads and prediction tables are withheld. |
| `provenance/` | Recorded execution identities, representation-migration receipts and integrity checks. |
| `repository/` | Portable validation entry point, plus the table-rebuilding path that requires the withheld predictions. |

## Inspect the completed results

Open `STUDY_STATUS.json`, then the three collection-specific `COMPLETENESS.json` files. The structural tables are under `results/structural/derived/`; their `analysis_registry.json` lists all generated tables. The per-constituent parameter table that joins each task and constituent's two ordered component vectors with the two panel MSEs and directed gain is withheld with the rest of the task-level outputs; panel values repeat over eleven parameter rows and do not create additional independent tasks.

The primary structural comparison joins PLS, Ridge, RBF-KRR, uniform SO-PLS and per-block SO-PLS for the U+R to U+R+N addition. Its Holm families contain five panel and fifty-five constituent tests. The separate within-mask uniform-to-per-block comparisons have two panel and twenty-two constituent tests. The component grid and interpretation are recorded in `docs/V9_SELECTION_RULES.md` and `docs/V9_STATISTICAL_PLAN.md`.

## Set up and verify

Use Python 3.8 through 3.11 on Windows. The numerical dependency pins in `requirements.txt` are unchanged from the completed experiment. The local environment is created by the setup script; third-party environments are not bundled.

1. Run `01_install_isolated.bat` once to install the pinned dependencies in `.venv`.
2. Run `02_verify_repository.bat` to verify all distributed source, data and result hashes without fitting models. A collection whose task caches are withheld is reported as `withheld` rather than failed.
3. `03_rebuild_reports.bat` and `04_rebuild_figures.bat` require the withheld task-level prediction tables and per-constituent metric tables, so neither runs against this distribution.

`05_synthetic_checks.bat` runs the three numerical test suites. These tests check software behavior and do not substitute for completed real-data validation.

## Refit from the supplied inputs

`run_baseline.bat`, `run_robustness.bat` and `run_structural.bat` launch the three experiment collections separately. `run_full.bat` runs all three in sequence. These entry points read the reference-concentration table and therefore do not run against this distribution; they are retained so that the code path producing the withheld results stays auditable. New fitting results go to `runs/reproduce_baseline/`, `runs/reproduce_robustness/` and `runs/reproduce_structural/`; the retained completed results remain in `results/`.

The robustness equal-budget condition intentionally uses the verified baseline inner-score archive, then refits the selected models for held-out prediction. The structural extension intentionally uses the completed robustness design and the same bottle partitions. These dependencies are included locally. No private workspace path is required. Repeated launching resumes verified fresh-run caches; edited source, rules, data or scientific runtime require a fresh output directory.

The structural extension has exactly fifteen new tasks. Its greedy component grid is [1,2,3,5], with all original preprocessing options and block orders retained. Every receiving inner fitting partition refits the full selected prefix. Any candidate or final-fit error stops the run; no outer-label fallback is allowed. The original uniform grid remains [1,2,3,5,10], so the comparison jointly changes allocation and maximum complexity.

## Interpretation

Constituent RMSE is reported in g/L. Panel MSE gives equal weight to all eleven targets after scaling squared errors by receiving-training target variance. Positive relative gain means lower error; a positive after-minus-before error difference means deterioration. Component gains and panel gains use different error definitions.

The 2,000-resample bottle intervals are conditional on saved OOF predictions, not a bootstrap of the entire retraining procedure. Corrected repeated-CV tests are approximate sensitivity analyses, with their stated context-specific Holm families. Five training refits or search-budget draws supply observed ranges, not confidence intervals. Counts of latent variables are not equivalent to effective degrees of freedom.

## Representation and numerical preservation

The completed executions were imported into a portable directory layout. Descriptive path names, text labels and historical machine-path fields were normalized. Original numerical inputs, task predictions and task payloads are preserved in the record of the completed run, and their recorded execution signatures remain distinct from the current representation signature; the restricted files themselves are not distributed. No model was refitted while building this repository.

`provenance/public_representation.json` attests the current file representation. `provenance/public_design_snapshot.json` is an export-time integrity snapshot, not a new pre-fit record. The original design and execution digests are retained as imported records. `CHECKSUMS.sha256` validates this distribution. See `provenance/numerical_preservation.json` for the preservation checks.

Distributed results and numerical inputs are tracked by Git. Local environments, temporary rebuilds and fresh reruns are excluded by `.gitignore`. No individual distributed file exceeds the standard 100 MiB size limit. Contributor identities, ownership and reuse terms are described in `CITATION_METADATA.md` and `LICENSE_STATUS.md`; no identity or license grant has been invented.
