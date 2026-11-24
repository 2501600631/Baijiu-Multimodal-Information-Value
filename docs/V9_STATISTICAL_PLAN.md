# Prespecified SO-PLS structural comparison

This document fixes the V9 reporting and statistical rules before the new fifteen real-data task fits. Historical V6/V8 fits and findings have already been examined. V9 is a prospective specification of this additional computational comparison within a retrospective, same-cohort study; it is not an independently preregistered clinical or external validation study.

## Question and estimand

The primary question is whether allowing a sequentially selected component count for each spectral block changes the incremental value of adding NIR to UV plus Raman. Both fixed acquisition masks, U+R and U+R+N, are retained. Every one of the eleven constituents is reported. No constituent is primary and no direction is favored.

The new component grid is exactly 1, 2, 3, 5 for every receiving block. Earlier blocks' selected component counts are fixed while later blocks are selected. The existing preprocessing candidates and block orders are retained. Within every inner fitting partition, the complete selected prefix is refitted, block orthogonalization and response residuals use that fitting partition, and its cumulative held-out prediction is scored against the original validation response. An outer-training residual is never generated once and then treated as a fixed response for all inner folds.

The historical uniform-component SO-PLS grid is 1, 2, 3, 5, 10. The per-block grid reduces the largest requested totals from 20 to 10 for U+R and from 30 to 15 for U+R+N. Consequently, this comparison jointly changes allocation flexibility, the greedy selection rule and the maximum complexity. It does not isolate a pure increase of an upper component bound. A selection at the historical upper requested value is a diagnostic of the declared search grid, not proof that all rank or sample-size limits were reached. Requested counts and active counts must be distinguished.

## Data and paired roles

Fifteen outer tasks reuse the historical three repeats and five folds, including the original ordered training IDs, test IDs and seeds. Each ordinary full-size fit contains 111 or 112 distinct training bottles. There are 139 independent bottles; 417 held-out visits per constituent do not represent 417 independent samples.

The analysis pairs the same repeat, fold, held-out bottle and constituent. Paired reference values, fitting-response SDs and training-row counts must agree. Any missing or inconsistent pairing is an error. A valid new task retains both masks and every constituent. All target-level tuning uses raw-unit inner MSE. The panel loss is the equal mean of eleven constituent squared errors after division by the corresponding receiving outer-training response variance.

## Primary joined Holm family

For the directed contrast U+R to U+R+N, the historical four methods and the new structural method form one declared family:

1. PLS.
2. Ridge.
3. RBF-KRR.
4. Historical uniform-component SO-PLS.
5. New sequential per-block SO-PLS.

The panel family contains five tests. The constituent family contains all five methods times eleven constituents, or 55 tests. Every historical and new raw approximate corrected-CV P value is jointly adjusted again using Holm's procedure. Historical files are retained unchanged; joined-family adjusted values are written to new V9 tables. A new method is not assigned an isolated one-test family.

The five-member family concerns the one declared NIR-addition contrast. It is not a claim of study-wide multiplicity control across all historical additions, V8 sensitivities or every exploratory comparison. Conditional bootstrap intervals are not adjusted simultaneous intervals.

## Secondary structural comparison

For each fixed mask, compare historical uniform-component SO-PLS with the new per-block version. These two mask-specific panel effects form one two-test Holm family. Their eleven-constituent effects form one 22-test family. This secondary family is reported separately from the primary five-method NIR-addition family.

The secondary effect evaluates complete fitted pipelines, including the different selected preprocessing, order and component allocation. It is not a coefficient-level or chemical-specific decomposition.

## Effect sizes and conditional bottle intervals

For each constituent, report original-unit RMSE before and after, the difference after minus before, and relative gain as 100 times [1 minus RMSE(after)/RMSE(before)]. For the panel, report equal-target standardized MSE before and after, its difference and relative gain using MSE. Positive gain favors the after strategy; positive error difference means deterioration. Constituent RMSE gains and panel MSE gains are different estimands.

For complete paired data, average squared losses over the three saved visits within each bottle and constituent. Use 2,000 bottle resamples with seed 20261104. A single resampling index matrix carries all constituents and all historical/new methods together. Recompute the effect in every draw and take the 2.5th and 97.5th percentiles.

The label is conditional 95% paired bottle-bootstrap interval of frozen OOF predictions. It holds the fitted models, hyperparameters, preprocessing and component selections fixed. It does not include complete-procedure retraining uncertainty and does not repair cross-bottle dependence from overlapping training sets. An interval excluding zero is not automatically a multiplicity-adjusted confirmatory result.

## Approximate corrected-CV sensitivity

The saved fold losses also supply an approximate corrected repeated-CV sensitivity calculation. For J=15 dependent tasks, the variance estimate is [1/J + mean(n_test/n_train)] times the sample variance of paired task-loss differences. The t calculation uses J minus one degrees of freedom. Constituent tests use raw-unit MSE differences; panel tests use equal-target standardized MSE differences.

This is an approximate sensitivity analysis, not an exact test. Ordinary t tests or Wilcoxon tests that treat the overlapping folds as independent are not added. Constant nonzero task differences produce undefined P values rather than invented significance. Exact zero contrasts produce P=1. Undefined P values retain a place in the prespecified family and are not removed to shrink multiplicity penalties.

Fold-mean loss contrasts differ slightly from bottle-weighted pooled RMSE/MSE effects because test-fold sizes are 27 or 28. Report each with its own label. Neither a large P value nor a near-zero point estimate establishes equivalence or a justified noninferiority margin.

## Required outputs

| File under the new run's `derived/` folder | Meaning |
|---|---|
| `parameter_selection_by_task_target.csv` | Historical and new target-specific block orders, requested k1/k2/k3, counts by modality, active counts, selected preprocessing and inner scores. |
| `uniform_perblock_paired_task_parameters.csv` | One joined row for each method, task and constituent: U+R and U+R+N orders, k1/k2/k3 and inner scores side by side, with both task panel MSEs and their gain. A complete run has 330 rows representing 30 method-task units, not 330 independent tasks. |
| `selected_stage_scores.csv` | The selected new greedy path's stage, receiving block, selected k and pooled inner raw-MSE score. Full candidate scores remain in task payloads. |
| `paired_task_comparison.csv` | Every primary and secondary comparison by outer task, with both panel MSEs and directed gain. |
| `uniform_perblock_task_comparison.csv` | Historical uniform and new per-block SO-PLS, side by side, for U+R to U+R+N in each task. |
| `secondary_fixed_mask_task_comparison.csv` | Historical uniform to new per-block, separately within the two fixed masks. |
| `fold_target_metrics.csv`, `fold_panel_metrics.csv`, `pooled_target_metrics.csv` | Absolute errors for all methods and all eleven constituents. |
| `paired_target_effects.csv`, `paired_panel_effects.csv` | Matched pooled effect estimates, with all directions preserved. |
| `target_conditional_intervals.csv`, `panel_conditional_intervals.csv` | The conditional 2,000-resample intervals for the seven declared comparisons. |
| `target_corrected_cv_joined_family.csv`, `panel_corrected_cv_joined_family.csv` | Raw and jointly recomputed Holm-adjusted approximate sensitivity P values, with the declared family size. |
| `comparison_status.csv` | Complete or partial status for each comparison. |

The requested k1/k2/k3 follow the recorded block order, which can differ between constituents. They are not universal UV/Raman/NIR labels. `k_uv`, `k_raman` and `k_nir` provide the corresponding modality labels. The k3 field is empty for a two-block mask. A requested count may exceed the effective active count after rank capping or residual degeneracy; both are retained.

In the joined parameter table, each method-task panel MSE is intentionally repeated across its eleven constituent-parameter rows. The U+R fields start with `ur_`; three-block fields start with `urn_`. Conditional 2,000-draw intervals pool the complete 139-bottle repeated held-out predictions and are reported in the interval tables. They are not newly constructed per-task intervals from only 27 or 28 held-out bottles.

## Completion, failure and stopping

All fifteen requested tasks must complete before V9 conditional intervals or joined-family P values are reported. Partial new runs receive descriptive matched summaries only. Completion cannot be inferred from the existence of a table or a registry. The new condition must have 15 expected tasks and 15 completed tasks in `COMPLETENESS.json`, while the copied 255 completed V8 tasks remain independently identifiable.

Any candidate or selected-fit failure aborts its task and stops the runner. The failure is reported. No outer response is used for a fallback, and no favorable subset of successful tasks is substituted for the full run.

The component grid, greedy rule, tie rules, preprocessing options, block-order scope, folds, seeds, bootstrap draw count and Holm families are frozen before fitting. Source/configuration/data/runtime signatures guard resumption. Any required later change receives a new output directory and a stated new protocol. Null or adverse findings are retained. No repeated expansion until significance is allowed.
