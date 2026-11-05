# Output dictionary

`analysis_registry.json` is the entry point for each analysis. It lists the files actually written, their row counts and SHA-256 hashes, together with the prediction-source hash, analysis-source hash and statistical settings. The registry is evidence of postprocessing, not proof that every requested V8 task has completed. Use the runner's `COMPLETENESS.json` for task completion.

Archive audit outputs are placed in `results/v8_archive_audit/`. New runs write to their own run directory, such as `results/robustness/`. In each case, tables are under `derived/`. Optional or ineligible tables can be absent; the registry is authoritative.

## Archived prediction audit

The source is the immutable `results/baseline/benchmark/predictions.csv.gz`.

| Table under `derived/` | Rows in the complete archive audit | Row meaning |
|---|---:|---|
| `archive_fixed_fold_target_metrics.csv` | 4,620 | Outer task x seven fixed masks x four algorithms x eleven constituents |
| `archive_fixed_fold_panel_metrics.csv` | 420 | Outer task x fixed mask x algorithm, with eleven equal constituent weights |
| `archive_all_strategy_target_metrics.csv` | 583 | All 53 archived strategies x eleven constituents, pooled across three repeats |
| `archive_all_strategy_panel_by_repeat.csv` | 159 | All 53 strategies x three repeats |
| `target_conditional_intervals.csv` | 396 | Nine directed additions x four algorithms x eleven constituents |
| `panel_conditional_intervals.csv` | 36 | Nine directed additions x four algorithms |
| `target_corrected_cv_sensitivity.csv` | 396 | Approximate corrected-CV sensitivity for each constituent addition |
| `panel_corrected_cv_sensitivity.csv` | 36 | Approximate corrected-CV sensitivity for each panel addition |
| `conditional_context_status.csv` | 36 | Eligibility and analysis status for each archived paired addition |
| `concentration_summary.csv` | 11 | One summary for each reference constituent |
| `spectral_dimensions_and_sample_ratios.csv` | 35 | Seven masks x five sample-count definitions |
| `archive_five_refit_observed_ranges.csv` | 1,936 | Retained historical five-refit effect ranges, with explicit range scope |
| `archive_acquisition_union_by_task.csv` | 4,680 | Retained benchmark/stability acquisition records by dependent outer task |
| `archive_acquisition_union_frequency.csv` | 150 | Retained benchmark/stability acquisition-union frequencies |
| `archive_selection_frequency.csv` | 1,967 | Retained benchmark/stability constituent-selection frequencies |

The first two row counts are `15 * 7 * 4 * 11` and `15 * 7 * 4`. The 53 archived strategies comprise four algorithms with thirteen strategies each and one training-mean baseline. The audit does not refit these models.

The nine directed additions are U to U+R, R to U+R, U to U+N, N to U+N, R to R+N, N to R+N, U+R to U+R+N, U+N to U+R+N, and R+N to U+R+N. U denotes ultraviolet-visible, R Raman and N near-infrared.

## New-run analysis

The runner consolidates completed task predictions into `predictions.csv.gz`. New-run output sizes depend on completed contexts.

| Table under `derived/` | Row meaning |
|---|---|
| `fold_target_metrics.csv` | Condition, budget/refit context, strategy, outer task and constituent |
| `fold_panel_metrics.csv` | Corresponding equal-target panel loss by outer task |
| `pooled_target_metrics.csv` | Strategy and constituent, pooled only within the same context |
| `panel_by_repeat_and_refit.csv` | Panel loss by repeat, preserving each budget/refit context |
| `paired_target_effects.csv` | Matched before/after constituent effects, including learning and stability contexts |
| `paired_panel_effects.csv` | Matched panel effects, including common-scale learning assessments |
| `target_conditional_intervals.csv` | Constituent intervals for eligible complete full-size paired contexts |
| `panel_conditional_intervals.csv` | Panel intervals for the same eligible contexts |
| `target_corrected_cv_sensitivity.csv` | Approximate corrected-CV constituent sensitivity results |
| `panel_corrected_cv_sensitivity.csv` | Approximate corrected-CV panel sensitivity results |
| `conditional_context_status.csv` | Eligible, incomplete, learning/refit descriptive-only, or no declared pair |
| `acquisition_by_task.csv` | Shared effective NIR weight and modality count, when supplied by the runner |
| `acquisition_frequency.csv` | Observed dependent-task acquisition counts and proportions |
| `target_observed_budget_or_refit_ranges.csv` | Observed across-draw/refit ranges of constituent effects |
| `panel_observed_budget_or_refit_ranges.csv` | Observed across-draw/refit ranges of panel effects |

Standard-family comparisons pair `algorithm__uv+raman` with `algorithm__uv+raman+nir`. The selected-weight comparison pairs `krr__weight_zero` with `krr__weight_selected`. The residual comparison pairs `ridge_residual__zero` with `ridge_residual__selected`.

`refit_index` normalizes the saved budget-refit/bootstrap-refit identifiers. Conditions, training sizes and refit indices are not pooled into independent replication. Conditional intervals are withheld for incomplete three-repeat fivefold contexts, learning budgets and training-bootstrap stability tasks. Descriptive paired tables remain available when their matching roles are complete.

## Important columns

| Columns | Meaning and interpretation |
|---|---|
| `condition`, `training_size`, `refit_index` | Analysis context; preserve these keys when filtering or joining |
| `repeat`, `fold` | Zero-based outer evaluation identifiers |
| `sample_id`, `target`, `model` | Bottle, reference constituent and fitted strategy |
| `n_unique_bottles` | Distinct evaluated bottles, at most 139 |
| `n_prediction_visits` | Saved held-out visits; 417 is three visits for each of 139 bottles |
| `n_train`, `n_unique_train` | Training rows and distinct training bottles; identical for ordinary fits, potentially different for bootstrap fits |
| `n_train_min`, `n_train_max`, `n_unique_train_min`, `n_unique_train_max` | Training-count ranges in pooled constituent and panel tables; use these rather than interpreting the first-record `n_train` as a constant budget |
| `rmse`, `mae`, `bias` | Original concentration-unit errors in g/L; bias is prediction minus reference |
| `r2` | Pooled R-squared within the stated group; negative values are retained |
| `standardized_mse`, `macro_standardized_mse` | Squared error standardized by receiving-training constituent variance; the macro value weights eleven constituents equally |
| `common_scale_mse`, `macro_common_scale_mse` | Assessment using full outer-training constituent variance, retained for learning-budget comparisons |
| `before`, `after`, `before_model`, `after_model`, `contrast` | Directed paired comparison; keep direction when interpreting signs |
| `gain_pct` | Positive means lower error after the change; constituent gain uses RMSE and panel gain uses standardized MSE |
| `delta_mse_after_minus_before` | Positive means panel prediction worsens after the change |
| `rmse_difference_after_minus_before` | Positive means constituent prediction worsens, in g/L |
| `conditional_ci_low_pct`, `conditional_ci_high_pct` | Conditional percentile bounds for constituent RMSE gain |
| `conditional_delta_ci_low`, `conditional_delta_ci_high` | Conditional percentile bounds for panel delta MSE |
| `conditional_gain_ci_low_pct`, `conditional_gain_ci_high_pct` | Conditional percentile bounds for panel relative MSE gain |
| `rmse_difference_ci_low`, `rmse_difference_ci_high` | Conditional percentile bounds for the original-unit RMSE difference |
| `mean_fold_mse_difference_after_minus_before` | Fold-mean loss contrast used by the approximate corrected-CV calculation |
| `corrected_se`, `corrected_ci_low`, `corrected_ci_high` | Approximate corrected-CV standard error and interval, distinct from the bottle-bootstrap interval |
| `p_uncorrected`, `p_holm`, `planned_family_size`, `multiplicity_scope` | Approximate sensitivity P values and declared Holm family; not exact confirmatory inference |
| `degrees_of_freedom`, `df_convention`, `test_status` | Explicit test construction and handling of degenerate contrasts |
| `bootstrap_draws`, `bootstrap_seed`, `interval_scope` | Conditional-bootstrap construction and interpretation |
| `effective_nir_weight` | Shared NIR gate; zero means NIR is not required by that fitted panel |
| `effective_modality_count` | Two for a zero NIR gate, three for a positive NIR gate; ordinary fixed masks retain their actual count |
| `n_tasks`, `fraction`, `frequency_scope` | Dependent-task acquisition count and proportion; no binomial confidence limits |
| `n_observed_refit_groups`, columns ending `_mean`, `_min`, `_max`, `range_scope` | Across-group descriptive range; never relabel as a 95% interval |

## Concentrations and dimensions

`concentration_summary.csv` retains the source g/L unit and all eleven endpoints. It includes minimum, maximum, mean, sample SD, median, first/third quartiles, IQR, zero counts and negative counts. The source chemical identity/reporting-basis confirmation fields remain applicable; no concentrations are corrected by this analysis.

`spectral_dimensions_and_sample_ratios.csv` reports coordinate count `p`, sample count `n` and `p_over_n` for all seven masks. The sample counts are 139 for the cohort, 111/112 for ordinary outer training, and 40/70 for the learning budgets. Coordinate counts do not measure effective rank, independent chemical information or a universal sample-size requirement.

## Reading status and uncertainty together

Use a constituent point estimate with its **conditional** interval when available. Use five-refit minima/maxima as **observed ranges**. Use corrected-CV results as **approximate sensitivity**. These three outputs answer different questions and should not be merged into one unlabeled uncertainty column. Current archive postprocessing and synthetic software checks remain separate from new V8 fitting results.
