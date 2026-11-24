# Version 8 experimental decisions

## Study question and estimand

The question is whether adding NIR to UV plus Raman reduces held-out concentration-prediction error for an equally weighted eleven-component panel under limited training bottles, and whether a training-only common acquisition rule is stable. All eleven targets and all seven fixed combinations remain public in the historical benchmark. The new tests target incremental NIR, rather than repeatedly searching for a favorable primary analyte. No prediction-interval-width experiment is added.

The historical cohort contains 139 independent bottles, with 911 UV, 1,019 Raman and 1,001 NIR channels. The complete three-block predictor count is 2,931, so p/n is 2931/111 to 2931/112 in full outer fits, approximately 26.4 to 26.2. The notation n:p approximately 111:2931 describes the inverse ratio. Lower training budgets have still larger p/n. Concentration medians, interquartile ranges, extrema and all-target standard deviations are exported; no inference of a broad deployment population follows from these data.

## Adopted recommendations

| Recommendation | Implementation | Scope and trade-off |
|---|---|---|
| Block-aware control in principal comparisons | Preserve all-order SO-PLS in the seven-subset historical benchmark and include it in the three new paired physical/search sensitivities plus the unscaled control. | Single-response SO-PLS; not a guarantee that all possible multiblock methods were tested. |
| Bottle-level paired inference | One shared resample of bottles carries all repeated predictions, models and targets together; 2,000 resamples in the full profile. | Conditional on fitted predictions and observed split design. |
| Pairing across outer tasks | share fold-target errors and approximate corrected repeated-CV sensitivity tests with prespecified Holm families. | Ordinary paired t/Wilcoxon tests treating overlapping tasks as independent are not used. |
| Scope the no-gain statement | Restrict conclusions to the declared model classes, this cohort and the actual training-bottle count. | Null evidence cannot establish absence of spectral information or equivalence without a justified margin. |
| Independent-batch validation | Declare absent; keep the evidence limitation explicit. | New external samples cannot be replaced by resampling this cohort. |
| Conditional NIR information diagnostic | Grouped cross-fitted Ridge baseline residuals, nested complete-pipeline validation, NIR correction and a common panel multiplier. | CCA is omitted: high-dimensional small-n conditional associations would add complexity without establishing independent predictive gain. The selected baseline penalty can differ from the zero-correction penalty, so report the complete pipeline increment including retuning. |
| NIR preprocessing and wavelength sensitivity | Add MSC and MSC plus first derivative; additionally select full NIR or one of three fixed equal-coordinate intervals. | Training-only reference and transforms; no unconfirmed absorbance conversion or chemical band labeling. |
| Match candidate-search budgets | Five seeded equal-count candidate draws within every family and each outer task; reuse verified training-only historical inner scores, then refit selected outer models. | Same count does not imply equal effective flexibility or independent selection noise. Retain all five draws. |
| Block scaling | Compare training-energy normalization with no energy normalization and a shared NIR block weight. | Total-variance and root-mean-square-energy normalization are not falsely counted as independent methods. |
| Applicability and reproducibility | Public 15 x 7 x 4 panel-fold table plus 4,620 target-fold rows, concentration summaries, dimension ratios, new-run source/data/runtime signatures and environment freeze. | Actual historical BLAS/runtime metadata not recorded at execution are not reconstructed from the new machine. |

## Nested data roles

The full profile reuses the historical three-repeat five-fold outer bottle identities and order, with seed 20261029 + 10000 x repeat + 100 x fold. The indices are preserved even when not in numeric bottle order. Every preprocessing reference, centering/scaling constant, candidate score and shared acquisition choice is fitted using the receiving training split only. Evaluation responses are not accepted by the model API.

Reduced budgets use the first 40 or 70 rows of the seeded permutation of the outer training indices, with seed outer_seed + 700000. These prefixes are nested and each occurs once per outer task. Five bootstrap refits sample the outer training rows with replacement, using seed outer_seed + 100000 + refit_index. Repeated copies of a bottle remain together in every inner and residual cross-fitting split. Both draw count and distinct-bottle count are recorded.

Candidate-budget draw seeds are outer_seed + 1700000 + budget_refit. The smaller U+R candidate count is applied to each of the two masks within each algorithm; all available smaller-mask candidates may therefore be retained. A draw never uses outer error to decide its candidate set. The full profile verifies payload SHA, ordered training arrays, axes, inner bottle roles, response scales and the archived candidate grid before using historical inner scores. Quick-profile scores are calculated afresh and never substituted into full-profile evidence.

## Preprocessing and scaling

The current training mean spectrum is the MSC reference; row-wise intercept and slope are fitted against that reference and applied to each row. MSC plus first derivative applies the same declared Savitzky-Golay settings after MSC. Any invalid transform or candidate-fold fit excludes that candidate for the affected target; selected final-fit failures abort the task instead of falling back to an outer-label-selected model.

The three NIR intervals divide the observed numeric coordinate range equally. They are fixed before any response fit. The full-spectrum option is also retained; the interval and preprocessing choice is made inside inner validation. Derivatives follow the existing coordinate-aware preprocessing implementation. Raman is not silently treated as uniformly spaced when it is not.

Centered training blocks are normally divided by sqrt(mean of squared row Euclidean norms), which gives equal total training block energy. Unscaled means retaining centering and omitting this division. For the RBF-KRR gate, the already standardized NIR block is multiplied by the square root of a distance weight w in 0, 0.25, 1 or 4. Its contribution to squared Euclidean kernel distance is multiplied by w. Median-distance bandwidth normalization is recomputed on each training design. Zero uses the identical U+R design and does not perform NIR preprocessing in the final prediction design.

## Residual diagnostic

For every fixed baseline Ridge penalty candidate and every receiving inner-fitting set, grouped subfold predictions generate U+R held-out residuals. An NIR Ridge corrector is fitted to these residuals on that receiving set. The complete base-plus-correction candidate is then scored on its inner validation set. No adaptive choice of baseline penalty sees the held-out responses used to construct that bottle's own nuisance prediction.

The full candidate grid uses baseline and correction penalties 0.01, 0.1, 1 and 10; correction multipliers are 0, 0.5 and 1. Raw-unit MSE selects penalties separately for each target within a multiplier. Equal mean inner-fitting-variance-standardized error selects the common multiplier across the eleven targets. The outer final fit generates grouped cross-fitted residuals anew on all receiving training rows and refits the selected base and correction models. The zero correction is independently tuned U+R Ridge and requires only UV and Raman. This diagnostic is deliberately restricted to one regularized baseline/correction family; it does not prove NIR chemical selectivity or causal information.

## Fixed report and stopping rule

The extension is specified after examining earlier results and is identified as retrospective. Freeze this configuration before running; share every target, family, condition, budget, draw and refit, regardless of direction. Do not expand grids repeatedly until significance appears. Report new results only after the completeness file verifies all 255 planned full tasks. Treat the five refits and five budget draws descriptively; a full learning-procedure bootstrap would require a separate much more expensive design.

The package captures actual new-run Python, packages, source and input hashes, NumPy build, BLAS libraries/thread counts, float64 precision and pip snapshot. Resume rejects changed scientific context. Pinned package installation is a reproducibility target, not a promise of bitwise identity across every CPU or operating system.
