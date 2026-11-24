# Statistical analysis

The analysis compares prediction strategies for the complete eleven-constituent panel. All eleven constituents retain equal weight after error standardization. Original concentration-unit results are also retained for every constituent.

The archived V6 results have been postprocessed without fitting new models. V8 model performance becomes available only after the corresponding new tasks are completed. A synthetic test pass or an archive analysis registry does not establish that a V8 model experiment has run.

## Observational unit and paired comparisons

The independent observational unit is a bottle. The archive contains 139 bottles, evaluated once per repeat in three repetitions of fivefold outer evaluation. Therefore, a complete strategy has 417 prediction visits per constituent and 15 dependent outer tasks. Neither number increases the independent bottle count.

Comparisons match condition, training budget, refit or budget-draw index, outer repeat, outer fold, bottle and constituent. The two strategies must have identical reference values and fitting-response scales. Missing or inconsistent paired bottle roles are rejected. Full-size conditional analyses additionally require all 139 bottles in every repeat, all eleven constituents, fifteen outer tasks and 111/112 distinct training bottles per task.

| Quantity | Definition | Favorable direction |
|---|---|---|
| Constituent RMSE | Square root of mean squared error across saved held-out visits, in g/L | Lower |
| Panel standardized MSE | Equal mean across eleven constituents and three repeats after dividing each squared error by its corresponding outer-training constituent variance | Lower |
| Panel delta MSE | MSE(after) minus MSE(before) | Negative means the added or selected strategy improves prediction |
| Constituent RMSE difference | RMSE(after) minus RMSE(before), in g/L | Negative |
| Relative gain | 100 times [1 minus error(after)/error(before)] | Positive |

Constituent relative gain uses RMSE; panel relative gain uses standardized MSE. These percentages are different quantities and must retain their labels. A zero or undefined reference error can make a relative gain undefined. The absolute difference remains the appropriate primary quantity in that situation.

Learning-budget tables additionally retain a common scale based on the complete outer-training response SD. This scale is used for assessment across training budgets and is not supplied to fitting or selection.

## Conditional paired bottle bootstrap

The default analysis uses 2,000 draws with seed 20261104. Within each bottle and constituent, it first averages squared errors across the three saved held-out visits. It then samples 139 bottle indices with replacement. A single index matrix is shared across all eleven constituents, models and comparisons. This preserves pairing and retains the complete outcome panel for every selected bottle. Calculations use chunks to limit memory use.

For every draw, the program recomputes constituent RMSE differences, constituent relative gains, panel standardized MSE differences and panel relative gains. The reported interval endpoints are the 2.5th and 97.5th percentiles. The registry records the seed, draw count and index-matrix hash.

These are conditional intervals for frozen cross-validation predictions. They describe changes in empirical evaluation-bottle weights while holding fitted models, hyperparameters, preprocessing choices and modality selections fixed. They do not include the changes that would occur if training bottles were resampled and the pipeline retrained. Cross-bottle dependence caused by overlapping training sets also remains. The intervals are not simultaneous intervals and do not carry a guarantee of population-level coverage for the complete learning procedure.

Use the label **conditional 95% bootstrap interval of frozen OOF predictions**. An interval excluding zero is descriptive evidence under that construction, not an automatically multiplicity-adjusted significance finding.

The archive analysis provides all nine directed single-to-double and double-to-triple additions for each of four algorithms. New-run conditional intervals are produced only for complete, paired, full-size three-repeat fivefold contexts. Learning and grouped-bootstrap stability contexts receive separate descriptive summaries. A quick profile does not meet the full conditional-analysis requirements.

## Approximate corrected repeated-CV sensitivity

The analysis also reports a clearly labelled approximate sensitivity calculation based on the Nadeau–Bengio correction and the repeated-fold extension. It uses the paired mean squared-loss difference within each outer task:

`SE_corrected^2 = [1/J + mean(n_test/n_train)] * sample_variance(task_loss_differences)`

Here J is 15 for a complete context; the test/train ratio is approximately 0.25. The implementation uses J minus one degrees of freedom and reports that convention explicitly. Compared with treating fifteen folds as independent, the correction increases the standard error by approximately 2.18 times at a ratio of 0.25.

Constituent calculations use raw-unit MSE differences; panel calculations use equal-target standardized MSE differences. These are means of fold losses, whereas the headline RMSE and MSE summaries weight bottles equally. Fold sizes of 27/28 create a small difference between these estimands. The sensitivity statistic and its P value must not be relabelled as an exact test of the pooled relative-gain estimate.

Ordinary independent-fold t tests and ordinary Wilcoxon tests are not used. The corrected calculation is still approximate: it does not repair retrospective workflow development or provide external validation. A constant nonzero task difference receives an undefined P value rather than an invented finite significance level. An exact zero self-contrast receives P = 1.

## Multiplicity

Holm adjustment is applied separately within each declared contrast and analysis context:

| Comparison family | Planned family size |
|---|---:|
| Four standard algorithms, panel effects | 4 |
| Four standard algorithms, all constituent effects | 44 |
| One selected-weight or residual-gate comparison, panel effect | 1 |
| One selected-weight or residual-gate comparison, all constituent effects | 11 |

Each directed archive addition is a separate family. Adjustment is not claimed across all nine additions, all conditions or all budget draws jointly. The tables retain the family size, unadjusted P value and adjusted P value. Holm adjustment cannot make an invalid underlying P value valid; these values remain approximate CV sensitivity results.

## Training perturbations and acquisition frequencies

Five grouped-bootstrap refits within each of fifteen outer tasks produce 75 dependent fitting tasks. The five refit-index groups pool predictions across the outer tasks; their observed minimum and maximum effects are sensitivity ranges. They are not confidence limits. Training resampling also changes the number of distinct bottles and their weights.

The five candidate-budget draws are kept as separate contexts. Their observed ranges describe budget-draw sensitivity; they are not five independent studies. Learning budgets and stability refits are not pooled into the ordinary full-size conditional analysis.

Selection frequencies are reported as observed counts and proportions. No binomial interval is attached to a frequency across dependent tasks. A selected NIR weight of zero represents two effective modalities; a positive weight represents three. Acquisition records retain this distinction.

## Scope of additional computation

Reading saved predictions, calculating conditional intervals and calculating corrected-CV sensitivity require no model refitting. A training-procedure bootstrap or a dedicated CV-error variance design requires new fitting, tuning and selection under the appropriate sampling design. More analysis of the same cohort does not create independent external measurements.
