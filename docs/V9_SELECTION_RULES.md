# Frozen selection rules for the per-block SO-PLS extension

The condition is `sopls_perblock`. The existing historical and V8 numerical results remain unchanged and are included in the distribution. This extension is specified after inspecting earlier results but before calculating its own held-out outcomes. It is not an externally preregistered or new-cohort prospective study. Every new outcome is retained, including an outcome that qualifies the earlier interpretation.

## Fixed scope

Exactly fifteen outer tasks reuse the archived three-repeat, five-fold bottle identities and row order. The base seed is 20261029; the task seed is base + 10000 * repeat + 100 * fold. Each receiving training partition uses three grouped inner folds from the existing frozen grouped-fold generator. All eleven constituents are tuned and reported equally. The two masks are U+R and U+R+N; neither mask changes and zero components are not a candidate.

The component candidate list is fixed at [1, 2, 3, 5] per block. Use the historical SO-PLS preprocessing candidates: UV raw; Raman raw, SNV or baseline plus SNV; NIR raw, SNV or first derivative. Every declared block order is retained: two orders for U+R and six for U+R+N. There are six and fifty-four complete preprocessing/order paths, respectively. Do not add MSC, extra intervals, another component value or another order after seeing the extension outcomes.

## Greedy prefix selection

For each target and complete preprocessing/order path, score the four first-block candidates using the same three inner validation partitions and pooled raw-unit squared error. Select k1 by lowest pooled MSE, breaking exact ties with the smaller k. Then hold k1 fixed and score the four two-block prefixes to choose k2. For the three-block mask, hold the first two chosen counts fixed and score the four three-block prefixes to choose k3.

Each scored prefix is independently fitted on the current inner fitting rows. Its preprocessing, training centering and block scaling use those rows only; its earlier blocks, score-space projection and response residuals are fitted there only. Compare the cumulative prefix prediction with the original inner-validation response. Do not construct residuals using a fit on all outer-training rows before the inner split. Safe prefix caches may reuse only identical fold, target, ordered modality/preprocessing prefix and component vector.

After the greedy stages, select a complete path separately for each target using: lowest final-stage pooled raw MSE; then smaller total requested components; lexicographically smaller ordered component vector; canonical block order; declared preprocessing order; and path ID. Ties mean exact equality of the computed score. The final chosen vector and path are refitted on the entire receiving outer-training partition. Outer test responses are supplied only after prediction for evaluation and are never accepted by the fitting API.

All intermediate candidate k values, pooled and fold scores, prefix selections, the final vector, block order, preprocessing, final active component counts and fitting warnings are saved. Ordered k1/k2/k3 must always be read with the block order; k_uv/k_raman/k_nir are also exported. The third ordered count is empty for U+R.

## Frozen fit arithmetic and failure handling

`executed/sopls.py` is byte-for-byte unchanged. `v9/sopls_vector.py` adapts only component-count normalization and the per-block requested count; its PLS fitting, orthogonalization, rank caps, residual updates and prediction arithmetic retain the frozen implementation. Uniform vectors are verified against the original scalar entry point on synthetic inputs.

Any candidate-prefix error or selected final-fit error aborts its task and stops later tasks. Save the failure identity and completeness state. Do not skip a failed candidate, replace a failed model or consult outer labels for a fallback. Existing verified completed tasks can be resumed; changed source, design, scientific runtime or parent evidence requires a new output directory, and an edited frozen design is rejected.

## Statistical report and stopping rule

Report each outer task's two macro standardized MSE values and the relative gain from adding NIR, all eleven target errors, ordered and modality-mapped component counts, inner scores, and original uniform-component SO-PLS alongside the new variant. The full conditional analysis requires all fifteen new tasks. Use 2000 paired bottle-bootstrap draws, seed 20261104, carrying all repeated visits and all eleven targets together. These intervals are conditional on the saved predictions and are not a bootstrap of complete retraining.

For the primary NIR-addition contrast, expand the original family to five methods: PLS, Ridge, RBF-KRR, uniform SO-PLS and per-block SO-PLS. Recalculate Holm-adjusted approximate corrected-CV P values in new joined tables: five panel comparisons and fifty-five target comparisons. Declare a secondary family for uniform versus per-block SO-PLS within the two fixed masks: two panel and twenty-two target comparisons. Original historical P-value files are retained unchanged. Do not add a standalone one-method family for the new primary result.

The original uniform candidate list [1,2,3,5,10] remains unchanged. The new vector list limits the maximum requested totals to 10 and 15, versus the old maxima 20 and 30. The contrast therefore tests component allocation and stronger regularization together; it does not establish that the old upper limit was too small. Inner scores are selection scores; all conclusions use held-out predictions with the stated conditional and approximate inference limitations. Retain every direction and stop after this fixed extension.
