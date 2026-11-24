import numpy as np

from executed.legacy_engine import _fit_predict, _validated, grouped_folds, make_design


MODELS = ('ridge_residual__zero', 'ridge_residual__selected')
BASE_MASK = ('uv', 'raman')
NIR_MASK = ('nir',)


def _positive_grid(values, name):
    result = [float(value) for value in values]
    if not result or any(not np.isfinite(value) or value <= 0 for value in result):
        raise ValueError(name + ' must contain positive finite values')
    if len(result) != len(set(result)):
        raise ValueError(name + ' must contain unique values')
    return result


def _settings(cfg):
    values = cfg.get('v8', {})
    base = _positive_grid(values.get('residual_base_alphas', [0.01, 0.1, 1, 10]), 'residual_base_alphas')
    correction = _positive_grid(values.get('residual_correction_alphas', [0.01, 0.1, 1, 10]), 'residual_correction_alphas')
    multipliers = [float(value) for value in values.get('residual_multipliers', [0, 0.5, 1])]
    if not multipliers or len(multipliers) != len(set(multipliers)) or 0.0 not in multipliers:
        raise ValueError('residual_multipliers must contain unique values including zero')
    if any(value not in (0.0, 0.5, 1.0) for value in multipliers):
        raise ValueError('residual_multipliers must be a subset of zero, one half and one')
    inner_value = cfg.get('inner_folds', 3)
    subfold_value = values.get('residual_subfolds', 3)
    inner = int(inner_value)
    subfolds = int(subfold_value)
    if inner != inner_value or subfolds != subfold_value:
        raise ValueError('Grouped fold counts must be integers')
    if inner < 2 or subfolds < 2:
        raise ValueError('At least two grouped folds are required')
    return dict(base_alphas=base, correction_alphas=correction,
                multipliers=sorted(multipliers), inner_folds=inner, subfolds=subfolds)


def _offset_seed(seed, offset):
    return (int(seed) + int(offset)) % (2 ** 32 - 1)


def _slice_blocks(blocks, indices):
    return {name: values[indices] for name, values in blocks.items()}


def _roles(train, validation, ids, source_indices, seed):
    if set(ids[train]) & set(ids[validation]):
        raise ValueError('Bottle identity overlaps grouped fitting and validation roles')
    return dict(train_indices=train.tolist(), validation_indices=validation.tolist(),
                train_row_indices=train.tolist(), validation_row_indices=validation.tolist(),
                source_train_indices=source_indices[train].tolist(),
                source_validation_indices=source_indices[validation].tolist(),
                train_ids=ids[train].tolist(), validation_ids=ids[validation].tolist(),
                n_train=len(train), n_validation=len(validation),
                n_unique_train=len(np.unique(ids[train])),
                n_unique_validation=len(np.unique(ids[validation])), seed=int(seed))


def _ridge_predictions(design, response, alpha):
    result, errors, warnings, details = _fit_predict(
        design, response, dict(algorithm='ridge', alpha=float(alpha)))
    if errors or not np.isfinite(result).all():
        raise RuntimeError('Residual-stage ridge fit failed: ' + str(errors))
    return result


def _crossfitted_base(blocks, response, ids, alphas, n_splits, seed, source_indices=None):
    ids = np.asarray(ids, dtype=str)
    source_indices = np.arange(len(ids)) if source_indices is None else np.asarray(source_indices)
    folds = grouped_folds(ids, int(n_splits), int(seed))
    predictions = {alpha: np.full(response.shape, np.nan) for alpha in alphas}
    roles = []
    counts = np.zeros(len(ids), dtype=int)
    for train, validation in folds:
        design = make_design(_slice_blocks(blocks, train), _slice_blocks(blocks, validation), BASE_MASK, snv=False)
        for alpha in alphas:
            predictions[alpha][validation] = _ridge_predictions(design, response[train], alpha)
        counts[validation] += 1
        roles.append(_roles(train, validation, ids, source_indices, seed))
    if not np.all(counts == 1) or any(not np.isfinite(values).all() for values in predictions.values()):
        raise RuntimeError('Cross-fitted residuals require exactly one finite held-out prediction per training row')
    return predictions, roles


def _pipeline_bank(blocks, response, ids, evaluation, settings, seed, source_indices):
    base_design = make_design(blocks, evaluation, BASE_MASK, snv=False)
    base_predictions = {alpha: _ridge_predictions(base_design, response, alpha)
                        for alpha in settings['base_alphas']}
    correction_predictions = {}
    subfold_roles = []
    if any(multiplier > 0 for multiplier in settings['multipliers']):
        crossfit, subfold_roles = _crossfitted_base(
            blocks, response, ids, settings['base_alphas'], settings['subfolds'], seed,
            source_indices=source_indices)
        correction_design = make_design(blocks, evaluation, NIR_MASK, snv=False)
        for base_alpha in settings['base_alphas']:
            residual = response - crossfit[base_alpha]
            for correction_alpha in settings['correction_alphas']:
                correction_predictions[base_alpha, correction_alpha] = _ridge_predictions(
                    correction_design, residual, correction_alpha)
    return base_predictions, correction_predictions, subfold_roles


def _candidates(settings):
    result = []
    for multiplier in settings['multipliers']:
        for base_alpha in settings['base_alphas']:
            corrections = [None] if multiplier == 0 else settings['correction_alphas']
            for correction_alpha in corrections:
                modalities = ['uv', 'raman'] if multiplier == 0 else ['uv', 'raman', 'nir']
                result.append(dict(candidate_id=len(result), algorithm='ridge_residual',
                                   base_alpha=base_alpha, correction_alpha=correction_alpha,
                                   multiplier=multiplier, effective_nir_weight=multiplier,
                                   mask='+'.join(modalities), modalities=modalities,
                                   preprocessing={modality: 'raw' for modality in modalities},
                                   block_scaling='Training-only centered spectral RMS norm',
                                   residual_target='Grouped cross-fitted U+R residual for a fixed declared base alpha'))
    return result


def _candidate_prediction(spec, base_predictions, correction_predictions):
    prediction = base_predictions[spec['base_alpha']]
    if spec['multiplier'] == 0:
        return prediction
    return prediction + spec['multiplier'] * correction_predictions[spec['base_alpha'], spec['correction_alpha']]


def _choose_within_multiplier(candidates, mse, standardized_mse, multiplier):
    eligible = [index for index, spec in enumerate(candidates) if spec['multiplier'] == multiplier]
    chosen = [min(eligible, key=lambda index: (float(mse[index, target]), index))
              for target in range(mse.shape[1])]
    score = float(np.mean([standardized_mse[index, target] for target, index in enumerate(chosen)]))
    return chosen, score


def evaluate_residual(X_train, Y_train, train_ids, X_eval, axes, cfg, task):
    blocks, response, ids, evaluation = _validated(X_train, Y_train, train_ids, X_eval)
    settings = _settings(cfg)
    seed = int(task.get('seed', cfg.get('seed', 0)))
    folds = grouped_folds(ids, settings['inner_folds'], seed)
    candidates = _candidates(settings)
    oof = np.full((len(candidates), len(ids), response.shape[1]), np.nan)
    row_variance = np.full(response.shape, np.nan)
    inner_roles = []
    subfold_roles = []
    row_counts = np.zeros(len(ids), dtype=int)
    for fold_index, (train, validation) in enumerate(folds):
        fitting_blocks = _slice_blocks(blocks, train)
        validation_blocks = _slice_blocks(blocks, validation)
        sub_seed = _offset_seed(seed, 300000 + fold_index)
        base_predictions, correction_predictions, roles = _pipeline_bank(
            fitting_blocks, response[train], ids[train], validation_blocks,
            settings, sub_seed, train)
        for index, spec in enumerate(candidates):
            oof[index, validation] = _candidate_prediction(spec, base_predictions, correction_predictions)
        variance = response[train].var(axis=0, ddof=1)
        variance = np.where(np.isfinite(variance) & (variance > 1e-28), variance, 1.0)
        row_variance[validation] = variance
        row_counts[validation] += 1
        role = _roles(train, validation, ids, np.arange(len(ids)), seed)
        role['training_response_sd'] = np.sqrt(variance).tolist()
        inner_roles.append(role)
        subfold_roles.append(dict(inner_fold_index=fold_index, seed=sub_seed,
                                  fitting_source_indices=train.tolist(), folds=roles,
                                  nuisance_parameter_policy='Each declared base alpha evaluated as fixed; no held-out label used for nuisance tuning'))
    if not np.all(row_counts == 1) or not np.isfinite(oof).all() or not np.isfinite(row_variance).all():
        raise RuntimeError('Complete-pipeline inner validation did not cover every training row exactly once')
    squared_error = (oof - response[None, :, :]) ** 2
    mse = squared_error.mean(axis=1)
    standardized_mse = (squared_error / row_variance[None, :, :]).mean(axis=1)
    choices = {}
    scores = {}
    for multiplier in settings['multipliers']:
        choices[multiplier], scores[multiplier] = _choose_within_multiplier(
            candidates, mse, standardized_mse, multiplier)
    chosen_multiplier = min(settings['multipliers'], key=lambda value: (scores[value], value))
    model_choices = {MODELS[0]: choices[0.0], MODELS[1]: choices[chosen_multiplier]}
    final_seed = _offset_seed(seed, 400000)
    base_predictions, correction_predictions, final_roles = _pipeline_bank(
        blocks, response, ids, evaluation, settings, final_seed, np.arange(len(ids)))
    predictions = {}
    selections = {}
    acquisitions = {}
    for model, selected in model_choices.items():
        prediction = np.empty((len(next(iter(evaluation.values()))), response.shape[1]))
        records = []
        for target, index in enumerate(selected):
            spec = dict(candidates[index])
            prediction[:, target] = _candidate_prediction(spec, base_predictions, correction_predictions)[:, target]
            records.append(dict(target_index=target, candidate_id=index, mask=spec['mask'], spec=spec,
                                multiplier=spec['multiplier'], inner_mse=float(mse[index, target]),
                                inner_standardized_mse=float(standardized_mse[index, target]),
                                inner_panel_standardized_mse=scores[spec['multiplier']]))
        if not np.isfinite(prediction).all():
            raise RuntimeError('Residual correction generated nonfinite outer predictions')
        predictions[model] = prediction
        selections[model] = records
        union = ['uv', 'raman'] if records[0]['multiplier'] == 0 else ['uv', 'raman', 'nir']
        acquisitions[model] = dict(modality_union=union, n_modalities=len(union),
                                   common_multiplier=records[0]['multiplier'],
                                   effective_nir_weight=records[0]['multiplier'])
    candidate_scores = [dict(candidate_id=index, spec=spec,
                             inner_mse=mse[index].tolist(),
                             inner_standardized_mse=standardized_mse[index].tolist())
                        for index, spec in enumerate(candidates)]
    diagnostics = dict(
        selections=selections, acquisition_by_model=acquisitions,
        inner_splits=inner_roles, inner_residual_subfolds=subfold_roles,
        final_residual_crossfit_splits=final_roles,
        final_residual_crossfit_seed=final_seed,
        candidate_scores=candidate_scores,
        shared_multiplier_scores=[dict(multiplier=value, inner_panel_standardized_mse=scores[value],
                                       selected_candidate_ids=choices[value]) for value in settings['multipliers']],
        chosen_multiplier=chosen_multiplier,
        candidate_counts=dict(complete_pipeline_candidates=len(candidates), base_alphas=len(settings['base_alphas']),
                              correction_alphas=len(settings['correction_alphas']),
                              shared_multipliers=len(settings['multipliers'])),
        settings=settings, seed=seed, n_targets=response.shape[1], n_training_rows=len(ids),
        n_distinct_training_bottles=len(np.unique(ids)), training_ids=ids.tolist(),
        selection_scope='Complete-pipeline inner validation on receiving training bottles only',
        residual_scope='Every residual uses a fixed-base-alpha grouped held-out prediction; no adaptive nuisance choice is made inside residual generation',
        target_tuning='Minimum pooled raw-unit inner MSE within each shared multiplier',
        panel_tuning='Equal target mean of inner-fitting-variance-standardized errors after target-specific raw-MSE tuning',
        tie_rule='Candidate order within each multiplier; lower multiplier at equal panel score',
        interpretation='Incremental prediction diagnostic for the declared baseline and correction class; not a conditional-independence or chemical-selectivity test',
        evaluation_labels_received=False, axes_used_for_transform=False)
    return dict(predictions=predictions, diagnostics=diagnostics)
