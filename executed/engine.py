import hashlib
import itertools
import json
import numpy as np
from legacy_engine import _validated, grouped_folds, _fit_predict
from preprocessing import transform_block, assemble_design, make_design, METHODS
from sopls import predict_sequential

MODALITIES = ('uv', 'raman', 'nir')
MASKS = (('uv',), ('raman',), ('nir',), ('uv', 'raman'), ('uv', 'nir'), ('raman', 'nir'), MODALITIES)
MASK_NAMES = tuple('+'.join(mask) for mask in MASKS)
ALGORITHMS = ('pls', 'ridge', 'krr', 'sopls')


def _algorithms(cfg):
    values = list(cfg.get('algorithms', ALGORITHMS))
    if not values or len(values) != len(set(values)) or not set(values).issubset(ALGORITHMS):
        raise ValueError('algorithms must be a nonempty unique subset of pls, ridge, krr and sopls')
    return values


def panel_names(cfg):
    values = list(cfg.get('panel_tolerances', [0, 0.02, 0.05, 0.10]))
    if not values or any(not np.isfinite(v) or v < 0 or v > 1 or abs(v*100-round(v*100)) > 1e-8 for v in values):
        raise ValueError('Panel tolerances must be percentages from zero to one')
    names = ['panel_t%03d' % int(round(v*100)) for v in values]
    if len(names) != len(set(names)):
        raise ValueError('Panel tolerances must be unique')
    return list(zip(names, map(float, values)))


def model_names(cfg):
    strategies = list(MASK_NAMES) + ['select_single', 'select_subset'] + [name for name, tolerance in panel_names(cfg)]
    return ['mean'] + [algorithm+'__'+strategy for algorithm in _algorithms(cfg) for strategy in strategies]


def _positive_grid(values, name, integer=False):
    values = list(values)
    if not values or any(not np.isfinite(v) or v <= 0 or (integer and int(v) != v) for v in values):
        raise ValueError(name+' must contain positive finite values')
    return list(dict.fromkeys(int(v) if integer else float(v) for v in values))


def _options(cfg, algorithm):
    values = cfg.get('sopls_preprocessing_options', cfg.get('preprocessing_options', {})) if algorithm == 'sopls' else cfg.get('preprocessing_options', {})
    result = {modality: list(values.get(modality, ['raw'])) for modality in MODALITIES}
    for modality, methods in result.items():
        if not methods or len(methods) != len(set(methods)) or not set(methods).issubset(METHODS):
            raise ValueError('Invalid preprocessing options for '+modality)
        if modality == 'uv' and set(methods) & {'snv', 'baseline_snv'}:
            raise ValueError('UV SNV is not part of the declared V6 preprocessing design')
        if modality != 'raman' and 'baseline_snv' in methods:
            raise ValueError('baseline_snv is declared only for the Raman block')
    return result


def candidate_grid(cfg):
    grid = cfg.get('grid', {})
    parameters = dict(
        pls_components=_positive_grid(grid.get('pls_components', [2, 5, 10, 15, 20]), 'pls_components', True),
        ridge_alphas=_positive_grid(grid.get('ridge_alphas', [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]), 'ridge_alphas'),
        krr_alphas=_positive_grid(grid.get('krr_alphas', [0.0001, 0.001, 0.01, 0.1, 1, 10]), 'krr_alphas'),
        krr_gammas=_positive_grid(grid.get('krr_gammas', [0.01, 0.1, 0.3, 1, 3]), 'krr_gammas'),
        sopls_components=_positive_grid(grid.get('sopls_components', [1, 2, 3]), 'sopls_components', True))
    result = []
    for algorithm in _algorithms(cfg):
        options = _options(cfg, algorithm)
        for mask in MASKS:
            for treatments in itertools.product(*(options[m] for m in mask)):
                preprocessing = dict(zip(mask, treatments))
                base = dict(algorithm=algorithm, mask='+'.join(mask), modalities=list(mask), preprocessing=preprocessing)
                if algorithm == 'pls':
                    variants = [dict(components=value) for value in parameters['pls_components']]
                elif algorithm == 'ridge':
                    variants = [dict(alpha=value) for value in parameters['ridge_alphas']]
                elif algorithm == 'krr':
                    variants = [dict(alpha=a, gamma=g) for a in parameters['krr_alphas'] for g in parameters['krr_gammas']]
                else:
                    variants = [dict(components=value, order=list(order)) for order in itertools.permutations(mask) for value in parameters['sopls_components']]
                for variant in variants:
                    result.append(dict(base, **variant))
    return [dict(spec, candidate_id=i) for i, spec in enumerate(result)]


def _design_key(spec):
    return tuple((modality, spec['preprocessing'][modality]) for modality in spec['modalities'])


def _prepare_training_banks(X, candidates, axes, settings):
    bank, metadata, errors = {}, {}, {}
    keys = sorted(set((modality, method) for spec in candidates for modality, method in spec['preprocessing'].items()))
    for key in keys:
        modality, method = key
        try:
            bank[key], metadata[key] = transform_block(X[modality], method, axes.get(modality), settings)
        except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError) as exc:
            errors[key] = type(exc).__name__+': '+str(exc)
    return bank, metadata, errors


def _assemble(bank, metadata, errors, spec, training, evaluation):
    keys = _design_key(spec)
    for key in keys:
        if key in errors:
            raise ValueError(errors[key])
    a = {modality: bank[(modality, method)][training] for modality, method in keys}
    b = {modality: bank[(modality, method)][evaluation] for modality, method in keys}
    meta = {modality: metadata[(modality, method)] for modality, method in keys}
    return assemble_design(a, b, meta)


def _predict(design, Y, spec, targets=None):
    if spec['algorithm'] != 'sopls':
        return _fit_predict(design, Y, spec, targets)
    q = Y.shape[1]
    targets = range(q) if targets is None else targets
    result = np.full((len(design['evaluation']), q), np.nan)
    errors, warnings, details = {}, {}, {}
    for j in targets:
        try:
            result[:, j], details[j], messages = predict_sequential(design['train_blocks'], design['evaluation_blocks'], Y[:, j], spec['order'], spec['components'])
            if messages:
                warnings[j] = messages
        except (ValueError, RuntimeError, RuntimeWarning, np.linalg.LinAlgError, FloatingPointError) as exc:
            errors[j] = type(exc).__name__+': '+str(exc)
    return result, errors, warnings, details


def _tie_key(index, scores, candidates, target):
    spec = candidates[index]
    treatment = tuple(spec['preprocessing'][m] for m in spec['modalities'])
    return scores[index, target], len(spec['modalities']), MASK_NAMES.index(spec['mask']), sum(m != 'raw' for m in treatment), treatment, index


def _boundaries(spec, cfg):
    grid = cfg.get('grid', {})
    keys = dict(pls={'components': 'pls_components'}, ridge={'alpha': 'ridge_alphas'}, krr={'alpha': 'krr_alphas', 'gamma': 'krr_gammas'}, sopls={'components': 'sopls_components'})
    records = []
    for parameter, name in keys[spec['algorithm']].items():
        values = grid.get(name, [])
        if values:
            value, lower, upper = spec[parameter], min(values), max(values)
            records.append(dict(parameter=parameter, selected=value, lower=lower, upper=upper, at_lower=bool(value == lower), at_upper=bool(value == upper)))
    return records


def select_and_predict(X_train, Y_train, train_ids, X_eval, cfg, seed, axes=None, return_oof=True):
    X, Y, ids, E = _validated(X_train, Y_train, train_ids, X_eval)
    axes = {} if axes is None else axes
    q = Y.shape[1]
    if np.any(Y.std(axis=0, ddof=1) <= 1e-12):
        raise ValueError('Every receiving-training target requires nonzero variation for equal-target panel selection')
    candidates = candidate_grid(cfg)
    folds = grouped_folds(ids, int(cfg.get('inner_folds', 3)), int(seed))
    shape = (len(candidates), q)
    sse, standardized_sse = np.zeros(shape), np.zeros(shape)
    counts, failed = np.zeros(shape, dtype=int), np.zeros(shape, dtype=bool)
    oof = np.full((len(candidates), len(Y), q), np.nan)
    records = [dict(spec=spec, failed_fold_targets=[], warning_count=0) for spec in candidates]
    bank, metadata, bank_errors = _prepare_training_banks(X, candidates, axes, cfg.get('preprocessing_settings', {}))
    split_records = []
    mean_oof = np.full_like(Y, np.nan)
    for fold, (tr, va) in enumerate(folds):
        scale = Y[tr].std(axis=0, ddof=1)
        if np.any(scale <= 1e-12):
            raise ValueError('An inner-fitting response has no variation; panel-normalized validation is undefined')
        split_records.append(dict(fold=fold, train_ids=ids[tr].tolist(), validation_ids=ids[va].tolist(), train_row_indices=tr.tolist(), validation_row_indices=va.tolist(), training_response_sd=scale.tolist()))
        mean_oof[va] = Y[tr].mean(axis=0)
        cache, cache_errors = {}, {}
        for i, spec in enumerate(candidates):
            key = _design_key(spec)
            try:
                if key in cache_errors:
                    raise ValueError(cache_errors[key])
                if key not in cache:
                    try:
                        cache[key] = _assemble(bank, metadata, bank_errors, spec, tr, va)
                    except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError) as exc:
                        cache_errors[key] = str(exc)
                        raise
                prediction, errors, messages, details = _predict(cache[key], Y[tr], spec)
            except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError) as exc:
                prediction = np.full((len(va), q), np.nan)
                errors = {j: type(exc).__name__+': '+str(exc) for j in range(q)}
                messages = {}
            records[i]['warning_count'] += sum(len(value) for value in messages.values())
            oof[i, va] = prediction
            for j in range(q):
                if j in errors or not np.isfinite(prediction[:, j]).all():
                    failed[i, j] = True
                    records[i]['failed_fold_targets'].append(dict(fold=fold, target_index=j, error=errors.get(j, 'Nonfinite prediction')))
                    continue
                error = float(np.sum((prediction[:, j]-Y[va, j])**2))
                sse[i, j] += error
                standardized_sse[i, j] += error/(scale[j]**2)
                counts[i, j] += len(va)
    valid = ~failed & (counts == len(Y))
    scores, standardized_scores = np.full(shape, np.inf), np.full(shape, np.inf)
    scores[valid] = sse[valid]/counts[valid]
    standardized_scores[valid] = standardized_sse[valid]/counts[valid]
    for i, record in enumerate(records):
        record.update(inner_mse_by_target=[float(value) if np.isfinite(value) else None for value in scores[i]], inner_standardized_mse_by_target=[float(value) if np.isfinite(value) else None for value in standardized_scores[i]], valid_by_target=valid[i].tolist(), n_scored_rows_by_target=counts[i].tolist())
    choices, panel_diagnostics = {}, {}
    for algorithm in _algorithms(cfg):
        for strategy in list(MASK_NAMES)+['select_single', 'select_subset']:
            eligible = [i for i, spec in enumerate(candidates) if spec['algorithm'] == algorithm and (strategy == 'select_subset' or (strategy == 'select_single' and len(spec['modalities']) == 1) or spec['mask'] == strategy)]
            selected = []
            for j in range(q):
                available = [i for i in eligible if valid[i, j]]
                if not available:
                    raise RuntimeError('All candidates failed for algorithm=%s strategy=%s target_index=%d' % (algorithm, strategy, j))
                selected.append(min(available, key=lambda i: _tie_key(i, scores, candidates, j)))
            choices[algorithm+'__'+strategy] = selected
        panel_scores = {mask: float(np.mean([standardized_scores[index, j] for j, index in enumerate(choices[algorithm+'__'+mask])])) for mask in MASK_NAMES}
        best = min(panel_scores.values())
        if not np.isfinite(best):
            raise RuntimeError('No fully valid panel candidate for '+algorithm)
        for name, tolerance in panel_names(cfg):
            eligible_masks = [mask for mask in MASK_NAMES if panel_scores[mask] <= best*(1+tolerance)+1e-15]
            mask = min(eligible_masks, key=lambda value: (len(value.split('+')), panel_scores[value], MASK_NAMES.index(value)))
            choices[algorithm+'__'+name] = list(choices[algorithm+'__'+mask])
            panel_diagnostics[algorithm+'__'+name] = dict(common_mask=mask, tolerance=tolerance, best_inner_standardized_mse=best, chosen_inner_standardized_mse=panel_scores[mask], all_mask_scores=panel_scores, interpretation='Predeclared predictive-error tolerance; not analytical or regulatory noninferiority')
    needed = {}
    for chosen in choices.values():
        for j, i in enumerate(chosen):
            needed.setdefault(i, set()).add(j)
    final_predictions, final_records, cache = {}, {}, {}
    evaluation_bank, evaluation_metadata, evaluation_errors = _prepare_training_banks(E, [candidates[i] for i in needed], axes, cfg.get('preprocessing_settings', {}))
    for i in sorted(needed):
        spec = candidates[i]
        key = _design_key(spec)
        for entry in key:
            if entry in bank_errors or entry in evaluation_errors:
                raise RuntimeError('Selected preprocessing failed at final fit: '+str(bank_errors.get(entry, evaluation_errors.get(entry))))
        if key not in cache:
            a = {m: bank[(m, method)] for m, method in key}
            b = {m: evaluation_bank[(m, method)] for m, method in key}
            meta = {m: metadata[(m, method)] for m, method in key}
            cache[key] = assemble_design(a, b, meta)
        prediction, errors, messages, details = _predict(cache[key], Y, spec, sorted(needed[i]))
        if errors:
            raise RuntimeError('Selected final fit failed for candidate %d: %s' % (i, errors))
        final_predictions[i] = prediction
        final_records[str(i)] = dict(spec=spec, fit_ids=ids.tolist(), preprocessing=cache[key]['preprocessing'], effective_by_target={str(j): value for j, value in details.items()}, warnings_by_target={str(j): value for j, value in messages.items()})
    output, selected_records, acquisition, selected_oof = {}, {}, {}, {}
    for model, chosen in choices.items():
        output[model] = np.column_stack([final_predictions[i][:, j] for j, i in enumerate(chosen)])
        if not np.isfinite(output[model]).all():
            raise RuntimeError('Nonfinite final prediction: '+model)
        selected_records[model] = [dict(target_index=j, candidate_id=i, spec=candidates[i], mask=candidates[i]['mask'], inner_mse=float(scores[i, j]), inner_standardized_mse=float(standardized_scores[i, j]), boundaries=_boundaries(candidates[i], cfg)) for j, i in enumerate(chosen)]
        union = [m for m in MODALITIES if any(m in candidates[i]['modalities'] for i in chosen)]
        acquisition[model] = dict(modality_union=union, n_modalities=len(union), per_target_masks=[candidates[i]['mask'] for i in chosen], mean_per_target_modalities=float(np.mean([len(candidates[i]['modalities']) for i in chosen])))
        if return_oof:
            selected_oof[model] = np.column_stack([oof[i, :, j] for j, i in enumerate(chosen)]).tolist()
    output['mean'] = np.repeat(Y.mean(axis=0, keepdims=True), len(next(iter(E.values()))), axis=0)
    acquisition['mean'] = dict(modality_union=[], n_modalities=0, per_target_masks=['none']*q, mean_per_target_modalities=0.0)
    selected_records['mean'] = []
    if return_oof:
        selected_oof['mean'] = mean_oof.tolist()
    if set(output) != set(model_names(cfg)):
        raise RuntimeError('Unexpected strategy set from engine')
    for algorithm in _algorithms(cfg):
        for selector in ['select_single', 'select_subset']+[name for name, value in panel_names(cfg)]:
            model = algorithm+'__'+selector
            for j, i in enumerate(choices[model]):
                fixed = algorithm+'__'+candidates[i]['mask']
                if not np.array_equal(output[model][:, j], output[fixed][:, j]):
                    raise AssertionError('Selector must reuse the selected fixed-mask fit')
                if return_oof and not np.array_equal(np.asarray(selected_oof[model])[:, j], np.asarray(selected_oof[fixed])[:, j]):
                    raise AssertionError('Selector OOF values must reuse the selected fixed-mask candidate')
    h = hashlib.sha256()
    h.update(json.dumps(ids.tolist()).encode('ascii'))
    for a in [Y]+[X[m] for m in MODALITIES]:
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    diagnostics = dict(training_ids=ids.tolist(), n_training_rows=len(Y), n_distinct_training_bottles=len(np.unique(ids)), n_targets=q, training_data_sha256=h.hexdigest(), seed=int(seed), inner_splits=split_records, candidate_scores=records, selections=selected_records, selection_by_model=selected_records, acquisition_by_model=acquisition, panel_selection=panel_diagnostics, final_fits=final_records, algorithms=_algorithms(cfg), fixed_masks=list(MASK_NAMES), target_weighting='Per-target tuning; shared-panel selection averages inner-fitting-SD-standardized validation errors equally across all endpoints', selection_scope='Receiving-training data only; evaluation labels never supplied', oof_scope='Chosen-candidate inner-held-out predictions from proper training only; selection also used these folds, so these are not independent outer-validation results', errors_policy='Any failed inner fold excludes that candidate for that target; a failed selected final fit aborts execution')
    if return_oof:
        diagnostics['oof_predictions'] = selected_oof
    return output, diagnostics
