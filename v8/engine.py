from pathlib import Path
from collections import Counter
import copy
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXECUTED = ROOT / 'executed'
if str(EXECUTED) not in sys.path:
    sys.path.insert(0, str(EXECUTED))
import engine as frozen_engine
import legacy_engine
import preprocessing as frozen_preprocessing

MODALITIES = ('uv', 'raman', 'nir')
PAIRED_MASKS = ('uv+raman', 'uv+raman+nir')
ERRORS = (ValueError, RuntimeError, RuntimeWarning, np.linalg.LinAlgError, FloatingPointError)


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _payload_hash(value):
    encoded = json.dumps(_plain(value), sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _training_hash(X, Y, ids):
    digest = hashlib.sha256()
    digest.update(json.dumps(ids.tolist()).encode('ascii'))
    for value in [Y] + [X[m] for m in MODALITIES]:
        digest.update(np.ascontiguousarray(value, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _setting(cfg, name, default):
    return cfg.get('v8', {}).get(name, cfg.get(name, default))


def _region_count(cfg):
    value = _setting(cfg, 'nir_interval_regions', 3)
    if not np.isfinite(value) or int(value) != value or value < 1:
        raise ValueError('The number of NIR intervals must be a positive integer')
    return int(value)


def _condition(task):
    value = task['condition']
    if value in ('nir_weight_learning', 'nir_weight_stability', 'weight_learning', 'weight_stability'):
        return 'nir_weight'
    if value not in ('unscaled', 'nir_extended', 'nir_intervals', 'balanced_budget', 'nir_weight'):
        raise ValueError('Unknown V8 engine condition: ' + str(value))
    return value


def _axis(axes, modality, n_features):
    value = np.asarray(axes.get(modality, np.arange(n_features)), dtype=float)
    if value.shape != (n_features,) or not np.isfinite(value).all() or (len(value) > 1 and not np.all(np.diff(value) > 0)):
        raise ValueError('Spectral coordinates must be finite, aligned and strictly increasing')
    return value


def _interval_indices(axis, regions=3):
    axis = np.asarray(axis, dtype=float)
    if axis.ndim != 1 or len(axis) < regions * 3 or not np.isfinite(axis).all() or not np.all(np.diff(axis) > 0):
        raise ValueError('NIR interval selection requires an increasing axis with at least three points per region')
    if int(regions) != regions or regions < 1:
        raise ValueError('The number of NIR intervals must be positive')
    boundaries = np.linspace(axis[0], axis[-1], int(regions) + 1)
    result = {'full': np.arange(len(axis), dtype=int)}
    for index in range(int(regions)):
        inside = (axis >= boundaries[index]) & ((axis <= boundaries[index + 1]) if index == regions - 1 else (axis < boundaries[index + 1]))
        selected = np.flatnonzero(inside)
        if len(selected) < 3:
            raise ValueError('An equal-coordinate NIR region contains fewer than three coordinates')
        result['region_' + str(index + 1)] = selected
    if not np.array_equal(np.sort(np.concatenate([result['region_' + str(i + 1)] for i in range(int(regions))])), np.arange(len(axis))):
        raise RuntimeError('NIR region boundaries do not partition the source coordinates')
    return result


def _msc_pair(training, evaluation):
    a = np.asarray(training, dtype=float)
    b = np.asarray(evaluation, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or a.shape[1] < 3 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('MSC requires aligned finite spectral matrices')
    reference = a.mean(axis=0)
    centered_reference = reference - reference.mean()
    denominator = float(centered_reference.dot(centered_reference))
    if denominator <= 1e-24:
        raise ValueError('The training-only MSC reference is constant')
    def transform(values):
        row_mean = values.mean(axis=1, keepdims=True)
        slope = (values - row_mean).dot(centered_reference) / denominator
        if np.any(~np.isfinite(slope)) or np.any(np.abs(slope) <= 1e-12):
            raise ValueError('A spectrum has an undefined MSC slope')
        intercept = row_mean[:, 0] - slope * reference.mean()
        corrected = (values - intercept[:, None]) / slope[:, None]
        if not np.isfinite(corrected).all():
            raise ValueError('MSC produced nonfinite spectra')
        return corrected
    return transform(a), transform(b), dict(method='msc', rowwise=False, n_reference_rows=len(a), msc_reference_sha256=frozen_preprocessing.digest(reference), msc_reference_mean=float(reference.mean()), reference_scope='Current fitting rows only, retaining any bootstrap multiplicities; the same fitted reference is applied to validation or evaluation')


def _processed_pair(training, evaluation, method, axis, settings):
    if method in ('msc', 'msc_sg1'):
        a, b, metadata = _msc_pair(training, evaluation)
        if method == 'msc_sg1':
            a, derivative = frozen_preprocessing.transform_block(a, 'sg1', axis, settings)
            b, unused = frozen_preprocessing.transform_block(b, 'sg1', axis, settings)
            metadata.update(derivative)
            metadata.update(method='msc_sg1', rowwise=False)
        return a, b, metadata
    a, metadata = frozen_preprocessing.transform_block(training, method, axis, settings)
    b, unused = frozen_preprocessing.transform_block(evaluation, method, axis, settings)
    return a, b, metadata


def _design_key(spec):
    return tuple(spec['modalities']), tuple((m, spec['preprocessing'][m]) for m in spec['modalities']), spec.get('nir_region', 'full'), spec.get('block_scaling', 'rms'), float(spec.get('nir_weight', 1.0))


def _make_design(X, E, axes, spec, cfg, transform_cache=None):
    transform_cache = {} if transform_cache is None else transform_cache
    training, evaluation, metadata = {}, {}, {}
    regions = None
    if 'nir' in spec['modalities'] and spec.get('nir_region', 'full') != 'full':
        regions = _interval_indices(_axis(axes, 'nir', X['nir'].shape[1]), _region_count(cfg))
    for modality in spec['modalities']:
        method = spec['preprocessing'][modality]
        region = spec.get('nir_region', 'full') if modality == 'nir' else 'full'
        key = (modality, method, region)
        if key not in transform_cache:
            try:
                a, b = X[modality], E[modality]
                coordinates = _axis(axes, modality, a.shape[1])
                if modality == 'nir' and regions is not None:
                    indices = regions[region]
                    a, b, coordinates = a[:, indices], b[:, indices], coordinates[indices]
                at, bt, meta = _processed_pair(a, b, method, coordinates, cfg.get('preprocessing_settings', {}))
                meta.update(nir_region=region, source_selected_coordinates=int(len(coordinates)), coordinate_min=float(coordinates[0]), coordinate_max=float(coordinates[-1]))
                transform_cache[key] = (at, bt, meta)
            except ERRORS as exc:
                transform_cache[key] = exc
        transformed = transform_cache[key]
        if isinstance(transformed, BaseException):
            raise ValueError(str(transformed))
        training[modality], evaluation[modality], metadata[modality] = transformed
    if spec.get('block_scaling', 'rms') == 'none':
        a, b, details = {}, {}, {}
        for modality in spec['modalities']:
            mean = training[modality].mean(axis=0)
            a[modality], b[modality] = training[modality] - mean, evaluation[modality] - mean
            details[modality] = dict(metadata[modality], n_fit=len(training[modality]), n_features=training[modality].shape[1], training_mean_sha256=frozen_preprocessing.digest(mean), block_rms=1.0, scaling='Training column centering only; no block-energy division')
        design = dict(train=np.concatenate(list(a.values()), axis=1), evaluation=np.concatenate(list(b.values()), axis=1), train_blocks=a, evaluation_blocks=b, preprocessing=details)
    else:
        design = frozen_preprocessing.assemble_design(training, evaluation, metadata)
    weight = float(spec.get('nir_weight', 1.0))
    if 'nir' in spec['modalities'] and weight != 1.0:
        if weight <= 0 or not np.isfinite(weight):
            raise ValueError('Zero NIR weight must use the exact U+R candidate class')
        design['train_blocks']['nir'] = design['train_blocks']['nir'] * np.sqrt(weight)
        design['evaluation_blocks']['nir'] = design['evaluation_blocks']['nir'] * np.sqrt(weight)
        design['train'] = np.concatenate([design['train_blocks'][m] for m in spec['modalities']], axis=1)
        design['evaluation'] = np.concatenate([design['evaluation_blocks'][m] for m in spec['modalities']], axis=1)
        design['preprocessing']['nir']['distance_weight'] = weight
    if not np.isfinite(design['train']).all() or not np.isfinite(design['evaluation']).all():
        raise ValueError('A spectral design contains nonfinite values')
    return design


def _candidates(cfg, condition):
    source_cfg = copy.deepcopy(cfg)
    if condition == 'nir_weight':
        source_cfg['algorithms'] = ['krr']
    base = [copy.deepcopy(s) for s in frozen_engine.candidate_grid(source_cfg) if s['mask'] in PAIRED_MASKS]
    result = []
    if condition == 'nir_extended':
        for spec in base:
            result.append(spec)
            if 'nir' in spec['modalities'] and spec['preprocessing']['nir'] == 'raw':
                for method in ('msc', 'msc_sg1'):
                    extra = copy.deepcopy(spec)
                    extra['preprocessing']['nir'] = method
                    result.append(extra)
    elif condition == 'nir_intervals':
        regions = ['full'] + ['region_' + str(i + 1) for i in range(_region_count(cfg))]
        for spec in base:
            for region in regions if 'nir' in spec['modalities'] else ['full']:
                extra = copy.deepcopy(spec)
                extra['nir_region'] = region
                result.append(extra)
    elif condition == 'nir_weight':
        weights = list(_setting(cfg, 'nir_weights', [0, 0.25, 1, 4]))
        if not weights or len(set(weights)) != len(weights) or 0 not in weights or any(not np.isfinite(w) or w < 0 for w in weights):
            raise ValueError('NIR weights must be distinct finite nonnegative values and include zero')
        for weight in sorted(weights):
            wanted = 'uv+raman' if weight == 0 else 'uv+raman+nir'
            for spec in base:
                if spec['mask'] == wanted:
                    extra = copy.deepcopy(spec)
                    extra['nir_weight'] = float(weight)
                    result.append(extra)
    else:
        result = base
    for index, spec in enumerate(result):
        spec['candidate_id'] = index
        if condition == 'unscaled':
            spec['block_scaling'] = 'none'
    return result


def _split_records(ids, folds, Y):
    result = []
    for index, (training, validation) in enumerate(folds):
        scale = Y[training].std(axis=0, ddof=1)
        if np.any(~np.isfinite(scale)) or np.any(scale <= 1e-12):
            raise ValueError('An inner-fitting constituent has zero variance')
        result.append(dict(fold=index, train_ids=ids[training].tolist(), validation_ids=ids[validation].tolist(), train_row_indices=training.tolist(), validation_row_indices=validation.tolist(), training_response_sd=scale.tolist()))
    return result


def _score(X, Y, ids, axes, cfg, candidates, seed):
    folds = legacy_engine.grouped_folds(ids, int(cfg.get('inner_folds', 3)), int(seed))
    splits = _split_records(ids, folds, Y)
    shape = len(candidates), Y.shape[1]
    sse, normalized = np.zeros(shape), np.zeros(shape)
    counts, failed = np.zeros(shape, dtype=int), np.zeros(shape, dtype=bool)
    failures, preprocessing_records = [], []
    for fold, (tr, va) in enumerate(folds):
        xt = {m: X[m][tr] for m in MODALITIES}
        xv = {m: X[m][va] for m in MODALITIES}
        scale = np.asarray(splits[fold]['training_response_sd'])
        transforms, designs = {}, {}
        for index, spec in enumerate(candidates):
            key = _design_key(spec)
            try:
                if key not in designs:
                    try:
                        designs[key] = _make_design(xt, xv, axes, spec, cfg, transforms)
                        preprocessing_records.append(dict(fold=fold, design_key=_plain(key), preprocessing=designs[key]['preprocessing']))
                    except ERRORS as exc:
                        designs[key] = exc
                if isinstance(designs[key], BaseException):
                    raise ValueError(str(designs[key]))
                predictions, errors, warnings, details = frozen_engine._predict(designs[key], Y[tr], spec)
            except ERRORS as exc:
                predictions = np.full((len(va), Y.shape[1]), np.nan)
                errors = {j: str(exc) for j in range(Y.shape[1])}
            for target in range(Y.shape[1]):
                if target in errors or not np.isfinite(predictions[:, target]).all():
                    failed[index, target] = True
                    failures.append(dict(fold=fold, candidate_id=spec['candidate_id'], target_index=target, error=str(errors.get(target, 'Nonfinite prediction'))))
                else:
                    error = float(np.sum((predictions[:, target] - Y[va, target]) ** 2))
                    normalized_error = error / scale[target] ** 2
                    if not np.isfinite(error) or not np.isfinite(normalized_error):
                        failed[index, target] = True
                        failures.append(dict(fold=fold, candidate_id=spec['candidate_id'], target_index=target, error='Nonfinite validation loss'))
                        continue
                    sse[index, target] += error
                    normalized[index, target] += normalized_error
                    counts[index, target] += len(va)
    valid = (~failed) & (counts == len(Y)) & np.isfinite(sse) & np.isfinite(normalized)
    raw, standardized = np.full(shape, np.inf), np.full(shape, np.inf)
    raw[valid] = sse[valid] / counts[valid]
    standardized[valid] = normalized[valid] / counts[valid]
    records = [dict(spec=spec, inner_mse_by_target=_plain(raw[i]), inner_standardized_mse_by_target=_plain(standardized[i]), valid_by_target=valid[i].tolist(), n_scored_rows_by_target=counts[i].tolist()) for i, spec in enumerate(candidates)]
    return raw, standardized, dict(inner_splits=splits, candidate_scores=records, candidate_failures=failures, preprocessing_fits=preprocessing_records, reused_archived_inner_scores=False)


def _archived_scores(X, Y, ids, axes, cfg, task, archived):
    if 'payload' in archived:
        payload = archived['payload']
        if _payload_hash(payload) != archived.get('payload_sha256'):
            raise ValueError('Archived payload hash is invalid')
        verified_hash = archived['payload_sha256']
    else:
        payload = archived
        verified_hash = _payload_hash(payload)
    diag = payload.get('diagnostics', {})
    seed = int(task['seed'])
    if payload.get('stage') != 'benchmark' or payload.get('condition') != 'enhanced' or payload.get('training_size') != 'full':
        raise ValueError('Balanced full-budget reuse requires the original enhanced full-training benchmark')
    if payload.get('repeat') != task.get('repeat') or payload.get('fold') != task.get('fold'):
        raise ValueError('Archived outer identity does not match the receiving task')
    if payload.get('fit_ids') != ids.tolist() or diag.get('training_ids') != ids.tolist():
        raise ValueError('Archived training bottle order differs')
    if diag.get('training_data_sha256') != _training_hash(X, Y, ids) or diag.get('seed') != seed:
        raise ValueError('Archived training arrays or inner seed differ')
    settings = cfg.get('preprocessing_settings', {})
    setting_keys = dict(baseline_lambda=('baseline_lambda', 100000.0), baseline_asymmetry=('baseline_asymmetry', 0.01), baseline_iterations=('baseline_iterations', 10))
    seen_axes = set()
    for fit in diag.get('final_fits', {}).values():
        for modality, metadata in fit.get('preprocessing', {}).items():
            expected_axis = frozen_preprocessing.digest(_axis(axes, modality, X[modality].shape[1]))
            if metadata.get('source_axis_sha256') != expected_axis:
                raise ValueError('Archived spectral coordinate arrays differ')
            seen_axes.add(modality)
            for name, (key, default) in setting_keys.items():
                if name in metadata and metadata[name] != settings.get(key, default):
                    raise ValueError('Archived physical preprocessing settings differ')
            if metadata.get('method') == 'sg1':
                length = int(metadata['n_features'])
                window = min(int(settings.get('sg_window', 11)), length if length % 2 else length - 1)
                degree = min(int(settings.get('sg_polyorder', 2)), window - 1)
                if metadata.get('sg_window') != window or metadata.get('sg_polyorder') != degree:
                    raise ValueError('Archived derivative preprocessing settings differ')
    if seen_axes != set(MODALITIES):
        raise ValueError('Archived coordinate provenance is incomplete')
    grid = frozen_engine.candidate_grid(cfg)
    records = diag.get('candidate_scores', [])
    if len(records) != len(grid) or any(r['spec'] != s for r, s in zip(records, grid)):
        raise ValueError('Archived candidate grid differs from the declared frozen grid')
    folds = legacy_engine.grouped_folds(ids, int(cfg.get('inner_folds', 3)), seed)
    splits = _split_records(ids, folds, Y)
    old_splits = diag.get('inner_splits', [])
    if len(old_splits) != len(splits):
        raise ValueError('Archived inner split count differs')
    for previous, current in zip(old_splits, splits):
        for key in ('train_ids', 'validation_ids', 'train_row_indices', 'validation_row_indices', 'training_response_sd'):
            if previous.get(key) != current[key]:
                raise ValueError('Archived inner roles or response scales differ')
    selected = [r for r in records if r['spec']['mask'] in PAIRED_MASKS]
    shape = len(selected), Y.shape[1]
    raw, standardized = np.full(shape, np.inf), np.full(shape, np.inf)
    for index, record in enumerate(selected):
        for key in ('inner_mse_by_target', 'inner_standardized_mse_by_target', 'valid_by_target', 'n_scored_rows_by_target'):
            if len(record.get(key, [])) != Y.shape[1]:
                raise ValueError('Archived score target dimensions differ')
        for target in range(Y.shape[1]):
            if record['valid_by_target'][target]:
                values = record['inner_mse_by_target'][target], record['inner_standardized_mse_by_target'][target]
                if record['n_scored_rows_by_target'][target] != len(Y) or any(v is None or not np.isfinite(v) or v < 0 for v in values):
                    raise ValueError('An archived valid score is incomplete or nonfinite')
                raw[index, target], standardized[index, target] = values
    return [copy.deepcopy(r['spec']) for r in selected], raw, standardized, dict(inner_splits=splits, reused_archived_inner_scores=True, archived_payload_sha256=verified_hash, archived_training_data_sha256=diag['training_data_sha256'], archive_validation='Payload integrity, complete ordered training arrays, inner roles, response scales, source candidate specifications and score completeness verified', candidate_scores=selected, candidate_failures=[], preprocessing_fits=[])


def _tie(index, scores, candidates, target):
    spec = candidates[index]
    treatment = tuple(spec['preprocessing'][m] for m in spec['modalities'])
    return scores[index, target], len(spec['modalities']), frozen_engine.MASK_NAMES.index(spec['mask']), sum(m != 'raw' for m in treatment), treatment, spec.get('nir_region', 'full'), index


def _best(indices, raw, candidates, q):
    winners = []
    for target in range(q):
        eligible = [i for i in indices if np.isfinite(raw[i, target])]
        if not eligible:
            raise RuntimeError('All permitted candidates failed for target ' + str(target))
        winners.append(min(eligible, key=lambda i: _tie(i, raw, candidates, target)))
    return winners


def _balanced_indices(candidates, cfg, task):
    seed = int(task['seed']) + int(_setting(cfg, 'balanced_seed_offset', 1700000)) + int(task.get('budget_refit', 0))
    random = np.random.RandomState(seed)
    result, records = {}, []
    for algorithm in frozen_engine._algorithms(cfg):
        groups = {mask: [i for i, spec in enumerate(candidates) if spec['algorithm'] == algorithm and spec['mask'] == mask] for mask in PAIRED_MASKS}
        budget = min(map(len, groups.values()))
        if budget < 1:
            raise ValueError('A balanced comparison is missing a paired candidate class')
        for mask in PAIRED_MASKS:
            indices = np.array(groups[mask], dtype=int)
            chosen = sorted(random.choice(indices, size=budget, replace=False).tolist())
            result[algorithm + '__' + mask] = chosen
            records.append(dict(algorithm=algorithm, mask=mask, available_candidates=len(indices), selected_candidates=budget, selected_candidate_ids=[candidates[i]['candidate_id'] for i in chosen]))
    return result, dict(seed=seed, budget_refit=int(task.get('budget_refit', 0)), allocation=records, scope='Equal candidate counts for paired masks within each algorithm; complete finite-grid scores reused where supplied')


def _finalize(X, Y, ids, E, axes, cfg, candidates, choices, raw, standardized):
    needed = {}
    for winners in choices.values():
        for target, index in enumerate(winners):
            needed.setdefault(index, set()).add(target)
    transforms, designs, predictions, fits = {}, {}, {}, {}
    for index in sorted(needed):
        spec = candidates[index]
        key = _design_key(spec)
        if key not in designs:
            designs[key] = _make_design(X, E, axes, spec, cfg, transforms)
        prediction, errors, warnings, details = frozen_engine._predict(designs[key], Y, spec, sorted(needed[index]))
        if errors or any(not np.isfinite(prediction[:, j]).all() for j in needed[index]):
            raise RuntimeError('A selected final fit failed: ' + str(errors))
        predictions[index] = prediction
        fits[str(spec['candidate_id'])] = dict(spec=spec, fit_ids=ids.tolist(), preprocessing=designs[key]['preprocessing'], effective_by_target=_plain(details), warnings_by_target=_plain(warnings))
    output, selections, acquisitions = {}, {}, {}
    for model, winners in choices.items():
        output[model] = np.column_stack([predictions[i][:, j] for j, i in enumerate(winners)])
        if not np.isfinite(output[model]).all():
            raise RuntimeError('Nonfinite V8 prediction: ' + model)
        selections[model] = [dict(target_index=j, candidate_id=candidates[i]['candidate_id'], spec=candidates[i], mask=candidates[i]['mask'], inner_mse=float(raw[i, j]), inner_standardized_mse=float(standardized[i, j])) for j, i in enumerate(winners)]
        union = [m for m in MODALITIES if any(m in candidates[i]['modalities'] for i in winners)]
        acquisitions[model] = dict(modality_union=union, n_modalities=len(union), n_acquisition_modalities=len(union), per_target_masks=[candidates[i]['mask'] for i in winners], mean_per_target_modalities=float(np.mean([len(candidates[i]['modalities']) for i in winners])))
        weights = {float(candidates[i]['nir_weight']) for i in winners if 'nir_weight' in candidates[i]}
        if len(weights) == 1 and all('nir_weight' in candidates[i] for i in winners):
            acquisitions[model]['effective_nir_weight'] = next(iter(weights))
    return output, selections, acquisitions, fits


def evaluate_task(X_train, Y_train, train_ids, X_eval, axes, cfg, task, archived_payload=None):
    X, Y, ids, E = legacy_engine._validated(X_train, Y_train, train_ids, X_eval)
    axes = {} if axes is None else axes
    if np.any(Y.std(axis=0, ddof=1) <= 1e-12):
        raise ValueError('Every training constituent requires nonzero variation')
    condition, seed = _condition(task), int(task['seed'])
    if condition == 'balanced_budget' and archived_payload is not None:
        candidates, raw, standardized, diagnostics = _archived_scores(X, Y, ids, axes, cfg, task, archived_payload)
    else:
        if condition == 'balanced_budget' and cfg.get('profile') == 'full':
            raise ValueError('Full balanced-budget tasks require verified original task caches')
        candidates = _candidates(cfg, condition)
        raw, standardized, diagnostics = _score(X, Y, ids, axes, cfg, candidates, seed)
    choices, shared = {}, {}
    if condition == 'nir_weight':
        weight_scores, weight_winners, invalid_weights = {}, {}, []
        for weight in sorted(set(float(s['nir_weight']) for s in candidates)):
            indices = [i for i, spec in enumerate(candidates) if spec['nir_weight'] == weight]
            try:
                winners = _best(indices, raw, candidates, Y.shape[1])
            except RuntimeError as exc:
                if weight == 0:
                    raise
                invalid_weights.append(dict(nir_weight=weight, reason=str(exc)))
                continue
            score = float(np.mean([standardized[i, j] for j, i in enumerate(winners)]))
            if not np.isfinite(score):
                raise RuntimeError('A common NIR weight has an invalid constituent score')
            weight_scores[weight], weight_winners[weight] = score, winners
        selected = min(weight_scores, key=lambda weight: (weight_scores[weight], weight))
        choices['krr__weight_zero'] = weight_winners[0.0]
        choices['krr__weight_selected'] = weight_winners[selected]
        shared = dict(common_nir_weight=float(selected), weight_scores=[dict(nir_weight=weight, inner_standardized_mse=weight_scores[weight]) for weight in sorted(weight_scores)], excluded_weights=invalid_weights, target_weighting='Equal mean across all constituents after target-specific raw-unit tuning', zero_class='Exact U+R candidates; the NIR block is neither transformed nor included in zero-weight designs')
    elif condition == 'balanced_budget':
        permitted, budget = _balanced_indices(candidates, cfg, task)
        choices = {model: _best(indices, raw, candidates, Y.shape[1]) for model, indices in permitted.items()}
        diagnostics['balanced_budget'] = budget
    else:
        for algorithm in frozen_engine._algorithms(cfg):
            for mask in PAIRED_MASKS:
                eligible = [i for i, spec in enumerate(candidates) if spec['algorithm'] == algorithm and spec['mask'] == mask]
                choices[algorithm + '__' + mask] = _best(eligible, raw, candidates, Y.shape[1])
    predictions, selections, acquisition, final_fits = _finalize(X, Y, ids, E, axes, cfg, candidates, choices, raw, standardized)
    counts = Counter((spec['algorithm'], spec['mask']) for spec in candidates)
    diagnostics.update(condition=condition, training_ids=ids.tolist(), n_training_rows=len(Y), n_distinct_training_bottles=len(np.unique(ids)), n_targets=Y.shape[1], training_data_sha256=_training_hash(X, Y, ids), seed=seed, selections=selections, selection_by_model=selections, acquisition_by_model=acquisition, final_fits=final_fits, candidate_counts=[dict(algorithm=a, mask=m, n_candidates=n) for (a, m), n in sorted(counts.items())], n_candidates=len(candidates), shared_weight=shared, selection_scope='Only receiving-training arrays and labels; no evaluation responses are accepted', target_weighting='Target-specific raw-unit MSE tuning; shared choices average inner-fitting-variance-standardized errors equally', failure_policy='Any failed inner fold excludes the candidate for the affected target; selected final-fit failure aborts the task')
    return dict(predictions=predictions, diagnostics=_plain(diagnostics))
