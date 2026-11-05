from pathlib import Path
import hashlib
import itertools
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'executed') not in sys.path:
    sys.path.insert(0, str(ROOT / 'executed'))
from legacy_engine import _validated, grouped_folds
from preprocessing import transform_block, assemble_design, METHODS
from v9.sopls_vector import FROZEN_SOURCE_SHA256, predict_sequential

MODALITIES = ('uv', 'raman', 'nir')
MASKS = (('uv', 'raman'), MODALITIES)


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
        if not np.isfinite(value):
            raise ValueError('Nonfinite structural-test diagnostics cannot be completed')
        return float(value)
    return value


def _training_hash(X, Y, ids):
    result = hashlib.sha256()
    result.update(json.dumps(ids.tolist()).encode('ascii'))
    for values in [Y] + [X[m] for m in MODALITIES]:
        result.update(np.ascontiguousarray(values, dtype=np.float64).tobytes())
    return result.hexdigest()


def _settings(cfg):
    perblock = cfg.get('perblock', {})
    grid = list(perblock.get('components', [1, 2, 3, 5]))
    if grid != [1, 2, 3, 5]:
        raise ValueError('The prespecified per-block grid is exactly [1, 2, 3, 5]')
    options = cfg.get('sopls_preprocessing_options', cfg.get('preprocessing_options', {}))
    result = {}
    for modality in MODALITIES:
        methods = list(options.get(modality, ['raw']))
        if not methods or len(set(methods)) != len(methods) or not set(methods).issubset(METHODS):
            raise ValueError('Invalid SO-PLS preprocessing options: ' + modality)
        if modality == 'uv' and set(methods) != {'raw'}:
            raise ValueError('The UV block must retain the prespecified raw treatment')
        if modality != 'raman' and 'baseline_snv' in methods:
            raise ValueError('Baseline-SNV is declared for Raman only')
        result[modality] = methods
    folds = int(cfg.get('inner_folds', 3))
    if folds < 2:
        raise ValueError('At least two grouped inner folds are required')
    return grid, result, folds


def _paths(options):
    result = []
    for mask in MASKS:
        for preprocessing_rank, methods in enumerate(itertools.product(*(options[m] for m in mask))):
            for order_rank, order in enumerate(itertools.permutations(mask)):
                result.append(dict(path_id=len(result), mask='+'.join(mask), modalities=list(mask),
                                   preprocessing=dict(zip(mask, methods)), order=list(order),
                                   order_rank=order_rank, preprocessing_rank=preprocessing_rank))
    return result


def _tie(path, selected):
    vector = tuple(selected['components_by_block'])
    return (selected['inner_mse'], sum(vector), vector, path['order_rank'],
            path['preprocessing_rank'], path['path_id'])


def evaluate_task(X_train, Y_train, train_ids, X_eval, axes, cfg, task):
    if task.get('condition') != 'sopls_perblock':
        raise ValueError('This engine supports the sopls_perblock condition only')
    X, Y, ids, E = _validated(X_train, Y_train, train_ids, X_eval)
    if Y.shape[1] != 11:
        raise ValueError('Every structural task must retain all eleven targets')
    if np.any(Y.std(axis=0, ddof=1) <= 1e-12):
        raise ValueError('Every training target must have nonzero variance')
    axes = {} if axes is None else axes
    grid, options, n_folds = _settings(cfg)
    paths = _paths(options)
    seed = int(task.get('seed', cfg.get('seed', 20261029)))
    folds = grouped_folds(ids, n_folds, seed)
    inner_roles = []
    for index, (training, validation) in enumerate(folds):
        sd = Y[training].std(axis=0, ddof=1)
        if not np.isfinite(sd).all() or np.any(sd <= 1e-12):
            raise ValueError('Every inner-fitting target must have nonzero variance')
        inner_roles.append(dict(fold=index, train_ids=ids[training].tolist(),
                                validation_ids=ids[validation].tolist(),
                                train_row_indices=training.tolist(), validation_row_indices=validation.tolist(),
                                training_response_sd=sd.tolist()))
    transformed, preprocessing_records = {}, {}
    inner_cache, final_cache = {}, {}
    counts = dict(inner_prefix_calls=0, inner_prefix_cache_hits=0,
                  inner_requested_block_fits=0, inner_active_block_fits=0,
                  final_prefix_calls=0, final_requested_block_fits=0, final_active_block_fits=0)

    def blocks(role, methods, order):
        training, evaluation, metadata = {}, {}, {}
        if role == 'final':
            receiving = np.arange(len(ids))
            evaluating = None
        else:
            receiving, evaluating = folds[int(role)]
        for modality in order:
            method = methods[modality]
            key = role, modality, method
            if key not in transformed:
                a, meta = transform_block(X[modality][receiving], method, axes.get(modality), cfg.get('preprocessing_settings', {}))
                b, unused = transform_block(E[modality] if role == 'final' else X[modality][evaluating], method,
                                            axes.get(modality), cfg.get('preprocessing_settings', {}))
                design = assemble_design({modality: a}, {modality: b}, {modality: meta})
                transformed[key] = design['train_blocks'][modality], design['evaluation_blocks'][modality], design['preprocessing'][modality]
                preprocessing_records[str(key)] = dict(role=role, modality=modality, method=method,
                                                       train_ids=ids[receiving].tolist(), metadata=design['preprocessing'][modality])
            training[modality], evaluation[modality], metadata[modality] = transformed[key]
        return training, evaluation, metadata

    def score_prefix(path, target, vector):
        order = path['order'][:len(vector)]
        prefix = tuple((m, path['preprocessing'][m]) for m in order)
        parts = []
        for fold_index, (training, validation) in enumerate(folds):
            key = fold_index, target, prefix, tuple(vector)
            if key in inner_cache:
                counts['inner_prefix_cache_hits'] += 1
                parts.append(inner_cache[key])
                continue
            try:
                a, b, metadata = blocks(fold_index, path['preprocessing'], order)
                prediction, details, messages = predict_sequential(a, b, Y[training, target], order, vector)
                if prediction.shape != (len(validation),) or not np.isfinite(prediction).all():
                    raise ValueError('Invalid held-out prefix predictions')
                squared = (prediction - Y[validation, target]) ** 2
                sd = float(inner_roles[fold_index]['training_response_sd'][target])
                part = dict(fold=fold_index, n_validation_rows=len(validation), squared_error_sum=float(squared.sum()),
                            standardized_squared_error_sum=float(squared.sum() / sd ** 2),
                            mse=float(squared.mean()), fit_details=details, warnings=messages)
                counts['inner_prefix_calls'] += 1
                counts['inner_requested_block_fits'] += len(order)
                counts['inner_active_block_fits'] += sum(int(item.get('components_active', 0)) > 0 for item in details.get('blocks', []))
                inner_cache[key] = part
                parts.append(part)
            except Exception as error:
                raise RuntimeError('Structural candidate failed; aborting task: path=%s target=%s stage=%s components=%s fold=%s: %s: %s' %
                                   (path['path_id'], target, len(vector), vector, fold_index, type(error).__name__, error)) from error
        n = sum(p['n_validation_rows'] for p in parts)
        if n != len(Y):
            raise RuntimeError('Inner validation must score every receiving-training row once')
        return dict(inner_mse=sum(p['squared_error_sum'] for p in parts) / n,
                    inner_standardized_mse=sum(p['standardized_squared_error_sum'] for p in parts) / n,
                    n_scored_rows=n,
                    fold_scores=[dict(fold=p['fold'], n_validation_rows=p['n_validation_rows'], mse=p['mse'],
                                      fit_details=p['fit_details'], warnings=p['warnings']) for p in parts])

    candidate_paths = []
    eligible = {}
    for path in paths:
        path_record = dict(path, target_paths=[])
        for target in range(Y.shape[1]):
            selected_vector, stages = [], []
            for position, modality in enumerate(path['order']):
                scores = []
                for k in grid:
                    vector = selected_vector + [int(k)]
                    scores.append(dict(k=int(k), components_by_block=vector,
                                       **score_prefix(path, target, vector)))
                best = min(scores, key=lambda record: (record['inner_mse'], record['k']))
                selected_vector = list(best['components_by_block'])
                stages.append(dict(stage=position + 1, modality=modality, fixed_prefix=selected_vector[:-1],
                                   selected_k=best['k'], selected_inner_mse=best['inner_mse'], candidates=scores))
            selected = dict(target_index=target, path_id=path['path_id'], components_by_block=selected_vector,
                            inner_mse=best['inner_mse'], inner_standardized_mse=best['inner_standardized_mse'],
                            stages=stages)
            path_record['target_paths'].append(selected)
            eligible.setdefault((path['mask'], target), []).append((path, selected))
        candidate_paths.append(path_record)
    predictions, selections, final_fits, acquisitions = {}, {}, {}, {}
    for mask in MASKS:
        name = '+'.join(mask)
        model = 'sopls_perblock__' + name
        prediction = np.empty((len(next(iter(E.values()))), Y.shape[1]))
        selected_records = []
        model_fits = []
        for target in range(Y.shape[1]):
            path, chosen = min(eligible[(name, target)], key=lambda pair: _tie(*pair))
            vector = chosen['components_by_block']
            order = path['order']
            key = target, tuple((m, path['preprocessing'][m]) for m in order), tuple(vector)
            try:
                if key not in final_cache:
                    a, b, metadata = blocks('final', path['preprocessing'], order)
                    values, details, messages = predict_sequential(a, b, Y[:, target], order, vector)
                    final_cache[key] = values, details, messages, metadata
                    counts['final_prefix_calls'] += 1
                    counts['final_requested_block_fits'] += len(order)
                    counts['final_active_block_fits'] += sum(int(item.get('components_active', 0)) > 0 for item in details.get('blocks', []))
                values, details, messages, metadata = final_cache[key]
                if values.shape != (len(prediction),) or not np.isfinite(values).all():
                    raise ValueError('Nonfinite final structural prediction')
                prediction[:, target] = values
            except Exception as error:
                raise RuntimeError('Selected structural final fit failed; aborting task: model=%s target=%s: %s: %s' %
                                   (model, target, type(error).__name__, error)) from error
            spec = dict(algorithm='sopls_perblock', path_id=path['path_id'], mask=name, modalities=list(mask),
                        order=list(order), preprocessing=dict(path['preprocessing']), components_by_block=list(vector),
                        components=list(vector), component_map=dict(zip(order, vector)))
            selected_records.append(dict(target_index=target, candidate_id=path['path_id'], path_id=path['path_id'],
                                         mask=name, spec=spec, inner_mse=chosen['inner_mse'],
                                         inner_standardized_mse=chosen['inner_standardized_mse'],
                                         final_components_active=[int(item.get('components_active', 0)) for item in details.get('blocks', [])],
                                         final_components_capped=[int(item.get('components_capped', 0)) for item in details.get('blocks', [])],
                                         stage_inner_scores=[dict(stage=s['stage'], modality=s['modality'], selected_k=s['selected_k'], inner_mse=s['selected_inner_mse']) for s in chosen['stages']]))
            model_fits.append(dict(target_index=target, spec=spec, fit_ids=ids.tolist(),
                                   preprocessing=metadata, details=details, warnings=messages))
        predictions[model] = prediction
        selections[model] = selected_records
        final_fits[model] = model_fits
        acquisitions[model] = dict(modality_union=list(mask), n_modalities=len(mask), n_acquisition_modalities=len(mask))
    counts['total_actual_block_fits'] = counts['inner_active_block_fits'] + counts['final_active_block_fits']
    diagnostics = dict(condition='sopls_perblock', training_ids=ids.tolist(), n_training_rows=len(Y),
                       n_distinct_training_bottles=len(np.unique(ids)), n_targets=11, seed=seed,
                       training_data_sha256=_training_hash(X, Y, ids), inner_splits=inner_roles,
                       candidate_paths=candidate_paths, selections=selections, selection_by_model=selections,
                       final_fits=final_fits, acquisition_by_model=acquisitions,
                       preprocessing_fits=list(preprocessing_records.values()), fit_counts=counts,
                       n_paths=len(paths), paths_per_mask={'+'.join(mask): sum(p['mask'] == '+'.join(mask) for p in paths) for mask in MASKS},
                       components_grid=grid, frozen_kernel_source_sha256=FROZEN_SOURCE_SHA256,
                       selection_scope='Receiving-training labels and grouped inner validation only; no evaluation responses accepted',
                       sequential_rule='For each order and preprocessing path, choose each block k by pooled raw-unit inner validation MSE after refitting the complete selected prefix in every inner fitting partition',
                       target_weighting='Each target is tuned independently in original response units; all eleven remain in reporting',
                       tie_rule='Stage: raw MSE then lower k. Final path: raw MSE, lower total requested components, lexicographic ordered vector, canonical order rank, preprocessing option rank, path ID',
                       failure_policy='Any declared candidate, transform or final-fit exception aborts the complete task; no exclusion or outer-label fallback',
                       model_scope='Greedy per-block allocation with maximum five requested components per block; not a test of whether the historical upper bound ten should be increased',
                       evaluation_labels_received=False)
    return dict(predictions=predictions, diagnostics=_plain(diagnostics))
