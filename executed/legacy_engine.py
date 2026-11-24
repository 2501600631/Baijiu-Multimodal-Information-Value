import hashlib
import json
import warnings
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import ConvergenceWarning
MODALITIES = ('uv', 'raman', 'nir')
MASKS = (('uv',), ('raman',), ('nir',), ('uv', 'raman'), ('uv', 'nir'), ('raman', 'nir'), MODALITIES)
ALGORITHMS = ('pls', 'ridge', 'krr')
MASK_NAMES = tuple(('+'.join(m) for m in MASKS))
STRATEGIES = tuple((a + '__' + m for a in ALGORITHMS for m in MASK_NAMES + ('select_single', 'select_subset'))) + ('mean',)

def _digest(value):
    a = np.ascontiguousarray(value, dtype=np.float64)
    return hashlib.sha256(str(a.shape).encode('ascii') + a.tobytes()).hexdigest()

def _validated(X, Y, ids, X_eval):
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2 or not np.isfinite(Y).all() or Y.shape[1] < 1:
        raise ValueError('Y_train must be a finite two-dimensional target array')
    ids = np.asarray(ids, dtype=str)
    if ids.ndim != 1 or len(ids) != len(Y):
        raise ValueError('Training identities must align with target rows')
    xt, xe = ({}, {})
    n_eval = None
    for m in MODALITIES:
        a, b = (np.asarray(X[m], dtype=float), np.asarray(X_eval[m], dtype=float))
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != len(Y) or (a.shape[1] != b.shape[1]):
            raise ValueError('Training/evaluation spectrum dimensions disagree: ' + m)
        if a.shape[1] < 1 or not np.isfinite(a).all() or (not np.isfinite(b).all()):
            raise ValueError('Finite, nonempty spectral blocks required: ' + m)
        if n_eval is None:
            n_eval = len(b)
        if n_eval < 1 or len(b) != n_eval:
            raise ValueError('Evaluation spectral rows must be aligned and nonempty')
        xt[m], xe[m] = (a, b)
    for identity in np.unique(ids):
        ix = np.flatnonzero(ids == identity)
        if len(ix) > 1:
            if not np.array_equal(Y[ix], np.repeat(Y[ix[:1]], len(ix), axis=0)):
                raise ValueError('A repeated identity has inconsistent targets')
            for m in MODALITIES:
                if not np.array_equal(xt[m][ix], np.repeat(xt[m][ix[:1]], len(ix), axis=0)):
                    raise ValueError('A repeated identity has inconsistent spectra')
    return (xt, Y, ids, xe)

def grouped_folds(ids, n_splits, seed):
    ids = np.asarray(ids, dtype=str)
    unique = np.unique(ids)
    if n_splits < 2 or len(unique) < n_splits:
        raise ValueError('Too few distinct training bottles for inner folds')
    shuffled = np.random.RandomState(int(seed)).permutation(unique)
    result = []
    for held in np.array_split(shuffled, int(n_splits)):
        va = np.flatnonzero(np.isin(ids, held))
        tr = np.flatnonzero(~np.isin(ids, held))
        if len(tr) < 3 or set(ids[tr]) & set(ids[va]):
            raise ValueError('Invalid or leaking inner training partition')
        result.append((tr, va))
    return result

def _snv(a, enabled):
    if not enabled:
        return a
    if a.shape[1] < 2:
        raise ValueError('SNV requires at least two spectral variables')
    sd = a.std(axis=1, ddof=1, keepdims=True)
    if np.any(sd <= 1e-14):
        raise ValueError('Constant spectrum cannot undergo SNV')
    return (a - a.mean(axis=1, keepdims=True)) / sd

def make_design(X_train, X_eval, mask, snv=False):
    zt, ze, metadata = ([], [], {})
    for m in mask:
        a = _snv(X_train[m], bool(snv) and m in ('raman', 'nir'))
        b = _snv(X_eval[m], bool(snv) and m in ('raman', 'nir'))
        mean = a.mean(axis=0)
        scale = max(float(np.sqrt(np.mean(np.sum((a - mean) ** 2, axis=1)))), 1e-12)
        zt.append((a - mean) / scale)
        ze.append((b - mean) / scale)
        metadata[m] = dict(n_fit=len(a), n_features=a.shape[1], block_rms=scale, training_mean_sha256=_digest(mean), snv=bool(snv) and m in ('raman', 'nir'))
    return dict(train=np.concatenate(zt, axis=1), evaluation=np.concatenate(ze, axis=1), preprocessing=metadata)

def candidate_grid(cfg):
    grid = cfg.get('grid', {})
    enabled = list(grid.get('snv', [False, True]))
    components = list(grid.get('pls_components', [2, 5, 10]))
    ridge = list(grid.get('ridge_alphas', [0.01, 0.1, 1.0, 10.0]))
    alphas = list(grid.get('krr_alphas', [0.01, 0.1, 1.0]))
    gammas = list(grid.get('krr_gammas', [0.3, 1.0, 3.0]))
    if not enabled or any((type(v) is not bool for v in enabled)):
        raise ValueError('grid.snv must be a nonempty list of booleans')
    if not components or any((int(k) != k or k < 1 for k in components)):
        raise ValueError('PLS component counts must be positive integers')
    for values in (ridge, alphas, gammas):
        if not values or any((not np.isfinite(v) or v <= 0 for v in values)):
            raise ValueError('Ridge/KRR penalties and bandwidth multipliers must be finite and positive')
    result = []
    for algorithm in ALGORITHMS:
        for mask in MASKS:
            for snv in [False] if mask == ('uv',) else enabled:
                base = dict(algorithm=algorithm, mask='+'.join(mask), modalities=list(mask), snv=snv)
                if algorithm == 'pls':
                    variants = [dict(components=int(k)) for k in components]
                elif algorithm == 'ridge':
                    variants = [dict(alpha=float(v)) for v in ridge]
                else:
                    variants = [dict(alpha=float(a), gamma=float(g)) for a in alphas for g in gammas]
                for variant in variants:
                    spec = dict(base, **variant)
                    if spec not in result:
                        result.append(spec)
    return [dict(spec, candidate_id=i) for i, spec in enumerate(result)]

def _fit_predict(design, Y, spec, targets=None):
    z, ze = (design['train'], design['evaluation'])
    q = Y.shape[1]
    targets = list(range(q)) if targets is None else list(targets)
    ym = Y.mean(axis=0)
    ys = Y.std(axis=0, ddof=1)
    ys = np.where(np.isfinite(ys) & (ys > 1e-14), ys, 1.0)
    yz = (Y - ym) / ys
    result = np.full((len(ze), q), np.nan)
    errors, warning_records, details = ({}, {}, {})
    if spec['algorithm'] == 'pls':
        cap = min(int(spec['components']), len(z) - 2, z.shape[1])
        if cap < 1:
            raise ValueError('No valid PLS components for this training set')
        for j in targets:
            try:
                if float(np.ptp(Y[:, j])) <= 1e-14:
                    result[:, j] = ym[j]
                    details[j] = dict(components_requested=spec['components'], components_capped=cap, components_active=0, constant_target=True)
                    continue
                pls = PLSRegression(n_components=cap, scale=False, max_iter=1000, tol=1e-07)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    warnings.filterwarnings('error', category=RuntimeWarning)
                    pls.fit(z, yz[:, j])
                    pred = pls.predict(ze).reshape(-1)
                messages = [dict(category=w.category.__name__, message=str(w.message)) for w in caught]
                if messages:
                    warning_records[j] = messages
                if any((issubclass(w.category, ConvergenceWarning) for w in caught)):
                    raise RuntimeError('PLS did not converge within configured iterations')
                if not np.isfinite(pred).all():
                    raise RuntimeError('PLS generated nonfinite predictions')
                result[:, j] = pred * ys[j] + ym[j]
                active = int(np.sum(np.linalg.norm(pls.x_weights_, axis=0) > 1e-12))
                details[j] = dict(components_requested=spec['components'], components_capped=cap, components_active=active, n_iter=list(map(int, pls.n_iter_)))
            except (ValueError, RuntimeError, RuntimeWarning, np.linalg.LinAlgError, FloatingPointError) as exc:
                errors[j] = type(exc).__name__ + ': ' + str(exc)
        return (result, errors, warning_records, details)
    try:
        if spec['algorithm'] == 'ridge':
            if 'linear_gram' not in design:
                design['linear_gram'] = z.dot(z.T)
                design['linear_eval'] = ze.dot(z.T)
            kernel, evaluation = (design['linear_gram'], design['linear_eval'])
            extra = {}
        elif spec['algorithm'] == 'krr':
            if 'distances' not in design:
                dist = cdist(z, z, 'sqeuclidean')
                positive = dist[np.triu_indices(len(z), 1)]
                positive = positive[positive > 1e-14]
                design['bandwidth'] = float(np.median(positive)) if len(positive) else 1.0
                design['distances'] = dist
                design['eval_distances'] = cdist(ze, z, 'sqeuclidean')
            kernel = np.exp(-spec['gamma'] * design['distances'] / design['bandwidth'])
            evaluation = np.exp(-spec['gamma'] * design['eval_distances'] / design['bandwidth'])
            extra = dict(training_distance_median=design['bandwidth'])
        else:
            raise ValueError('Unknown algorithm')
        dual = cho_solve(cho_factor(kernel + spec['alpha'] * np.eye(len(z)), lower=True), yz[:, targets])
        pred = evaluation.dot(dual)
        if pred.ndim == 1:
            pred = pred[:, None]
        result[:, targets] = pred * ys[targets] + ym[targets]
        if not np.isfinite(result[:, targets]).all():
            raise RuntimeError('Linear/kernel solution generated nonfinite predictions')
        details = {j: dict(extra) for j in targets}
    except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError) as exc:
        errors = {j: type(exc).__name__ + ': ' + str(exc) for j in targets}
    return (result, errors, warning_records, details)

def _tie_key(index, scores, candidates, target):
    spec = candidates[index]
    return (scores[index, target], len(spec['modalities']), MASK_NAMES.index(spec['mask']), bool(spec['snv']), index)

def select_and_predict(X_train, Y_train, train_ids, X_eval, cfg, seed):
    X, Y, ids, E = _validated(X_train, Y_train, train_ids, X_eval)
    q = Y.shape[1]
    candidates = candidate_grid(cfg)
    folds = grouped_folds(ids, int(cfg.get('inner_folds', 3)), int(seed))
    sse = np.zeros((len(candidates), q))
    count = np.zeros((len(candidates), q), dtype=int)
    failed = np.zeros((len(candidates), q), dtype=bool)
    records = [dict(spec=c, fold_results=[]) for c in candidates]
    split_records = []
    for fold, (tr, va) in enumerate(folds):
        split_records.append(dict(fold=fold, train_ids=ids[tr].tolist(), validation_ids=ids[va].tolist(), train_row_indices=tr.tolist(), validation_row_indices=va.tolist()))
        a, b = ({m: X[m][tr] for m in MODALITIES}, {m: X[m][va] for m in MODALITIES})
        cache, cache_errors = ({}, {})
        for i, spec in enumerate(candidates):
            key = (spec['mask'], spec['snv'])
            try:
                if key in cache_errors:
                    raise ValueError(cache_errors[key])
                if key not in cache:
                    try:
                        cache[key] = make_design(a, b, spec['modalities'], spec['snv'])
                    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                        cache_errors[key] = str(exc)
                        raise
                design = cache[key]
                pred, errors, warn, details = _fit_predict(design, Y[tr], spec)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                pred = np.full((len(va), q), np.nan)
                errors, warn, details = ({j: type(exc).__name__ + ': ' + str(exc) for j in range(q)}, {}, {})
            fold_mse = []
            for j in range(q):
                if j in errors or not np.isfinite(pred[:, j]).all():
                    failed[i, j] = True
                    fold_mse.append(None)
                else:
                    value = float(np.sum((pred[:, j] - Y[va, j]) ** 2))
                    sse[i, j] += value
                    count[i, j] += len(va)
                    fold_mse.append(value / len(va))
            records[i]['fold_results'].append(dict(fold=fold, mse_by_target=fold_mse, errors_by_target={str(k): v for k, v in errors.items()}, warnings_by_target={str(k): v for k, v in warn.items()}, effective_by_target={str(k): v for k, v in details.items()}))
    valid = ~failed & (count == len(Y))
    scores = np.full_like(sse, np.inf)
    scores[valid] = sse[valid] / count[valid]
    for i, record in enumerate(records):
        record['inner_mse_by_target'] = [float(v) if np.isfinite(v) else None for v in scores[i]]
        record['n_scored_rows_by_target'] = count[i].tolist()
    choices = {}
    for algorithm in ALGORITHMS:
        for name in MASK_NAMES + ('select_single', 'select_subset'):
            eligible = [i for i, c in enumerate(candidates) if c['algorithm'] == algorithm and (name == 'select_subset' or (name == 'select_single' and len(c['modalities']) == 1) or c['mask'] == name)]
            chosen = []
            for j in range(q):
                available = [i for i in eligible if valid[i, j]]
                if not available:
                    raise RuntimeError('All candidates failed for algorithm=%s strategy=%s target_index=%d' % (algorithm, name, j))
                chosen.append(min(available, key=lambda i: _tie_key(i, scores, candidates, j)))
            choices[algorithm + '__' + name] = chosen
    needed = {}
    for chosen in choices.values():
        for j, i in enumerate(chosen):
            needed.setdefault(i, set()).add(j)
    final_predictions, final_records, design_cache = ({}, {}, {})
    for i in sorted(needed):
        spec = candidates[i]
        key = (spec['mask'], spec['snv'])
        if key not in design_cache:
            design_cache[key] = make_design(X, E, spec['modalities'], spec['snv'])
        design = design_cache[key]
        pred, errors, warn, details = _fit_predict(design, Y, spec, sorted(needed[i]))
        if errors:
            raise RuntimeError('Final receiving-training fit failed for candidate %d: %s' % (i, errors))
        final_predictions[i] = pred
        final_records[str(i)] = dict(spec=spec, fit_ids=ids.tolist(), preprocessing=design['preprocessing'], effective_by_target={str(k): v for k, v in details.items()}, warnings_by_target={str(k): v for k, v in warn.items()})
    output, selection_records = ({}, {})
    for strategy, chosen in choices.items():
        output[strategy] = np.column_stack([final_predictions[i][:, j] for j, i in enumerate(chosen)])
        if not np.isfinite(output[strategy]).all():
            raise RuntimeError('Nonfinite final strategy prediction: ' + strategy)
        selection_records[strategy] = [dict(target_index=j, candidate_id=i, spec=candidates[i], inner_mse=float(scores[i, j])) for j, i in enumerate(chosen)]
    output['mean'] = np.repeat(Y.mean(axis=0, keepdims=True), len(next(iter(E.values()))), axis=0)
    assert set(output) == set(STRATEGIES)
    for algorithm in ALGORITHMS:
        for selector in ('select_single', 'select_subset'):
            key = algorithm + '__' + selector
            for j, i in enumerate(choices[key]):
                fixed = algorithm + '__' + candidates[i]['mask']
                if not np.array_equal(output[key][:, j], output[fixed][:, j]):
                    raise AssertionError('Selector did not reuse chosen fixed-mask fit')
    training_hash = hashlib.sha256()
    training_hash.update(json.dumps(ids.tolist(), ensure_ascii=False).encode('utf-8'))
    training_hash.update(_digest(Y).encode('ascii'))
    for m in MODALITIES:
        training_hash.update(_digest(X[m]).encode('ascii'))
    diagnostics = dict(training_ids=ids.tolist(), n_training_rows=len(Y), n_distinct_training_bottles=len(np.unique(ids)), n_targets=q, training_data_sha256=training_hash.hexdigest(), seed=int(seed), inner_splits=split_records, candidate_scores=records, selections=selection_records, final_fits=final_records, selection_scope='Receiving-training data and inner validation only; evaluation labels never supplied', tie_rule='inner pooled MSE, fewer modalities, fixed mask order, no SNV, candidate order', target_weighting='Independent tuning for each target; no outcome-based target prioritization', algorithms=list(ALGORITHMS), fixed_masks=list(MASK_NAMES), errors_policy='A candidate with any failed fold for a target is excluded for that target; no outer-label fallback')
    return (output, diagnostics)
