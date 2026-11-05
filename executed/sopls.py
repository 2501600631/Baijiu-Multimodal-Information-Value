import warnings
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import ConvergenceWarning


def predict_sequential(training_blocks, evaluation_blocks, y, order, components):
    y = np.asarray(y, dtype=float).reshape(-1)
    if not np.isfinite(y).all() or len(y) < 3 or set(order) != set(training_blocks) or len(order) != len(training_blocks):
        raise ValueError('Invalid SO-PLS response or block order')
    mean = float(y.mean())
    scale = float(y.std(ddof=1))
    n_eval = len(next(iter(evaluation_blocks.values())))
    if not np.isfinite(scale) or scale <= 1e-14:
        return np.full(n_eval, mean), dict(constant_target=True, block_order=list(order), components_active=[]), []
    residual = (y-mean)/scale
    prediction = np.zeros(n_eval)
    scores = np.empty((len(y), 0))
    evaluation_scores = np.empty((n_eval, 0))
    records, warning_records = [], []
    for modality in order:
        x = np.asarray(training_blocks[modality], dtype=float)
        xe = np.asarray(evaluation_blocks[modality], dtype=float)
        if len(x) != len(y) or x.shape[1] != xe.shape[1]:
            raise ValueError('SO-PLS block dimensions disagree')
        if scores.shape[1]:
            projection = np.linalg.lstsq(scores, x, rcond=1e-12)[0]
            x = x-scores.dot(projection)
            xe = xe-evaluation_scores.dot(projection)
        norm = float(np.linalg.norm(x))
        if norm <= 1e-12 or float(np.std(residual)) <= 1e-12:
            records.append(dict(modality=modality, requested_components=int(components), components_active=0, skipped_degenerate_residual=True))
            continue
        gram = x.dot(x.T)
        eigenvalues = np.linalg.eigvalsh(gram)
        rank = int(np.sum(eigenvalues > max(float(eigenvalues[-1]), 1e-30)*1e-12))
        cap = min(int(components), len(y)-2, x.shape[1], rank)
        if cap < 1:
            records.append(dict(modality=modality, requested_components=int(components), components_active=0, skipped_degenerate_residual=True))
            continue
        model = PLSRegression(n_components=cap, scale=False, max_iter=1000, tol=1e-7)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            warnings.filterwarnings('error', category=RuntimeWarning)
            model.fit(x, residual)
            fit_prediction = model.predict(x).reshape(-1)
            eval_prediction = model.predict(xe).reshape(-1)
            new_scores = model.transform(x)
            new_evaluation_scores = model.transform(xe)
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            raise RuntimeError('SO-PLS block fit did not converge')
        warning_records.extend(dict(modality=modality, category=w.category.__name__, message=str(w.message)) for w in caught)
        if not np.isfinite(eval_prediction).all() or not np.isfinite(fit_prediction).all() or not np.isfinite(new_scores).all() or not np.isfinite(new_evaluation_scores).all():
            raise RuntimeError('SO-PLS generated nonfinite predictions or scores')
        active = np.linalg.norm(model.x_weights_, axis=0) > 1e-12
        scores = np.column_stack([scores, new_scores[:, active]])
        evaluation_scores = np.column_stack([evaluation_scores, new_evaluation_scores[:, active]])
        residual = residual-fit_prediction
        prediction += eval_prediction
        records.append(dict(modality=modality, requested_components=int(components), components_capped=cap, components_active=int(active.sum()), n_iter=list(map(int, model.n_iter_)), rank_threshold_relative=1e-12))
    return prediction*scale+mean, dict(block_order=list(order), blocks=records, total_components_active=int(scores.shape[1]), response_mean=mean, response_sd=scale), warning_records
