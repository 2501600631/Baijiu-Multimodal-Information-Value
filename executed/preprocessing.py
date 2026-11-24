import hashlib
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter

METHODS = ('raw', 'snv', 'baseline_snv', 'sg1')


def digest(value):
    value = np.ascontiguousarray(value, dtype=np.float64)
    return hashlib.sha256(str(value.shape).encode('ascii') + value.tobytes()).hexdigest()


def _axis(axis, length):
    values = np.arange(length, dtype=float) if axis is None else np.asarray(axis, dtype=float)
    if values.shape != (length,) or not np.isfinite(values).all() or (length > 1 and not np.all(np.diff(values) > 0)):
        raise ValueError('Spectral coordinates must be finite, aligned and strictly increasing')
    return values


def _uniform(a, axis):
    if a.shape[1] < 2:
        raise ValueError('At least two coordinates are required for physical preprocessing')
    intervals = np.diff(axis)
    is_uniform = np.allclose(intervals, intervals[0], rtol=1e-7, atol=1e-12)
    grid = axis if is_uniform else np.linspace(axis[0], axis[-1], len(axis))
    result = a.copy() if is_uniform else np.vstack([np.interp(grid, axis, row) for row in a])
    return result, grid, not is_uniform


def _snv(a):
    if a.shape[1] < 2:
        raise ValueError('SNV requires at least two spectral coordinates')
    sd = a.std(axis=1, ddof=1, keepdims=True)
    if np.any(sd <= 1e-14):
        raise ValueError('A constant spectrum cannot undergo SNV')
    return (a - a.mean(axis=1, keepdims=True)) / sd


def _baseline(a, lam, asymmetry, iterations):
    length = a.shape[1]
    if length < 3:
        raise ValueError('Baseline correction requires at least three coordinates')
    if not np.isfinite(lam) or lam <= 0 or not 0 < asymmetry < 1 or iterations < 1:
        raise ValueError('Invalid asymmetric least-squares baseline settings')
    differences = sparse.diags([np.ones(length-2), -2*np.ones(length-2), np.ones(length-2)], [0, 1, 2], shape=(length-2, length), format='csc')
    penalty = lam * differences.T.dot(differences)
    result = np.empty_like(a)
    for i, row in enumerate(a):
        weights = np.ones(length)
        for unused in range(iterations):
            system = sparse.diags(weights, 0, shape=(length, length), format='csc') + penalty
            baseline = spsolve(system, weights * row)
            weights = np.where(row > baseline, asymmetry, 1-asymmetry)
        result[i] = row - baseline
    return result


def transform_block(values, method='raw', axis=None, settings=None):
    a = np.asarray(values, dtype=float)
    if a.ndim != 2 or a.shape[1] < 1 or not np.isfinite(a).all():
        raise ValueError('A finite two-dimensional spectral block is required')
    if method not in METHODS:
        raise ValueError('Unknown preprocessing method: ' + str(method))
    settings = {} if settings is None else settings
    coordinates = _axis(axis, a.shape[1])
    meta = dict(method=method, rowwise=True, source_axis_sha256=digest(coordinates), source_axis_min=float(coordinates[0]), source_axis_max=float(coordinates[-1]), interpolated=False)
    if method == 'raw':
        result = a.copy()
    elif method == 'snv':
        result = _snv(a)
    else:
        result, grid, interpolated = _uniform(a, coordinates)
        meta.update(interpolated=interpolated, processed_axis_sha256=digest(grid), processed_grid_step=float(grid[1]-grid[0]), coordinate_interpretation='Exported numerical coordinates; grid spacing is not instrumental resolution')
        if method == 'baseline_snv':
            lam = float(settings.get('baseline_lambda', 100000.0))
            asymmetry = float(settings.get('baseline_asymmetry', 0.01))
            iterations = int(settings.get('baseline_iterations', 10))
            result = _snv(_baseline(result, lam, asymmetry, iterations))
            meta.update(baseline_lambda=lam, baseline_asymmetry=asymmetry, baseline_iterations=iterations)
        else:
            desired = int(settings.get('sg_window', 11))
            if desired < 3 or desired % 2 != 1:
                raise ValueError('SG window must be an odd integer of at least three')
            window = min(desired, len(grid) if len(grid) % 2 else len(grid)-1)
            degree = min(int(settings.get('sg_polyorder', 2)), window-1)
            if window < 3 or degree < 1:
                raise ValueError('SG derivative needs at least three coordinates and polynomial degree at least one')
            result = savgol_filter(result, window_length=window, polyorder=degree, deriv=1, delta=float(grid[1]-grid[0]), axis=1, mode='interp')
            meta.update(sg_window=window, sg_polyorder=degree, derivative_order=1)
    if not np.isfinite(result).all():
        raise ValueError('Preprocessing generated nonfinite values')
    return result, meta


def assemble_design(training_blocks, evaluation_blocks, metadata=None):
    if set(training_blocks) != set(evaluation_blocks) or not training_blocks:
        raise ValueError('The same nonempty modality set is required for training and evaluation')
    train, evaluation, details = {}, {}, {}
    for modality in training_blocks:
        a, b = training_blocks[modality], evaluation_blocks[modality]
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
            raise ValueError('Spectral block dimensions disagree')
        mean = a.mean(axis=0)
        rms = max(float(np.sqrt(np.mean(np.sum((a-mean)**2, axis=1)))), 1e-12)
        train[modality], evaluation[modality] = (a-mean)/rms, (b-mean)/rms
        details[modality] = dict({} if metadata is None else metadata[modality])
        details[modality].update(n_fit=len(a), n_features=a.shape[1], training_mean_sha256=digest(mean), block_rms=rms)
    return dict(train=np.concatenate(list(train.values()), axis=1), evaluation=np.concatenate(list(evaluation.values()), axis=1), train_blocks=train, evaluation_blocks=evaluation, preprocessing=details)


def make_design(X_train, X_eval, mask, methods=None, axes=None, settings=None):
    methods = {} if methods is None else methods
    axes = {} if axes is None else axes
    train, evaluation, metadata = {}, {}, {}
    for modality in mask:
        method = methods.get(modality, 'raw')
        train[modality], metadata[modality] = transform_block(X_train[modality], method, axes.get(modality), settings)
        evaluation[modality], unused = transform_block(X_eval[modality], method, axes.get(modality), settings)
    return assemble_design(train, evaluation, metadata)
