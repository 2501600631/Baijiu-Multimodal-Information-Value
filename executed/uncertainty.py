import math
import numpy as np


METHODS = ('constant_marginal', 'constant_joint', 'adaptive_marginal', 'adaptive_joint')
MODALITIES = ('uv', 'raman', 'nir')


def conformal_quantile(scores, alpha):
    values = np.asarray(scores, dtype=float)
    if values.ndim not in (1, 2) or len(values) < 1:
        raise ValueError('Calibration scores must be a nonempty vector or matrix')
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Calibration scores must be finite and nonnegative')
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('Interval alpha must lie strictly between zero and one')
    rank = int(math.ceil((len(values) + 1) * (1 - alpha)))
    if rank > len(values):
        result = float('inf') if values.ndim == 1 else np.full(values.shape[1], np.inf)
    else:
        result = np.sort(values, axis=0)[rank - 1]
        if values.ndim == 1:
            result = float(result)
    return result, rank


def _matrix(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError('Invalid shape or nonfinite values in ' + name)
    return result


def _modalities(model, options, available):
    supplied = options.get('model_modalities', {})
    if model == 'mean':
        chosen = ()
    elif model in supplied:
        chosen = tuple(supplied[model])
    elif '__' in model and all(m in MODALITIES for m in model.split('__', 1)[1].split('+')):
        chosen = tuple(model.split('__', 1)[1].split('+'))
    else:
        raise ValueError('Supply uncertainty.model_modalities for the selected model: ' + model)
    if len(set(chosen)) != len(chosen) or any(m not in MODALITIES or m not in available for m in chosen):
        raise ValueError('Invalid available modalities for model: ' + model)
    if model != 'mean' and not chosen:
        raise ValueError('A spectral model must identify at least one available modality: ' + model)
    return tuple(m for m in MODALITIES if m in chosen)


def _validate_features(X_fit, X_cal, X_test, n_fit, n_cal, n_test):
    if not all(isinstance(x, dict) for x in (X_fit, X_cal, X_test)):
        raise ValueError('Spectral features must be modality dictionaries')
    if set(X_fit) != set(X_cal) or set(X_fit) != set(X_test):
        raise ValueError('Fitting, calibration and test modalities differ')
    if any(m not in MODALITIES for m in X_fit):
        raise ValueError('Unknown spectral modality')
    result = [{}, {}, {}]
    for modality in X_fit:
        source = np.asarray(X_fit[modality], dtype=float)
        if source.ndim != 2 or source.shape[1] < 1:
            raise ValueError('Invalid spectral feature matrix: ' + modality)
        for destination, value, n in zip(result, (X_fit[modality], X_cal[modality], X_test[modality]), (n_fit, n_cal, n_test)):
            destination[modality] = _matrix(value, (n, source.shape[1]), modality)
    return tuple(result)


def _feature_space(X_fit, X_cal, X_test, chosen, requested):
    n_fit = len(next(iter(X_fit.values())))
    n_cal = len(next(iter(X_cal.values())))
    n_test = len(next(iter(X_test.values())))
    if not chosen or requested == 0:
        return np.empty((n_fit, 0)), np.empty((n_cal, 0)), np.empty((n_test, 0)), {'components': 0, 'modalities': list(chosen), 'block_scales': {}}
    fit_blocks, cal_blocks, test_blocks, block_scales = [], [], [], {}
    for modality in chosen:
        fit = X_fit[modality]
        mean = fit.mean(axis=0)
        centered = fit - mean
        scale = float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1))))
        if not np.isfinite(scale):
            raise ValueError('Nonfinite training spectral scale')
        scale = max(scale, 1e-12)
        fit_blocks.append(centered / scale)
        cal_blocks.append((X_cal[modality] - mean) / scale)
        test_blocks.append((X_test[modality] - mean) / scale)
        block_scales[modality] = scale
    fit = np.concatenate(fit_blocks, axis=1)
    cal = np.concatenate(cal_blocks, axis=1)
    test = np.concatenate(test_blocks, axis=1)
    gram = fit.dot(fit.T)
    values, vectors = np.linalg.eigh((gram + gram.T) * 0.5)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    threshold = max(float(values[0]) * 1e-12, 1e-14)
    count = min(int(requested), n_fit - 1, fit.shape[1], int(np.sum(values > threshold)))
    if count == 0:
        return np.empty((n_fit, 0)), np.empty((n_cal, 0)), np.empty((n_test, 0)), {'components': 0, 'modalities': list(chosen), 'block_scales': block_scales}
    eigenvectors = vectors[:, :count]
    singular = np.sqrt(values[:count])
    projection = eigenvectors / singular
    fit_scores = gram.dot(projection)
    cal_scores = cal.dot(fit.T).dot(projection)
    test_scores = test.dot(fit.T).dot(projection)
    center = fit_scores.mean(axis=0)
    scale = np.maximum(fit_scores.std(axis=0, ddof=1), 1e-12)
    features = tuple((a - center) / scale for a in (fit_scores, cal_scores, test_scores))
    if not all(np.isfinite(a).all() for a in features):
        raise ValueError('Nonfinite uncertainty features')
    metadata = {'components': count, 'modalities': list(chosen), 'block_scales': block_scales, 'training_eigenvalues': values[:count].tolist(), 'pca_fit_scope': 'Proper training spectra only; calibration and test spectra are transformed without refitting'}
    return features + (metadata,)


def _local_scale(Y_fit, oof, train_sd, features, options):
    fit_features, cal_features, test_features = features
    fraction = float(options.get('floor_fraction_sd', 0.02))
    multiplier = float(options.get('clip_multiplier', 10.0))
    alpha = float(options.get('ridge_alpha', 10.0))
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError('Scale floor fraction must lie in (0, 1]')
    if not np.isfinite(multiplier) or multiplier < 1:
        raise ValueError('Scale clip multiplier must be finite and at least one')
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('Residual-scale ridge penalty must be finite and positive')
    residual = np.abs(Y_fit - oof)
    floor = np.maximum(fraction * train_sd, np.finfo(float).eps * np.maximum(np.abs(Y_fit.mean(axis=0)), 1.0))
    ceiling = np.maximum(multiplier * train_sd, multiplier * np.median(residual, axis=0))
    ceiling = np.maximum(ceiling, floor)
    if not np.isfinite(ceiling).all():
        raise ValueError('Nonfinite residual-scale bounds')
    response = np.log(np.maximum(residual, floor))
    mean = response.mean(axis=0)
    if fit_features.shape[1]:
        penalty = alpha * np.eye(fit_features.shape[1])
        coefficient = np.linalg.solve(fit_features.T.dot(fit_features) + penalty, fit_features.T.dot(response - mean))
    else:
        coefficient = np.empty((0, Y_fit.shape[1]))
    log_floor, log_ceiling = np.log(floor), np.log(ceiling)
    result = []
    for value in (cal_features, test_features):
        prediction = value.dot(coefficient) + mean
        result.append(np.exp(np.clip(prediction, log_floor, log_ceiling)))
    metadata = {'ridge_alpha': alpha, 'floor_fraction_sd': fraction, 'clip_multiplier': multiplier, 'scale_floor': floor.tolist(), 'scale_ceiling': ceiling.tolist(), 'log_residual_intercept': mean.tolist(), 'coefficients': coefficient.tolist(), 'training_residual_source': 'Proper-training inner held-out predictions of selected candidates; not external test predictions', 'scale_fit_scope': 'Proper training data only; no calibration or test labels', 'scale_output': 'Positive predictor of absolute residual magnitude; not a calibrated posterior standard deviation'}
    return result[0], result[1], metadata


def _bounds(prediction, scale, scores, alpha, joint):
    q, rank = conformal_quantile(scores.max(axis=1) if joint else scores, alpha)
    half_width = scale * q
    return {'lower': prediction - half_width, 'upper': prediction + half_width, 'scale': scale.copy(), 'q': q, 'rank': rank}


def build_intervals(Y_fit, X_fit, X_cal, X_test, oof_predictions, cal_predictions, test_predictions, Y_cal, cfg):
    Y_fit = np.asarray(Y_fit, dtype=float)
    Y_cal = np.asarray(Y_cal, dtype=float)
    if Y_fit.ndim != 2 or Y_fit.shape[0] < 3 or Y_fit.shape[1] < 1 or not np.isfinite(Y_fit).all():
        raise ValueError('Proper training concentrations must be a finite matrix with at least three rows')
    n_fit, n_targets = Y_fit.shape
    if Y_cal.ndim != 2 or Y_cal.shape[0] < 1 or Y_cal.shape[1] != n_targets or not np.isfinite(Y_cal).all():
        raise ValueError('Calibration concentrations must be a finite aligned nonempty matrix')
    if not isinstance(cfg, dict):
        raise ValueError('Configuration must be a dictionary')
    options = cfg.get('uncertainty', {})
    if not isinstance(options, dict):
        raise ValueError('Uncertainty configuration must be a dictionary')
    requested = options.get('pca_components', 8)
    if isinstance(requested, bool) or int(requested) != requested or requested < 0:
        raise ValueError('Uncertainty PCA components must be a nonnegative integer')
    alpha = float(cfg.get('interval_alpha', 0.1))
    quantile_level = float(options.get('predicted_high_quantile', 0.8))
    if not np.isfinite(quantile_level) or not 0 < quantile_level < 1:
        raise ValueError('Predicted-high quantile must lie strictly between zero and one')
    if not all(isinstance(value, dict) and value for value in (oof_predictions, cal_predictions, test_predictions)):
        raise ValueError('Prediction dictionaries must be nonempty')
    if set(oof_predictions) != set(cal_predictions) or set(oof_predictions) != set(test_predictions):
        raise ValueError('Fitting, calibration and test model sets differ')
    first = np.asarray(next(iter(test_predictions.values())))
    if first.ndim != 2 or first.shape[0] < 1 or first.shape[1] != n_targets:
        raise ValueError('Test predictions must be a nonempty aligned matrix')
    n_test = len(first)
    n_cal = len(Y_cal)
    X_fit, X_cal, X_test = _validate_features(X_fit, X_cal, X_test, n_fit, n_cal, n_test)
    if not X_fit:
        raise ValueError('At least one spectral block is required')
    train_sd = Y_fit.std(axis=0, ddof=1)
    if not np.isfinite(train_sd).all() or np.any(train_sd <= 1e-12):
        raise ValueError('Each target requires a finite positive proper-training standard deviation')
    constant_cal_scale = np.broadcast_to(train_sd, (n_cal, n_targets)).copy()
    constant_test_scale = np.broadcast_to(train_sd, (n_test, n_targets)).copy()
    output, diagnostics, cache = {}, {}, {}
    thresholds = {}
    for model in test_predictions:
        oof = _matrix(oof_predictions[model], (n_fit, n_targets), model + ' training held-out predictions')
        cal = _matrix(cal_predictions[model], (n_cal, n_targets), model + ' calibration predictions')
        test = _matrix(test_predictions[model], (n_test, n_targets), model + ' test predictions')
        chosen = _modalities(model, options, X_fit)
        if chosen not in cache:
            cache[chosen] = _feature_space(X_fit, X_cal, X_test, chosen, int(requested))
        fit_features, cal_features, test_features, feature_metadata = cache[chosen]
        adaptive_cal_scale, adaptive_test_scale, scale_metadata = _local_scale(Y_fit, oof, train_sd, (fit_features, cal_features, test_features), options)
        constant_scores = np.abs(Y_cal - cal) / constant_cal_scale
        adaptive_scores = np.abs(Y_cal - cal) / adaptive_cal_scale
        output[model] = {
            'constant_marginal': _bounds(test, constant_test_scale, constant_scores, alpha, False),
            'constant_joint': _bounds(test, constant_test_scale, constant_scores, alpha, True),
            'adaptive_marginal': _bounds(test, adaptive_test_scale, adaptive_scores, alpha, False),
            'adaptive_joint': _bounds(test, adaptive_test_scale, adaptive_scores, alpha, True)
        }
        thresholds[model] = np.quantile(oof, quantile_level, axis=0).tolist()
        diagnostics[model] = {'feature_space': feature_metadata, 'scale_learner': scale_metadata, 'adaptive_calibration_scale': adaptive_cal_scale.tolist(), 'available_modalities': list(chosen), 'calibration_score_scope': 'Calibration labels used only for conformal order statistics; no test labels accepted'}
    return output, {'methods': list(METHODS), 'interval_alpha': alpha, 'n_fit': n_fit, 'n_cal': n_cal, 'n_test': n_test, 'n_targets': n_targets, 'fit_sd': train_sd.tolist(), 'predicted_high_quantile': quantile_level, 'predicted_high_thresholds': thresholds, 'predicted_high_scope': 'Thresholds from proper-training inner held-out predictions; test groups use predicted concentrations only', 'models': diagnostics, 'interpretation': 'Joint calibration targets marginal vector coverage under exchangeability; no universal true-concentration-subgroup guarantee is asserted'}
