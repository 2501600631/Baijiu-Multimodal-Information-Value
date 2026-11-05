from pathlib import Path
import hashlib
import gzip
import json
import math
import numpy as np
import pandas as pd
from data_io import TARGETS

STAGES = ('benchmark', 'learning', 'preprocessing', 'stability', 'reliability', 'controls')
MASKS = ('uv', 'raman', 'nir', 'uv+raman', 'uv+nir', 'raman+nir', 'uv+raman+nir')
EDGES = (('uv', 'uv+raman'), ('raman', 'uv+raman'), ('uv', 'uv+nir'), ('nir', 'uv+nir'), ('raman', 'raman+nir'), ('nir', 'raman+nir'), ('uv+raman', 'uv+raman+nir'), ('uv+nir', 'uv+raman+nir'), ('raman+nir', 'uv+raman+nir'))
CONTEXT = ['condition', 'training_size']
UNIT = CONTEXT + ['repeat', 'fold', 'bootstrap_refit']
POINT_KEY = UNIT + ['model', 'sample_id', 'target']
INTERVAL_METHODS = ('constant_marginal', 'constant_joint', 'adaptive_marginal', 'adaptive_joint')

def _save(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding='utf-8-sig')

def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _seed(seed, values):
    value = json.dumps(list(values), ensure_ascii=True, separators=(',', ':')).encode('ascii')
    return (int(seed) + int(hashlib.sha256(value).hexdigest()[:8], 16)) % (2 ** 32 - 1)

def _targets(cfg):
    names = cfg.get('target_names', list(TARGETS))
    if not isinstance(names, list) or not names or len(set(names)) != len(names):
        raise ValueError('Target names must be a nonempty ordered unique list')
    return names

def _format(value, digits=4):
    if pd.isna(value):
        return 'Undefined'
    if np.isposinf(value):
        return 'Infinity'
    if np.isneginf(value):
        return '-Infinity'
    return ('%.' + str(digits) + 'f') % value

def _table(headers, rows):
    return '| ' + ' | '.join(headers) + ' |\n| ' + ' | '.join(['---'] * len(headers)) + ' |\n' + ''.join('| ' + ' | '.join(map(str, row)) + ' |\n' for row in rows)

def _bool(series):
    values = series.astype(str).str.lower()
    if not values.isin(['true', 'false', '1', '0']).all():
        raise ValueError('Saved Boolean value is invalid')
    return values.isin(['true', '1'])

def _normalize(frame):
    result = frame.copy()
    for column, default in [('condition', 'enhanced'), ('training_size', 'full'), ('bootstrap_refit', -1)]:
        if column not in result:
            result[column] = default
    result['condition'] = result.condition.astype(str)
    result['training_size'] = result.training_size.astype(str)
    result['sample_id'] = result.sample_id.astype(str)
    return result

def validate_predictions(frame, targets):
    required = set(POINT_KEY + ['y_true', 'y_pred', 'train_sd', 'n_train', 'n_unique_train', 'n_cal'])
    if not required.issubset(frame):
        raise ValueError('Prediction columns absent: ' + ', '.join(sorted(required - set(frame))))
    if frame.empty:
        raise ValueError('Prediction table is empty')
    if set(frame.target) != set(targets):
        raise ValueError('Every declared target must be retained in each saved stage')
    if frame.duplicated(POINT_KEY).any():
        raise ValueError('Prediction identities are duplicated')
    if not np.isfinite(frame[['y_true', 'y_pred', 'train_sd']].to_numpy(float)).all():
        raise ValueError('Prediction, truth and scale must be finite')
    if (frame.train_sd <= 0).any() or (frame.n_train <= 1).any():
        raise ValueError('Invalid receiving-training budget or target scale')
    if 'outer_train_sd' in frame and (not np.isfinite(frame.outer_train_sd.to_numpy(float)).all() or (frame.outer_train_sd <= 0).any()):
        raise ValueError('Outer-training evaluation scales must be finite and positive')
    if (frame.n_unique_train > frame.n_train).any() or (frame.n_unique_train <= 1).any():
        raise ValueError('Distinct training bottles disagree with row budget')
    if not frame.groupby(['sample_id', 'target']).y_true.nunique().eq(1).all():
        raise ValueError('Reference values change across receiving-training tasks')
    for _, task in frame.groupby(UNIT, sort=False):
        if not task.groupby('model').target.nunique().eq(len(targets)).all():
            raise ValueError('A model omits one or more declared targets')
        sizes = task.groupby(['model', 'target']).size()
        if sizes.nunique() != 1:
            raise ValueError('Models and targets have unequal test-visit counts')
        reference = None
        for _, group in task.groupby(['model', 'target'], sort=False):
            ids = frozenset(group.sample_id)
            if reference is None:
                reference = ids
            elif ids != reference:
                raise ValueError('Models and targets do not share the same test bottles')
        for column in ['n_train', 'n_unique_train', 'n_cal']:
            if task[column].nunique() != 1:
                raise ValueError('Models have unmatched fitting or calibration budgets')
        if not task.groupby('target').train_sd.nunique().eq(1).all():
            raise ValueError('Target scales disagree across matched models')
        if 'outer_train_sd' in task and not task.groupby('target').outer_train_sd.nunique().eq(1).all():
            raise ValueError('Outer-training evaluation scales disagree across matched models')
    return True

def _metric(group):
    truth = group.y_true.to_numpy(float)
    prediction = group.y_pred.to_numpy(float)
    error = prediction - truth
    variance = np.sum((truth - truth.mean()) ** 2)
    return dict(n_visits=len(group), n_unique_bottles=group.sample_id.nunique(), rmse=float(np.sqrt(np.mean(error ** 2))), mae=float(np.mean(np.abs(error))), bias=float(np.mean(error)), r2=float(1 - np.sum(error ** 2) / variance) if variance > 0 else np.nan, standardized_mse=float(np.mean((error / group.train_sd.to_numpy(float)) ** 2)), n_train_min=int(group.n_train.min()), n_train_max=int(group.n_train.max()), n_unique_train_min=int(group.n_unique_train.min()), n_unique_train_max=int(group.n_unique_train.max()), n_cal_min=int(group.n_cal.min()), n_cal_max=int(group.n_cal.max()))

def _group_metrics(frame, keys):
    rows = []
    for values, group in frame.groupby(keys, sort=False, dropna=False):
        if not isinstance(values, tuple):
            values = (values,)
        row = dict(zip(keys, values))
        row.update(_metric(group))
        rows.append(row)
    return pd.DataFrame(rows)

def stage_metrics(frame):
    pooled = _group_metrics(frame, CONTEXT + ['model', 'target'])
    by_repeat = _group_metrics(frame, CONTEXT + ['model', 'repeat', 'bootstrap_refit', 'target'])
    macro = by_repeat.groupby(CONTEXT + ['model', 'repeat', 'bootstrap_refit'], sort=False).standardized_mse.mean().reset_index(name='macro_standardized_mse')
    return pooled, by_repeat, macro

def _gain(candidate, reference):
    return float(100 * (1 - candidate / reference)) if np.isfinite(reference) and reference > 0 else np.nan

def _contrasts(models):
    result = []
    for candidate in sorted(models):
        if '__' not in candidate:
            continue
        algorithm, strategy = candidate.split('__', 1)
        if strategy != 'select_subset' and not strategy.startswith('panel_t'):
            continue
        references = list(MASKS) + ['select_single']
        if strategy.startswith('panel_t'):
            references.append('select_subset')
        for name in references:
            reference = algorithm + '__' + name
            if reference in models and candidate != reference:
                result.append((algorithm, candidate, reference))
    return result

def paired_effects(frame, draws, seed):
    effects = []
    edges = []
    for values, group in frame.groupby(CONTEXT + ['target'], sort=False):
        context = dict(zip(CONTEXT + ['target'], values))
        work = group.copy()
        work['squared_error'] = (work.y_pred - work.y_true) ** 2
        bottle = work.groupby(['sample_id', 'model']).squared_error.mean().unstack('model')
        if bottle.isna().any().any():
            raise ValueError('Paired comparison has incomplete bottle-model predictions')
        means = np.sqrt(bottle.mean(axis=0))
        index = np.random.RandomState(_seed(seed, values)).randint(len(bottle), size=(draws, len(bottle))) if draws else None
        boot = {}
        if index is not None:
            for name in bottle.columns:
                boot[name] = np.sqrt(bottle[name].to_numpy()[index].mean(axis=1))
        for algorithm, candidate, reference in _contrasts(set(bottle.columns)):
            low = high = np.nan
            if index is not None:
                valid = boot[reference] > 0
                numbers = 100 * (1 - boot[candidate][valid] / boot[reference][valid])
                if len(numbers):
                    low, high = np.quantile(numbers, [0.025, 0.975])
            row = dict(context, algorithm=algorithm, candidate=candidate, reference=reference, n_unique_bottles=len(bottle), rmse_candidate=float(means[candidate]), rmse_reference=float(means[reference]), gain_pct=_gain(means[candidate], means[reference]), conditional_ci_low_pct=low, conditional_ci_high_pct=high, ci_scope='Paired bottle-cluster bootstrap of fixed prediction errors; excludes refitting and multiplicity protection')
            effects.append(row)
        algorithms = sorted({name.split('__', 1)[0] for name in bottle.columns if '__' in name})
        for algorithm in algorithms:
            for before, after in EDGES:
                a, b = algorithm + '__' + before, algorithm + '__' + after
                if a in means and b in means:
                    edges.append(dict(context, algorithm=algorithm, before=before, after=after, rmse_before=float(means[a]), rmse_after=float(means[b]), gain_pct=_gain(means[b], means[a]), interpretation='Prediction-strategy contrast, not a chemical causal effect'))
    return pd.DataFrame(effects), pd.DataFrame(edges)

def acquisition_and_parameters(out, targets):
    acquisition, selections, parameters = [], [], []
    paths = list((out / 'tasks').glob('*.json')) + list((out / 'tasks').glob('*.json.gz'))
    for path in sorted(paths):
        if path.suffix == '.gz':
            with gzip.open(str(path), 'rt', encoding='utf-8') as handle:
                stored = json.load(handle)
        else:
            stored = json.loads(path.read_text(encoding='utf-8'))
        task = stored.get('payload', stored)
        diagnostics = task.get('diagnostics', {})
        choices = diagnostics.get('selection_by_model', diagnostics.get('selections', {}))
        context = {key: task.get(key, 'enhanced' if key == 'condition' else 'full' if key == 'training_size' else -1 if key == 'bootstrap_refit' else None) for key in ['stage'] + UNIT}
        grids = {}
        for candidate in diagnostics.get('candidate_scores', []):
            spec = candidate.get('spec', candidate) if isinstance(candidate, dict) else {}
            algorithm = spec.get('algorithm')
            for name in ['components', 'alpha', 'gamma']:
                value = spec.get(name)
                if algorithm and isinstance(value, (int, float)) and not isinstance(value, bool):
                    grids.setdefault((algorithm, name), set()).add(float(value))
        for model, chosen in choices.items():
            if not isinstance(chosen, list) or not chosen:
                continue
            masks, union, count = [], set(), 0
            for item in chosen:
                j = int(item['target_index'])
                if j < 0 or j >= len(targets):
                    raise ValueError('Selection target index is outside the declared target order')
                spec = item.get('spec', {})
                mask = item.get('mask', spec.get('mask'))
                if mask not in MASKS:
                    raise ValueError('Selection mask is missing or invalid')
                modalities = set(mask.split('+'))
                masks.append(mask)
                union.update(modalities)
                count += len(modalities)
                preprocessing = spec.get('preprocessing', {m: 'snv' if bool(spec.get('snv', False)) and m in ('raman', 'nir') else 'raw' for m in modalities})
                selections.append(dict(context, model=model, target=targets[j], mask=mask, n_modalities=len(modalities), uv_method=preprocessing.get('uv', 'not_acquired'), raman_method=preprocessing.get('raman', 'not_acquired'), nir_method=preprocessing.get('nir', 'not_acquired'), preprocessing_json=json.dumps(preprocessing, sort_keys=True, ensure_ascii=True), sequential_order='+'.join(spec.get('order', []))))
                algorithm = model.split('__', 1)[0]
                for name in ['components', 'alpha', 'gamma']:
                    value = spec.get(name)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        values = grids.get((algorithm, name), {float(value)})
                        parameters.append(dict(context, model=model, target=targets[j], parameter=name, value=float(value), n_grid_values=len(values), grid_min=min(values), grid_max=max(values), at_min=math.isclose(float(value), min(values), rel_tol=1e-12, abs_tol=1e-14), at_max=math.isclose(float(value), max(values), rel_tol=1e-12, abs_tol=1e-14), boundary_scope='Observed declared candidate range, not proof that a wider range improves prediction'))
            if len(chosen) != len(targets) or {int(item['target_index']) for item in chosen} != set(range(len(targets))):
                raise ValueError('A selection record does not contain every declared target exactly once')
            if '__panel_t' in model and len(set(masks)) != 1:
                raise ValueError('A panel selector does not use one shared mask for all targets')
            acquisition.append(dict(context, model=model, acquisition_union='+'.join(m for m in ('uv', 'raman', 'nir') if m in union), n_acquisition_modalities=len(union), mean_per_target_modalities=count / len(targets), shared_mask=len(set(masks)) == 1, acquisition_interpretation='Whole-panel acquisition uses the union of all target-specific inputs'))
    return pd.DataFrame(acquisition), pd.DataFrame(selections), pd.DataFrame(parameters)

def interval_metrics(intervals, points, targets, alpha):
    required = set(POINT_KEY + ['interval_method', 'lower', 'upper', 'width', 'covered', 'fit_sd', 'high_true_concentration', 'high_predicted_concentration', 'interval_score'])
    if not required.issubset(intervals):
        raise ValueError('Interval columns absent: ' + ', '.join(sorted(required - set(intervals))))
    if intervals.duplicated(POINT_KEY + ['interval_method']).any():
        raise ValueError('Interval identities are duplicated')
    if set(intervals.interval_method) != set(INTERVAL_METHODS):
        raise ValueError('The four declared interval methods are incomplete')
    if len(intervals) != len(points) * len(INTERVAL_METHODS):
        raise ValueError('Interval rows do not match all point predictions')
    truth = points[POINT_KEY + ['y_true']].rename(columns={'y_true': '_point_truth'})
    d = intervals.merge(truth, on=POINT_KEY, how='left', validate='many_to_one')
    if d._point_truth.isna().any():
        raise ValueError('An interval has no matching point prediction')
    if 'y_true' in d and not np.allclose(d.y_true, d._point_truth, rtol=0, atol=0):
        raise ValueError('Interval and point truth disagree')
    d['y_true'] = d._point_truth
    if d[['lower', 'upper', 'width', 'fit_sd', 'interval_score']].isna().any().any() or (d.upper < d.lower).any() or (d.fit_sd <= 0).any():
        raise ValueError('Interval endpoints, widths or scales are invalid')
    if not np.allclose(d.upper - d.lower, d.width, rtol=1e-10, atol=1e-10):
        raise ValueError('Saved interval width differs from endpoints')
    covered = (d.y_true >= d.lower) & (d.y_true <= d.upper)
    if not np.array_equal(covered.to_numpy(), _bool(d.covered).to_numpy()):
        raise ValueError('Saved coverage differs from endpoints')
    expected_score = d.width.to_numpy(float).copy()
    below = (d.y_true < d.lower).to_numpy()
    above = (d.y_true > d.upper).to_numpy()
    expected_score[below] += 2 / alpha * (d.lower.to_numpy()[below] - d.y_true.to_numpy()[below])
    expected_score[above] += 2 / alpha * (d.y_true.to_numpy()[above] - d.upper.to_numpy()[above])
    if not np.allclose(expected_score, d.interval_score, rtol=1e-9, atol=1e-9):
        raise ValueError('Saved interval score differs from recomputed score')
    d['covered'] = covered
    for flag in ['high_true_concentration', 'high_predicted_concentration']:
        d[flag] = _bool(d[flag])
        if not d.groupby(POINT_KEY)[flag].nunique().eq(1).all():
            raise ValueError('Concentration stratum changes across interval constructions')
    if not d.groupby(POINT_KEY).interval_method.nunique().eq(len(INTERVAL_METHODS)).all():
        raise ValueError('A prediction lacks one interval construction')
    rows = []
    keys = CONTEXT + ['model', 'interval_method', 'target']
    for values, group in d.groupby(keys, sort=False):
        context = dict(zip(keys, values))
        for stratum in ['all', 'high_true_concentration', 'high_predicted_concentration']:
            g = group if stratum == 'all' else group[group[stratum]]
            rows.append(dict(context, stratum=stratum, n_visits=len(g), n_unique_bottles=g.sample_id.nunique(), coverage=float(g.covered.mean()) if len(g) else np.nan, mean_width=float(g.width.mean()) if len(g) else np.nan, mean_standardized_width=float(np.mean(g.width / g.fit_sd)) if len(g) else np.nan, mean_interval_score=float(g.interval_score.mean()) if len(g) else np.nan, mean_standardized_interval_score=float(np.mean(g.interval_score / g.fit_sd)) if len(g) else np.nan, infinite_width_fraction=float(np.isinf(g.width).mean()) if len(g) else np.nan))
    joint = []
    for values, group in d.groupby(CONTEXT + ['model', 'interval_method'], sort=False):
        context = dict(zip(CONTEXT + ['model', 'interval_method'], values))
        visit = group.groupby(['repeat', 'fold', 'bootstrap_refit', 'sample_id'], sort=False)
        if not visit.target.nunique().eq(len(targets)).all() or not visit.size().eq(len(targets)).all():
            raise ValueError('The joint interval does not contain a complete target vector')
        joint.append(dict(context, n_vector_visits=visit.ngroups, n_unique_bottles=group.sample_id.nunique(), joint_coverage=float(visit.covered.all().mean()), mean_standardized_width=float(np.mean(group.width / group.fit_sd)), mean_standardized_interval_score=float(np.mean(group.interval_score / group.fit_sd)), n_fit_min=int(group.n_fit.min()) if 'n_fit' in group else np.nan, n_fit_max=int(group.n_fit.max()) if 'n_fit' in group else np.nan, n_cal_min=int(group.n_cal.min()) if 'n_cal' in group else np.nan, n_cal_max=int(group.n_cal.max()) if 'n_cal' in group else np.nan))
    return pd.DataFrame(rows), pd.DataFrame(joint)

def workflow_effects(frames):
    candidates = []
    if 'preprocessing' in frames:
        candidates.append(frames['preprocessing'])
    if 'benchmark' in frames:
        copy = frames['benchmark'].copy()
        copy['condition'] = 'physical'
        candidates.append(copy)
    if not candidates:
        return pd.DataFrame()
    all_rows = pd.concat(candidates, ignore_index=True)
    conditions = ['legacy', 'independent_snv', 'expanded_grid', 'physical']
    rows = []
    for (size, model, target), group in all_rows.groupby(['training_size', 'model', 'target'], sort=False):
        for before, after in zip(conditions[:-1], conditions[1:]):
            a, b = group[group.condition.eq(before)], group[group.condition.eq(after)]
            if a.empty or b.empty:
                continue
            paired = a.merge(b, on=['repeat', 'fold', 'sample_id', 'bootstrap_refit'], suffixes=('_before', '_after'), validate='one_to_one')
            if len(paired) != len(a) or len(paired) != len(b):
                continue
            matched = all(np.array_equal(paired[column+'_before'], paired[column+'_after']) for column in ['n_train', 'n_unique_train', 'n_cal'])
            if not np.allclose(paired.y_true_before, paired.y_true_after, rtol=0, atol=0) or not matched or not np.allclose(paired.train_sd_before, paired.train_sd_after, rtol=1e-12, atol=1e-12):
                raise ValueError('Preprocessing comparisons have different truth or training budgets')
            ra = np.sqrt(np.mean((paired.y_pred_before - paired.y_true_before) ** 2))
            rb = np.sqrt(np.mean((paired.y_pred_after - paired.y_true_after) ** 2))
            rows.append(dict(training_size=size, model=model, target=target, before=before, after=after, n_visits=len(paired), n_unique_bottles=paired.sample_id.nunique(), rmse_before=float(ra), rmse_after=float(rb), gain_pct=_gain(rb, ra), interpretation='Matched retrospective workflow ablation, not an independently validated winner'))
    return pd.DataFrame(rows)

def control_effects(frames):
    if 'controls' not in frames or 'benchmark' not in frames:
        return pd.DataFrame()
    rows = []
    baseline = frames['benchmark']
    for values, group in frames['controls'].groupby(['condition', 'bootstrap_refit', 'model', 'target'], sort=False):
        condition, draw, model, target = values
        real = baseline[baseline.model.eq(model) & baseline.target.eq(target)]
        if real.empty:
            continue
        pair = real.merge(group, on=['repeat', 'fold', 'sample_id'], suffixes=('_real', '_control'), validate='one_to_one')
        if len(pair) != len(real) or len(pair) != len(group):
            continue
        matched = all(np.array_equal(pair[column+'_real'], pair[column+'_control']) for column in ['n_train', 'n_unique_train', 'n_cal'])
        if not np.array_equal(pair.y_true_real, pair.y_true_control) or not matched or not np.allclose(pair.train_sd_real, pair.train_sd_control, rtol=1e-12, atol=1e-12):
            raise ValueError('NIR controls have unmatched truth or fitting budgets')
        truth = pair.y_true_real.to_numpy()
        real_rmse = np.sqrt(np.mean((pair.y_pred_real - truth) ** 2))
        null_rmse = np.sqrt(np.mean((pair.y_pred_control - truth) ** 2))
        rows.append(dict(condition=condition, permutation=int(draw), model=model, target=target, n_visits=len(pair), n_unique_bottles=pair.sample_id.nunique(), rmse_real_nir=float(real_rmse), rmse_control_nir=float(null_rmse), gain_real_over_control_pct=_gain(real_rmse, null_rmse), interpretation='Shuffled-block diagnostic only; no conditional-independence or chemical-specificity test'))
    return pd.DataFrame(rows)

def adaptive_interval_effects(per_target, joint):
    rows = []
    for scope, frame, keys, coverage_name in [('individual_target', per_target, CONTEXT+['model', 'target', 'stratum'], 'coverage'), ('target_vector', joint, CONTEXT+['model'], 'joint_coverage')]:
        for suffix in ['marginal', 'joint']:
            reference = frame[frame.interval_method.eq('constant_'+suffix)]
            candidate = frame[frame.interval_method.eq('adaptive_'+suffix)]
            pair = reference.merge(candidate, on=keys, suffixes=('_constant', '_adaptive'), validate='one_to_one')
            for _, record in pair.iterrows():
                row = {key: record[key] for key in keys}
                row.update(scope=scope, calibration=suffix, coverage_constant=record[coverage_name+'_constant'], coverage_adaptive=record[coverage_name+'_adaptive'], coverage_change_percentage_points=100*(record[coverage_name+'_adaptive']-record[coverage_name+'_constant']), standardized_width_constant=record['mean_standardized_width_constant'], standardized_width_adaptive=record['mean_standardized_width_adaptive'], width_reduction_pct=_gain(record['mean_standardized_width_adaptive'], record['mean_standardized_width_constant']), standardized_interval_score_constant=record['mean_standardized_interval_score_constant'], standardized_interval_score_adaptive=record['mean_standardized_interval_score_adaptive'], interval_score_reduction_pct=_gain(record['mean_standardized_interval_score_adaptive'], record['mean_standardized_interval_score_constant']), interpretation='Matched empirical coverage and efficiency comparison; smaller width alone does not establish superiority or subgroup validity')
                rows.append(row)
    return pd.DataFrame(rows)

def summarize(out_dir, cfg):
    out = Path(out_dir)
    derived = out / 'derived'
    derived.mkdir(parents=True, exist_ok=True)
    targets = _targets(cfg)
    draws = int(cfg.get('summary_bootstrap_draws', 2000))
    if draws < 0:
        raise ValueError('Bootstrap draw count must be nonnegative')
    complete_path = out / 'COMPLETENESS.json'
    complete = json.loads(complete_path.read_text(encoding='utf-8')) if complete_path.exists() else {}
    result = dict(available_stages=[], missing_stages=[], files={}, figures=[], execution_status=complete.get('status', 'not_provided'), models_refitted_by_summary=False)
    report = ['# Information value and reliability of multispectral prediction\n', 'This is a retrospective internal analysis of the same bottle cohort. All declared targets receive equal reporting weight. Repeated folds, bootstrap refits and multiple strategies do not create additional independent bottles.\n', 'Overall standardized MSE scales each squared test error by its receiving-training target variance, averages within each target, and weights targets equally. Original-unit RMSE, MAE and bias are retained for every target.\n', 'Conditional bootstrap intervals resample bottles jointly across paired saved predictions. They exclude model-refitting variation and provide no multiplicity protection. Positive gains are not automatically analytical or clinical equivalence.\n', 'Execution status: `' + result['execution_status'] + '`. Only the completeness record determines whether planned tasks finished. Incomplete stages remain descriptive partial results.\n']
    frames = {}
    for stage in STAGES:
        path = out / stage / 'predictions.csv'
        if not path.exists() or path.stat().st_size == 0:
            result['missing_stages'].append(stage)
            continue
        frame = _normalize(pd.read_csv(path, dtype={'sample_id': str, 'training_size': str}))
        if frame.empty:
            result['missing_stages'].append(stage)
            continue
        validate_predictions(frame, targets)
        frames[stage] = frame
        result['available_stages'].append(stage)
        result['files'][path.relative_to(out).as_posix()] = _hash(path)
        folder = derived / stage
        pooled, repeated, macro = stage_metrics(frame)
        _save(folder / 'target_metrics_pooled.csv', pooled)
        _save(folder / 'target_metrics_by_repeat.csv', repeated)
        _save(folder / 'macro_standardized_mse.csv', macro)
        effects, edges = paired_effects(frame, 0 if stage in ('stability', 'controls') else draws, cfg.get('seed', 20261102))
        if not effects.empty:
            _save(folder / 'subset_and_panel_vs_controls.csv', effects)
        if not edges.empty:
            _save(folder / 'nine_modality_addition_edges.csv', edges)
        if stage == 'stability':
            refits = []
            for b, group in frame.groupby('bootstrap_refit', sort=False):
                effect, _ = paired_effects(group, 0, cfg.get('seed', 20261102))
                if not effect.empty:
                    effect['bootstrap_refit'] = b
                    refits.append(effect)
            if refits:
                refit = pd.concat(refits, ignore_index=True)
                _save(folder / 'effects_by_refit.csv', refit)
                ranges = refit.groupby(CONTEXT + ['target', 'algorithm', 'candidate', 'reference'], sort=False).gain_pct.agg(n_refits='count', mean_gain_pct='mean', min_gain_pct='min', max_gain_pct='max').reset_index()
                ranges['range_scope'] = 'Observed refit-index sensitivity range; not a confidence interval or future success probability'
                _save(folder / 'refit_observed_ranges.csv', ranges)
        state = complete.get('stages', {}).get(stage, {})
        report.extend(['## ' + stage + '\n', 'Completeness: ' + ('Complete' if state.get('complete') is True else 'Not confirmed; this may be a partial stage') + '. Independent test bottles represented: ' + str(frame.sample_id.nunique()) + '.\n'])
        average = macro.groupby(CONTEXT + ['model'], sort=False).macro_standardized_mse.mean().reset_index()
        report.append(_table(['Condition', 'Training-size label', 'Strategy', 'Equal-target standardized MSE'], [[row.condition, row.training_size, row.model, _format(row.macro_standardized_mse, 6)] for row in average.itertuples()]))
        report.append('\nEvery target is retained in `' + stage + '/target_metrics_pooled.csv`; paired effects, where applicable, are in the same directory. Fitting budgets must be matched when interpreting algorithm gains.\n')
        if stage == 'benchmark':
            for target in targets:
                name = TARGETS.get(target, target)
                report.append('### ' + name + '\n')
                group = pooled[pooled.target.eq(target)]
                report.append(_table(['Condition', 'Strategy', 'RMSE (g/L)', 'MAE (g/L)', 'Bias (g/L)', 'R2'], [[r.condition, r.model, _format(r.rmse, 6), _format(r.mae, 6), _format(r.bias, 6), _format(r.r2, 4)] for r in group.itertuples()]))
    acquisition, selections, parameters = acquisition_and_parameters(out, targets)
    if not acquisition.empty:
        _save(derived / 'acquisition_union_by_task.csv', acquisition)
        frequency = acquisition.groupby(['stage'] + CONTEXT + ['model', 'acquisition_union', 'n_acquisition_modalities'], sort=False).size().reset_index(name='n_tasks')
        frequency['fraction'] = frequency.n_tasks / frequency.groupby(['stage'] + CONTEXT + ['model']).n_tasks.transform('sum')
        _save(derived / 'acquisition_union_frequency.csv', frequency)
        report.append('## Whole-panel measurement requirement\n\nThe number of measurements required for all targets is the union of their selected modalities. The average per-target subset size is not a whole-panel instrument saving. Shared-panel tolerance values are sensitivity settings, not established analytical noninferiority margins. Exact task identities and union counts are in `acquisition_union_by_task.csv`.\n')
    if not selections.empty:
        _save(derived / 'selected_preprocessing_records.csv', selections)
        frequency = selections.groupby(['stage'] + CONTEXT + ['model', 'target', 'mask'], sort=False).size().reset_index(name='n_selections')
        frequency['fraction'] = frequency.n_selections / frequency.groupby(['stage'] + CONTEXT + ['model', 'target']).n_selections.transform('sum')
        _save(derived / 'selection_frequency.csv', frequency)
        treatment = selections.melt(id_vars=['stage']+CONTEXT+['model','target'],value_vars=['uv_method','raman_method','nir_method'],var_name='modality',value_name='method')
        treatment['modality'] = treatment.modality.str.replace('_method','',regex=False)
        treatment = treatment.groupby(['stage']+CONTEXT+['model','target','modality','method'],sort=False).size().reset_index(name='n_selections')
        treatment['fraction'] = treatment.n_selections/treatment.groupby(['stage']+CONTEXT+['model','target','modality']).n_selections.transform('sum')
        _save(derived / 'preprocessing_selection_frequency.csv',treatment)
        ordered = selections[selections.sequential_order.ne('')]
        if not ordered.empty:
            ordered = ordered.groupby(['stage']+CONTEXT+['model','target','mask','sequential_order'],sort=False).size().reset_index(name='n_selections')
            ordered['fraction'] = ordered.n_selections/ordered.groupby(['stage']+CONTEXT+['model','target']).n_selections.transform('sum')
            _save(derived / 'sopls_order_selection_frequency.csv',ordered)
    if not parameters.empty:
        _save(derived / 'selected_parameter_records.csv', parameters)
        boundaries = parameters.groupby(['stage'] + CONTEXT + ['model', 'parameter', 'n_grid_values', 'grid_min', 'grid_max'], sort=False).agg(n_selections=('value', 'size'), at_min_fraction=('at_min', 'mean'), at_max_fraction=('at_max', 'mean')).reset_index()
        _save(derived / 'parameter_boundary_frequency.csv', boundaries)
    learning = []
    if 'learning' in frames:
        learning.append(frames['learning'])
    if 'benchmark' in frames:
        full = frames['benchmark'].copy()
        full['training_size'] = 'full'
        learning.append(full)
    if learning:
        combined = pd.concat(learning, ignore_index=True).drop_duplicates(POINT_KEY, keep='first')
        pooled, repeated, macro = stage_metrics(combined)
        _save(derived / 'learning' / 'learning_target_metrics.csv', pooled)
        _save(derived / 'learning' / 'learning_macro_by_repeat.csv', macro)
        scale_record = dict(common_scale_available='outer_train_sd' in combined, ordinary_macro_scale='Each receiving-training subset target SD', common_scale='Full outer-training target SD held fixed within each outer fold across training sizes; evaluation weights only, never model fitting, preprocessing, selection or interval construction')
        if 'outer_train_sd' in combined:
            if not np.isfinite(combined.outer_train_sd.to_numpy(float)).all() or (combined.outer_train_sd <= 0).any():
                raise ValueError('A learning-curve row lacks its common outer-training evaluation scale')
            consistency = combined.groupby(['repeat', 'fold', 'target']).outer_train_sd.nunique()
            if not consistency.eq(1).all():
                raise ValueError('Learning sizes do not share a fixed outer-training target scale')
            fixed_scale = combined.copy()
            fixed_scale['train_sd'] = fixed_scale.outer_train_sd
            common_target, _, common_macro = stage_metrics(fixed_scale)
            common_target['normalization_scope'] = 'Full outer-training SD; evaluation only'
            common_macro['normalization_scope'] = 'Full outer-training SD; evaluation only'
            _save(derived / 'learning' / 'learning_target_metrics_common_scale.csv', common_target)
            _save(derived / 'learning' / 'learning_macro_common_scale_by_repeat.csv', common_macro)
        (derived / 'learning' / 'scale_definition.json').write_text(json.dumps(scale_record,indent=2),encoding='utf-8')
        learning_budget = combined.groupby(CONTEXT + ['model', 'repeat', 'fold', 'bootstrap_refit'], sort=False).agg(n_train=('n_train', 'first'), n_unique_train=('n_unique_train', 'first'), n_test=('sample_id', 'nunique')).reset_index()
        _save(derived / 'learning' / 'training_budgets.csv', learning_budget)
        report.append('## Controlled learning curves\n\nBenchmark full-training predictions are reused as the full-size anchor when present. Training subsets are nested and test identities are fixed by the runner. Original-unit target RMSE is unchanged by normalization. The common-scale learning tables use each full outer-training target SD as a fixed evaluation weight across training sizes; this scale is not supplied to model fitting, preprocessing, selection or interval construction. It uses no outer-test labels. Ordinary subset-specific fitting-SD macro tables are retained separately, because varying their denominator can confound a learning-curve comparison. These curves describe this cohort and do not establish performance in a new batch or distillery. Common-scale availability: '+str(scale_record['common_scale_available'])+'.\n')
    workflow = workflow_effects(frames)
    if not workflow.empty:
        _save(derived / 'preprocessing' / 'matched_workflow_effects.csv', workflow)
    controls = control_effects(frames)
    if not controls.empty:
        _save(derived / 'controls' / 'real_vs_shuffled_nir.csv', controls)
    interval_path = out / 'reliability' / 'intervals.csv'
    if interval_path.exists() and 'reliability' in frames:
        interval = _normalize(pd.read_csv(interval_path, dtype={'sample_id': str, 'training_size': str}))
        target_interval, joint = interval_metrics(interval, frames['reliability'], targets, float(cfg.get('interval_alpha', 0.1)))
        _save(derived / 'reliability' / 'coverage_width_score_by_target.csv', target_interval)
        _save(derived / 'reliability' / 'joint_coverage_width_score.csv', joint)
        _save(derived / 'reliability' / 'adaptive_vs_constant_interval_effects.csv', adaptive_interval_effects(target_interval, joint))
        result['files'][interval_path.relative_to(out).as_posix()] = _hash(interval_path)
        report.append('## Reliability and interval efficiency\n\nAll four interval constructions use matched point predictions and fitting/calibration/test identities. Compare coverage jointly with standardized width and interval score. The training-only local scale learner is an experimental adaptation, not a promise of concentration-conditional validity.\n')
        report.append('The true-high stratum uses reference concentrations and is a retrospective diagnostic. The predicted-high stratum uses a model-specific threshold set from proper-training OOF predictions and is available at prediction time; its membership can differ across point models, so interval constructions should be compared within the same point model. Empty strata are reported as undefined, not zero coverage. Grouping on true future concentration can produce shifted residuals even for a well-specified conditional-mean predictor; subgroup undercoverage alone does not establish heteroscedasticity or misspecification.\n')
        report.append(_table(['Strategy', 'Interval construction', 'Joint coverage (%)', 'Mean standardized width', 'Mean standardized interval score'], [[r.model, r.interval_method, _format(100 * r.joint_coverage, 2), _format(r.mean_standardized_width, 4), _format(r.mean_standardized_interval_score, 4)] for r in joint.itertuples()]))
    if not frames:
        report.append('No new experiment predictions exist. This document is a preparation record, not experimental evidence. Archived V4 results remain separate and are not relabelled as V6 findings.\n')
    (derived / 'report.md').write_text('\n'.join(report), encoding='utf-8')
    if cfg.get('summary_make_figures', True):
        from plots import make_figures
        result['figures'] = make_figures(out, cfg)
    result['target_order'] = targets
    result['independent_bottles_represented'] = sorted({int(frame.sample_id.nunique()) for frame in frames.values()})
    (derived / 'summary_provenance.json').write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding='utf-8')
    return result

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--config', default='configs/full.json')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    print(json.dumps(summarize(args.out, config), indent=2))
