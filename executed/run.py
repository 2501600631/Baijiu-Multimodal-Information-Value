import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
from pathlib import Path
import argparse
import copy
import gzip
import hashlib
import json
import math
import platform
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from data_io import load_dataset, TARGETS, sha256
import engine
import legacy_engine
from uncertainty import build_intervals

ROOT = Path(__file__).resolve().parent
STAGES = ('benchmark', 'learning', 'preprocessing', 'stability', 'reliability', 'controls')
OLD_GRID = dict(snv=[False, True], pls_components=[2, 5, 10], ridge_alphas=[0.01, 0.1, 1, 10], krr_alphas=[0.01, 0.1, 1], krr_gammas=[0.3, 1, 3])


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else 'Infinity' if value > 0 else '-Infinity' if value < 0 else None
    return value


def payload_hash(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('utf-8')).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    data = json.dumps(plain(value), ensure_ascii=True, allow_nan=False, separators=(',', ':')).encode('utf-8')
    if path.suffix == '.gz':
        with temporary.open('wb') as handle:
            with gzip.GzipFile(filename='', mode='wb', fileobj=handle, mtime=0) as zipped:
                zipped.write(data)
    else:
        temporary.write_bytes(data)
    temporary.replace(path)


def read_json(path):
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(str(path), 'rt', encoding='utf-8') as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding='utf-8'))


class Lock:
    def __init__(self, out):
        self.path = Path(out) / '.run.lock'

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open('a+b')
        self.handle.seek(0, 2)
        if not self.handle.tell():
            self.handle.write(b'0')
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            raise RuntimeError('This output is already running. Close the other run before continuing.')
        return self

    def __exit__(self, *args):
        self.handle.close()


def execution_signature(cfg):
    import scipy
    import sklearn
    import matplotlib
    import PIL
    files = sorted(ROOT.glob('*.py')) + sorted((ROOT / 'data').glob('*')) + [ROOT / 'requirements.txt']
    sources = {p.relative_to(ROOT).as_posix(): sha256(p) for p in files if p.is_file()}
    context = dict(config=cfg, files=sources, runtime=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__, matplotlib=matplotlib.__version__, pillow=PIL.__version__, system=platform.system(), machine=platform.machine(), threads={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}))
    return payload_hash(context), context


def make_splits(ids, cfg):
    ids = np.asarray(ids, dtype=str)
    if len(set(ids)) != len(ids):
        raise ValueError('Outer splitting requires independent unique bottle identities')
    if cfg.get('profile') == 'full':
        records = copy.deepcopy(read_json(ROOT / 'data' / 'outer_splits.json')['splits'])
    else:
        records = []
        for repeat in range(int(cfg['repeats'])):
            cv = KFold(int(cfg['outer_folds']), shuffle=True, random_state=int(cfg['seed']) + repeat)
            for fold, (train, test) in enumerate(cv.split(ids)):
                records.append(dict(repeat=repeat, fold=fold, train_ids=ids[train].tolist(), test_ids=ids[test].tolist()))
    if len(records) != int(cfg['repeats']) * int(cfg['outer_folds']):
        raise ValueError('Declared outer fold count differs from archived identities')
    lookup = {v: i for i, v in enumerate(ids)}
    for repeat in range(int(cfg['repeats'])):
        held = []
        group = [s for s in records if s['repeat'] == repeat]
        if sorted(s['fold'] for s in group) != list(range(int(cfg['outer_folds']))):
            raise ValueError('Outer fold identities are incomplete or duplicated')
        for item in group:
            train, test = list(map(str, item['train_ids'])), list(map(str, item['test_ids']))
            if len(set(train)) != len(train) or len(set(test)) != len(test) or set(train) & set(test) or set(train) | set(test) != set(ids):
                raise ValueError('Invalid bottle-level outer split')
            item['train_ids'], item['test_ids'] = train, test
            item['tr'] = np.array([lookup[v] for v in train], dtype=int)
            item['te'] = np.array([lookup[v] for v in test], dtype=int)
            held.extend(test)
        if sorted(held) != sorted(ids):
            raise ValueError('Every bottle must be tested exactly once per repeat')
    return records


def task_plan(cfg, splits):
    tasks = []
    for split in splits:
        base = dict(repeat=int(split['repeat']), fold=int(split['fold']), bootstrap_refit=-1, condition='enhanced', training_size='full')
        tasks.append(dict(base, stage='benchmark'))
        for size in cfg.get('learning_sizes', [40, 70]):
            if int(size) != size or size < max(6, int(cfg['inner_folds']) * 2) or size >= len(split['tr']):
                raise ValueError('Learning sizes must be distinct valid sizes below every full outer-training count')
            tasks.append(dict(base, stage='learning', training_size=str(int(size))))
        if len(set(cfg.get('learning_sizes', [40, 70]))) != len(cfg.get('learning_sizes', [40, 70])):
            raise ValueError('Learning sizes must be unique')
        for condition in ('legacy', 'independent_snv', 'expanded_grid'):
            tasks.append(dict(base, stage='preprocessing', condition=condition))
        for bootstrap in range(int(cfg.get('bootstrap_refits', 5))):
            tasks.append(dict(base, stage='stability', bootstrap_refit=bootstrap))
        tasks.append(dict(base, stage='reliability'))
        if cfg.get('run_controls', True):
            for permutation in range(int(cfg.get('control_permutations', 3))):
                tasks.append(dict(base, stage='controls', condition='shuffled_nir', bootstrap_refit=permutation))
    return sorted(tasks, key=lambda t: (STAGES.index(t['stage']), t['repeat'], t['fold'], t['condition'], t['training_size'], t['bootstrap_refit']))


def task_name(task):
    return '{stage}_r{repeat}_f{fold}_c{condition}_n{training_size}_b{bootstrap_refit}.json.gz'.format(**task)


def indices(cfg, split, task):
    base = split['tr']
    seed = int(cfg['seed']) + 10000 * int(split['repeat']) + 100 * int(split['fold'])
    fit = base.copy()
    cal = np.array([], dtype=int)
    if task['stage'] == 'learning':
        fit = np.random.RandomState(seed + 700000).permutation(base)[:int(task['training_size'])]
    elif task['stage'] == 'stability':
        fit = np.random.RandomState(seed + 100000 + task['bootstrap_refit']).choice(base, len(base), replace=True)
    elif task['stage'] == 'reliability':
        shuffled = np.random.RandomState(seed + 500000).permutation(base)
        n_cal = int(math.ceil(len(base) * float(cfg.get('calibration_fraction', 0.3))))
        if not 0 < n_cal < len(base) - 5:
            raise ValueError('Insufficient independent proper-training or calibration bottles')
        cal, fit = shuffled[:n_cal], shuffled[n_cal:]
    return fit, cal, split['te'], seed


def task_config(cfg, task):
    result = copy.deepcopy(cfg)
    if task['stage'] == 'preprocessing':
        result['algorithms'] = ['pls', 'ridge', 'krr']
        result['preprocessing_options'] = {'uv': ['raw'], 'raman': ['raw', 'snv'], 'nir': ['raw', 'snv']}
        if task['condition'] in ('legacy', 'independent_snv'):
            result['grid'] = copy.deepcopy(cfg.get('legacy_grid', OLD_GRID))
    if task['stage'] == 'controls':
        result['algorithms'] = list(cfg.get('control_algorithms', ['pls', 'ridge', 'krr']))
    return result


def take(X, index):
    return {key: value[index] for key, value in X.items()}


def _acquisition(diagnostics, models):
    output = {}
    for model in models:
        if model == 'mean':
            output[model] = []
            continue
        if model in diagnostics.get('acquisition_by_model', {}):
            output[model] = list(diagnostics['acquisition_by_model'][model]['modality_union'])
            continue
        records = diagnostics.get('selection_by_model', diagnostics.get('selections', {}))[model]
        output[model] = [m for m in engine.MODALITIES if any(m in rec.get('mask', rec['spec']['mask']).split('+') for rec in records)]
    return output


def run_task(dataset, cfg, split, task):
    X, Y, ids = dataset.X, dataset.Y.to_numpy(float), dataset.ids.astype(str)
    targets = list(dataset.Y.columns)
    fit, cal, test, seed = indices(cfg, split, task)
    if set(ids[fit]) & set(ids[test]) or set(ids[fit]) & set(ids[cal]) or set(ids[test]) & set(ids[cal]):
        raise ValueError('Fitting, calibration and test bottle roles overlap')
    sd = Y[fit].std(axis=0, ddof=1)
    outer_sd = Y[split['tr']].std(axis=0, ddof=1)
    if not np.isfinite(sd).all() or np.any(sd <= 1e-12):
        raise ValueError('A training target has no measurable spread')
    local = task_config(cfg, task)
    evaluation = np.concatenate([cal, test])
    fit_X, eval_X = take(X, fit), take(X, evaluation)
    control = None
    if task['stage'] == 'controls':
        rng = np.random.RandomState(seed + 900000 + task['bootstrap_refit'])
        training_permutation, evaluation_permutation = rng.permutation(len(fit)), rng.permutation(len(evaluation))
        fit_X['nir'] = fit_X['nir'][training_permutation]
        eval_X['nir'] = eval_X['nir'][evaluation_permutation]
        control = dict(training_permutation=training_permutation.tolist(), evaluation_permutation=evaluation_permutation.tolist(), interpretation='Unconditional bottle-permuted NIR diagnostic; not a conditional-independence test or new cohort')
    legacy = task['stage'] == 'preprocessing' and task['condition'] == 'legacy'
    if legacy:
        predictions, diagnostics = legacy_engine.select_and_predict(fit_X, Y[fit], ids[fit], eval_X, local, seed)
        diagnostics['selection_by_model'] = diagnostics['selections']
    else:
        predictions, diagnostics = engine.select_and_predict(fit_X, Y[fit], ids[fit], eval_X, local, seed, axes=getattr(dataset, 'axes', None), return_oof=task['stage'] == 'reliability')
    n_cal = len(cal)
    test_predictions = {m: np.asarray(value, dtype=float)[n_cal:] for m, value in predictions.items()}
    models = sorted(predictions)
    expected_models = set(legacy_engine.STRATEGIES) if legacy else set(engine.model_names(local))
    if set(models) != expected_models:
        raise ValueError('A task did not produce every declared model')
    available = _acquisition(diagnostics, models)
    diagnostics.setdefault('acquisition_by_model', {})
    for model in models:
        diagnostics['acquisition_by_model'].setdefault(model, dict(modality_union=available[model], n_modalities=len(available[model])))
    result = dict(task, fit_ids=ids[fit].tolist(), cal_ids=ids[cal].tolist(), test_ids=ids[test].tolist(), model_names=models, targets=targets, diagnostics=diagnostics, control=control, predictions=[], intervals=[])
    base = {key: task[key] for key in ('stage', 'condition', 'training_size', 'repeat', 'fold', 'bootstrap_refit')}
    for model, prediction in test_predictions.items():
        if prediction.shape != (len(test), len(targets)) or not np.isfinite(prediction).all():
            raise ValueError('Incomplete or nonfinite final predictions')
        for i, identity in enumerate(ids[test]):
            for j, target in enumerate(targets):
                result['predictions'].append(dict(base, model=model, sample_id=identity, target=target, y_true=float(Y[test[i], j]), y_pred=float(prediction[i, j]), train_sd=float(sd[j]), outer_train_sd=float(outer_sd[j]), n_train=len(fit), n_unique_train=len(set(ids[fit])), n_cal=n_cal))
    if n_cal:
        cal_predictions = {m: np.asarray(value, dtype=float)[:n_cal] for m, value in predictions.items()}
        local.setdefault('uncertainty', {})['model_modalities'] = available
        intervals, uncertainty = build_intervals(Y[fit], fit_X, take(X, cal), take(X, test), diagnostics['oof_predictions'], cal_predictions, test_predictions, Y[cal], local)
        result['uncertainty'] = uncertainty
        result['calibration_truth'] = Y[cal].tolist()
        result['calibration_predictions'] = {m: value.tolist() for m, value in cal_predictions.items()}
        high_true = np.quantile(Y[fit], 0.8, axis=0)
        alpha = float(cfg['interval_alpha'])
        for model in models:
            high_prediction = np.asarray(uncertainty['predicted_high_thresholds'][model])
            for method, interval in intervals[model].items():
                lower, upper = interval['lower'], interval['upper']
                for i, identity in enumerate(ids[test]):
                    for j, target in enumerate(targets):
                        truth = Y[test[i], j]
                        lo, hi = float(lower[i, j]), float(upper[i, j])
                        width = hi - lo
                        score = width + (2 / alpha) * max(lo - truth, 0) + (2 / alpha) * max(truth - hi, 0)
                        result['intervals'].append(dict(base, model=model, sample_id=identity, target=target, interval_method=method, lower=lo, upper=hi, width=width, covered=bool(lo <= truth <= hi), fit_sd=float(sd[j]), n_fit=len(fit), n_cal=n_cal, rank=int(interval['rank']), high_true_concentration=bool(truth > high_true[j]), high_predicted_concentration=bool(test_predictions[model][i, j] > high_prediction[j]), interval_score=float(score)))
    return result


def read_cache(path, signature, task=None):
    wrapper = read_json(path)
    if wrapper.get('signature') != signature:
        raise RuntimeError('Execution signature changed; use a new output folder')
    payload = wrapper.get('payload')
    if payload_hash(payload) != wrapper.get('payload_sha256'):
        raise RuntimeError('Task cache is damaged: ' + str(path))
    if task is not None and any(payload.get(k) != value for k, value in task.items()):
        raise RuntimeError('Task cache identity differs from its declared plan: ' + str(path))
    return payload


def validate_payload(payload, dataset, cfg, split, task):
    fit, cal, test, unused = indices(cfg, split, task)
    ids, Y = dataset.ids.astype(str), dataset.Y
    if payload['fit_ids'] != ids[fit].tolist() or payload['cal_ids'] != ids[cal].tolist() or payload['test_ids'] != ids[test].tolist():
        raise RuntimeError('Task roles differ from the deterministic bottle plan')
    local = task_config(cfg, task)
    models = set(legacy_engine.STRATEGIES) if task['condition'] == 'legacy' else set(engine.model_names(local))
    rows = pd.DataFrame(payload['predictions'])
    n_expected = len(test) * len(Y.columns) * len(models)
    if len(rows) != n_expected or set(rows.model) != models or set(rows.target) != set(Y.columns) or set(rows.sample_id.astype(str)) != set(ids[test]) or rows.duplicated(['model', 'sample_id', 'target']).any():
        raise RuntimeError('Task prediction identities are incomplete or duplicated')
    for key in ('stage', 'condition', 'training_size', 'repeat', 'fold', 'bootstrap_refit'):
        if key not in rows or not rows[key].eq(task[key]).all():
            raise RuntimeError('Prediction row context differs from its task plan')
    if not rows.n_train.eq(len(fit)).all() or not rows.n_unique_train.eq(len(set(ids[fit]))).all() or not rows.n_cal.eq(len(cal)).all():
        raise RuntimeError('Prediction row fitting/calibration budget differs from task roles')
    if not np.isfinite(rows[['y_true', 'y_pred', 'train_sd']].to_numpy(float)).all():
        raise RuntimeError('Nonfinite task values')
    reference = Y.copy()
    reference.index = ids
    sd = Y.iloc[fit].to_numpy(float).std(axis=0, ddof=1)
    outer_sd = Y.iloc[split['tr']].to_numpy(float).std(axis=0, ddof=1)
    for j, target in enumerate(Y.columns):
        group = rows[rows.target.eq(target)]
        expected = reference.loc[group.sample_id.astype(str), target].to_numpy(float)
        if not np.array_equal(group.y_true.to_numpy(float), expected) or not np.allclose(group.train_sd.to_numpy(float), sd[j], rtol=0, atol=1e-12):
            raise RuntimeError('Cached concentration or fitting scale differs from source')
        if 'outer_train_sd' not in group or not np.allclose(group.outer_train_sd.to_numpy(float), outer_sd[j], rtol=0, atol=1e-12):
            raise RuntimeError('Cached assessment-only outer-training scale differs from source')
    interval_rows = pd.DataFrame(payload['intervals'])
    if len(cal):
        methods = {'constant_marginal', 'constant_joint', 'adaptive_marginal', 'adaptive_joint'}
        if len(interval_rows) != 4 * n_expected or set(interval_rows.interval_method) != methods or interval_rows.duplicated(['model', 'sample_id', 'target', 'interval_method']).any():
            raise RuntimeError('Incomplete interval constructions')
        keys = ['model', 'sample_id', 'target']
        for key in ('stage', 'condition', 'training_size', 'repeat', 'fold', 'bootstrap_refit'):
            if key not in interval_rows or not interval_rows[key].eq(task[key]).all():
                raise RuntimeError('Interval row context differs from its task plan')
        merged = interval_rows.merge(rows[keys+['y_true', 'y_pred', 'train_sd']], on=keys, how='left', validate='many_to_one')
        if merged.y_true.isna().any() or set(interval_rows.model) != models or set(interval_rows.target) != set(Y.columns):
            raise RuntimeError('Interval identity does not match a declared point prediction')
        lower = merged.lower.to_numpy(float)
        upper = merged.upper.to_numpy(float)
        width = merged.width.to_numpy(float)
        truth = merged.y_true.to_numpy(float)
        covered_text = merged.covered.astype(str).str.lower()
        if np.isnan(lower).any() or np.isnan(upper).any() or np.any(lower > upper) or not np.allclose(width, upper-lower, rtol=1e-10, atol=1e-12):
            raise RuntimeError('Invalid cached interval bounds or width')
        if not covered_text.isin(['true', 'false']).all() or not np.array_equal(covered_text.eq('true').to_numpy(), (lower <= truth) & (truth <= upper)):
            raise RuntimeError('Cached coverage indicator disagrees with reference values')
        rank = int(math.ceil((len(cal)+1)*(1-float(cfg['interval_alpha']))))
        if not merged['rank'].eq(rank).all() or not merged.n_fit.eq(len(fit)).all() or not merged.n_cal.eq(len(cal)).all() or not np.allclose(merged.fit_sd.to_numpy(float), merged.train_sd.to_numpy(float)):
            raise RuntimeError('Interval calibration rank or fitting budget differs from protocol')
        expected_score = width+(2/float(cfg['interval_alpha']))*np.maximum(lower-truth, 0)+(2/float(cfg['interval_alpha']))*np.maximum(truth-upper, 0)
        if not np.allclose(merged.interval_score.to_numpy(float), expected_score, rtol=1e-10, atol=1e-12):
            raise RuntimeError('Cached interval score differs from saved bounds and references')
        high_reference = dict(zip(Y.columns, np.quantile(Y.iloc[fit].to_numpy(float), 0.8, axis=0)))
        expected_true_group = truth > merged.target.map(high_reference).to_numpy(float)
        saved_true_group = merged.high_true_concentration.astype(str).str.lower()
        if not saved_true_group.isin(['true', 'false']).all() or not np.array_equal(saved_true_group.eq('true').to_numpy(), expected_true_group):
            raise RuntimeError('Cached true-concentration diagnostic differs from fitting thresholds')
        levels = payload['uncertainty']['predicted_high_thresholds']
        target_index = {target: j for j, target in enumerate(Y.columns)}
        threshold = np.array([levels[model][target_index[target]] for model, target in zip(merged.model, merged.target)], dtype=float)
        saved_prediction_group = merged.high_predicted_concentration.astype(str).str.lower()
        if not saved_prediction_group.isin(['true', 'false']).all() or not np.array_equal(saved_prediction_group.eq('true').to_numpy(), merged.y_pred.to_numpy(float) > threshold):
            raise RuntimeError('Cached prediction-based diagnostic differs from training OOF thresholds')
    elif len(interval_rows):
        raise RuntimeError('Uncalibrated task contains intervals')


def consolidate(out, cfg, splits, plan, signature, dataset):
    lookup = {(s['repeat'], s['fold']): s for s in splits}
    stages = {}
    for stage in STAGES:
        tasks = [t for t in plan if t['stage'] == stage]
        records, intervals, missing = [], [], []
        for task in tasks:
            path = out / 'tasks' / task_name(task)
            if not path.exists():
                missing.append(task_name(task))
                continue
            payload = read_cache(path, signature, task)
            validate_payload(payload, dataset, cfg, lookup[(task['repeat'], task['fold'])], task)
            records.extend(payload['predictions'])
            intervals.extend(payload['intervals'])
        if records:
            folder = out / stage
            folder.mkdir(exist_ok=True)
            pd.DataFrame(records).to_csv(folder / 'predictions.csv', index=False, encoding='utf-8')
            if intervals:
                pd.DataFrame(intervals).to_csv(folder / 'intervals.csv', index=False, encoding='utf-8')
        stages[stage] = dict(complete=bool(tasks) and not missing, enabled=bool(tasks), completed_tasks=len(tasks)-len(missing), expected_tasks=len(tasks), missing_tasks=missing, prediction_rows=len(records), interval_rows=len(intervals))
    status = 'all_requested_stages_complete' if all(item['complete'] for item in stages.values() if item['enabled']) else 'incomplete'
    dump(out / 'COMPLETENESS.json', dict(signature=signature, profile=cfg['profile'], status=status, stages=stages))
    return stages


def main():
    parser = argparse.ArgumentParser(description='V6 retrospective information-value and reliability experiments')
    parser.add_argument('--config', default='configs/full.json')
    parser.add_argument('--out', default='results/new_full')
    parser.add_argument('--stage', choices=['prepare', 'all', 'summarize'] + list(STAGES), default='all')
    parser.add_argument('--max-tasks', type=int, default=None)
    args = parser.parse_args()
    cfg_path = Path(args.config)
    cfg = read_json(cfg_path if cfg_path.is_absolute() else ROOT / cfg_path)
    out = Path(args.out)
    out = out.resolve() if out.is_absolute() else (ROOT / out).resolve()
    protected = [ROOT, ROOT / 'data', ROOT / 'configs', ROOT / 'provenance', ROOT / 'results' / 'archived_v4']
    if out == ROOT or any(out == p.resolve() or p.resolve() in out.parents for p in protected[1:]):
        raise ValueError('Use a new results directory outside immutable data, configuration and archive folders')
    if args.max_tasks is not None and args.max_tasks < 1:
        raise ValueError('--max-tasks must be positive')
    out.mkdir(parents=True, exist_ok=True)
    with Lock(out):
        signature, context = execution_signature(cfg)
        if (out / 'protocol.json').exists() and read_json(out / 'protocol.json')['signature'] != signature:
            raise RuntimeError('Code, data, configuration or runtime changed; choose a fresh output directory')
        dataset = load_dataset(ROOT / 'data', strict=True)
        if dataset.Y.shape != (int(cfg['expected_bottles']), 11):
            raise ValueError('The declared independent bottle/endpoint count differs from the input')
        source = read_json(ROOT / 'provenance' / 'inputs.json')
        for filename, digest in source['converted_input_sha256'].items():
            if sha256(ROOT / 'data' / filename) != digest:
                raise ValueError('Converted measurement input hash changed: ' + filename)
        splits = make_splits(dataset.ids, cfg)
        plan = task_plan(cfg, splits)
        dump(out / 'protocol.json', dict(signature=signature, context=context, models=engine.model_names(cfg), targets=list(TARGETS), study_status='Retrospective extension of a previously inspected cohort; not new external validation'))
        dump(out / 'data_audit.json', dataset.audit)
        dump(out / 'splits.json', [{k: v for k, v in s.items() if k not in ('tr', 'te')} for s in splits])
        dump(out / 'task_plan.json', plan)
        selected = [t for t in plan if args.stage == 'all' or t['stage'] == args.stage]
        if args.stage in ('prepare', 'summarize'):
            selected = []
        print('Planned tasks by stage: ' + ', '.join('%s=%d' % (stage, sum(t['stage'] == stage for t in plan)) for stage in STAGES), flush=True)
        lookup = {(s['repeat'], s['fold']): s for s in splits}
        executed = 0
        for i, task in enumerate(selected, 1):
            path = out / 'tasks' / task_name(task)
            print('[%d/%d] %s' % (i, len(selected), path.name), flush=True)
            if path.exists():
                cached = read_cache(path, signature, task)
                validate_payload(cached, dataset, cfg, lookup[(task['repeat'], task['fold'])], task)
                print('  verified cached task reused', flush=True)
                continue
            if args.max_tasks is not None and executed >= args.max_tasks:
                print('Task limit reached; saved results remain resumable and incomplete.', flush=True)
                break
            start = time.time()
            payload = run_task(dataset, cfg, lookup[(task['repeat'], task['fold'])], task)
            validate_payload(payload, dataset, cfg, lookup[(task['repeat'], task['fold'])], task)
            dump(path, dict(signature=signature, payload_sha256=payload_hash(payload), payload=payload))
            executed += 1
            print('  completed %.1f seconds; %d models, %d endpoints' % (time.time()-start, len(payload['model_names']), len(payload['targets'])), flush=True)
        completion = consolidate(out, cfg, splits, plan, signature, dataset)
        if args.stage != 'prepare':
            from summarize import summarize
            summarize(out, cfg)
        compact = {stage: {key: value[key] for key in ('enabled', 'complete', 'completed_tasks', 'expected_tasks')} for stage, value in completion.items()}
        print(json.dumps(dict(output=str(out), status=read_json(out / 'COMPLETENESS.json')['status'], stages=compact), indent=2), flush=True)


if __name__ == '__main__':
    main()
