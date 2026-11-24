from pathlib import Path
import hashlib
import json
import math
import numpy as np
import pandas as pd
from scipy import stats

MASKS = ('uv', 'raman', 'nir', 'uv+raman', 'uv+nir', 'raman+nir', 'uv+raman+nir')
ALGORITHMS = ('pls', 'ridge', 'krr', 'sopls')
EDGES = (('uv', 'uv+raman'), ('raman', 'uv+raman'), ('uv', 'uv+nir'), ('nir', 'uv+nir'), ('raman', 'raman+nir'), ('nir', 'raman+nir'), ('uv+raman', 'uv+raman+nir'), ('uv+nir', 'uv+raman+nir'), ('raman+nir', 'uv+raman+nir'))
TARGETS = ('acetaldehyde', 'methanol', 'n_propanol', 'ethyl_acetate', 'isobutanol', 'acetal', 'n_butanol', 'isoamyl_alcohol', 'ethyl_butyrate', 'ethyl_lactate', 'ethyl_hexanoate')
REQUIRED = ('condition', 'training_size', 'model', 'repeat', 'fold', 'sample_id', 'target', 'y_true', 'y_pred', 'train_sd', 'outer_train_sd', 'n_train', 'n_unique_train')
CONTEXT = ['condition', 'training_size', 'refit_index']
PAIR_KEY = ['repeat', 'fold', 'sample_id', 'target']
CI_SCOPE = 'Conditional paired bottle bootstrap of frozen OOF losses; no refitting, cross-bottle training dependence correction or multiplicity guarantee'
CV_SCOPE = 'Approximate corrected repeated-CV sensitivity; not an exact test and not independent external validation'


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def _dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False), encoding='utf-8')


def _save(path, frame):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format='%.17g')


def _seed(base, values):
    encoded = json.dumps(list(values), ensure_ascii=True, separators=(',', ':')).encode('ascii')
    return (int(base) + int(hashlib.sha256(encoded).hexdigest()[:8], 16)) % (2 ** 32 - 1)


def holm_adjust(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError('Holm adjustment requires a one-dimensional planned family')
    finite = np.isfinite(values)
    if np.any((values[finite] < 0) | (values[finite] > 1)):
        raise ValueError('Finite p-values must lie between zero and one')
    result = np.full(len(values), np.nan)
    order = np.flatnonzero(finite)[np.argsort(values[finite], kind='mergesort')]
    previous = 0.0
    for rank, position in enumerate(order):
        previous = max(previous, min(1.0, (len(values) - rank) * values[position]))
        result[position] = previous
    return result


def corrected_cv(differences, test_train_ratios):
    differences = np.asarray(differences, dtype=float)
    ratios = np.asarray(test_train_ratios, dtype=float)
    if differences.ndim != 1 or ratios.shape != differences.shape or len(differences) < 2:
        raise ValueError('Corrected CV requires matched task losses and test/train ratios')
    if not np.isfinite(differences).all() or not np.isfinite(ratios).all() or np.any(ratios <= 0):
        raise ValueError('Corrected CV inputs must be finite with positive ratios')
    n = len(differences)
    effect = float(differences.mean())
    variance = float(differences.var(ddof=1))
    ratio = float(ratios.mean())
    se = math.sqrt((1.0 / n + ratio) * variance)
    if se == 0:
        p = 1.0 if effect == 0 else np.nan
        low = high = effect
        status = 'exact_zero_contrast' if effect == 0 else 'constant_nonzero_task_difference'
    else:
        quantile = float(stats.t.ppf(0.975, n - 1))
        p = float(2 * stats.t.sf(abs(effect / se), n - 1))
        low, high = effect - quantile * se, effect + quantile * se
        status = 'approximate'
    return dict(mean_fold_mse_difference_after_minus_before=effect, corrected_se=se,
                corrected_ci_low=low, corrected_ci_high=high, p_uncorrected=p,
                n_dependent_tasks=n, test_train_ratio=ratio, degrees_of_freedom=n - 1,
                df_convention='number_of_CV_tasks_minus_one', test_status=status,
                inference_scope=CV_SCOPE)


def bootstrap_draws(n_bottles, draws, seed):
    if n_bottles < 2 or int(draws) != draws or draws < 1:
        raise ValueError('Bootstrap requires at least two bottles and a positive integer draw count')
    return np.random.RandomState(int(seed)).randint(n_bottles, size=(int(draws), n_bottles)).astype(np.int32)


def conditional_pair(squared_before, squared_after, standardized_before, standardized_after, indices, chunk_size=128):
    arrays = [np.asarray(value, dtype=float) for value in (squared_before, squared_after, standardized_before, standardized_after)]
    if any(value.ndim != 2 or value.shape != arrays[0].shape for value in arrays):
        raise ValueError('Paired bottle losses must have matching bottle-by-target shapes')
    if any(not np.isfinite(value).all() or np.any(value < 0) for value in arrays):
        raise ValueError('Loss arrays must be finite and nonnegative')
    n, q = arrays[0].shape
    indices = np.asarray(indices, dtype=int)
    if indices.ndim != 2 or indices.shape[1] != n or np.any(indices < 0) or np.any(indices >= n):
        raise ValueError('Bootstrap indices must select complete matched bottle rows')
    b, a, zb, za = arrays
    rmse_b, rmse_a = np.sqrt(b.mean(axis=0)), np.sqrt(a.mean(axis=0))
    macro_b, macro_a = float(zb.mean()), float(za.mean())
    gain = np.divide(100 * (rmse_b - rmse_a), rmse_b, out=np.full(q, np.nan), where=rmse_b > 0)
    boot_gain = np.empty((len(indices), q))
    boot_rmse_difference = np.empty_like(boot_gain)
    boot_macro_difference = np.empty(len(indices))
    boot_macro_gain = np.empty(len(indices))
    for start in range(0, len(indices), int(chunk_size)):
        index = indices[start:start + int(chunk_size)]
        bb, aa = np.sqrt(b[index].mean(axis=1)), np.sqrt(a[index].mean(axis=1))
        mb, ma = zb[index].mean(axis=(1, 2)), za[index].mean(axis=(1, 2))
        boot_gain[start:start + len(index)] = np.divide(100 * (bb - aa), bb, out=np.full(bb.shape, np.nan), where=bb > 0)
        boot_rmse_difference[start:start + len(index)] = aa - bb
        boot_macro_difference[start:start + len(index)] = ma - mb
        boot_macro_gain[start:start + len(index)] = np.divide(100 * (mb - ma), mb, out=np.full(len(mb), np.nan), where=mb > 0)
    def bounds(value):
        return np.nanquantile(value, [0.025, 0.975], axis=0)
    return dict(rmse_before=rmse_b, rmse_after=rmse_a, target_gain_pct=gain,
                target_gain_bounds=bounds(boot_gain), target_rmse_difference_after_minus_before=rmse_a-rmse_b,
                target_rmse_difference_bounds=bounds(boot_rmse_difference),
                macro_mse_before=macro_b, macro_mse_after=macro_a,
                macro_mse_difference_after_minus_before=macro_a-macro_b,
                macro_mse_difference_bounds=bounds(boot_macro_difference),
                macro_gain_pct=100*(1-macro_a/macro_b) if macro_b > 0 else np.nan,
                macro_gain_bounds=bounds(boot_macro_gain),
                bootstrap_target_gains=boot_gain, bootstrap_macro_differences=boot_macro_difference)


def read_predictions(path):
    path = Path(path)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = sorted(set(REQUIRED) - set(header))
    if missing:
        raise ValueError('Missing prediction role or metric columns: ' + ', '.join(missing))
    extra = ['stage', 'budget_refit', 'bootstrap_refit', 'effective_nir_weight', 'n_acquisition_modalities']
    columns = list(REQUIRED) + [value for value in extra if value in header]
    frame = pd.read_csv(path, usecols=columns, dtype={'sample_id': str, 'training_size': str}, float_precision='round_trip')
    return normalize_predictions(frame)


def normalize_predictions(frame):
    frame = frame.copy()
    missing = sorted(set(REQUIRED) - set(frame.columns))
    if missing:
        raise ValueError('Missing prediction role or metric columns: ' + ', '.join(missing))
    frame['sample_id'] = frame.sample_id.astype(str)
    frame['training_size'] = frame.training_size.astype(str)
    refit = pd.Series(-1, index=frame.index, dtype=int)
    if 'bootstrap_refit' in frame:
        refit = frame.bootstrap_refit.fillna(-1).astype(int)
    if 'budget_refit' in frame:
        budget = frame.budget_refit.fillna(-1).astype(int)
        refit = budget.where(budget.ge(0), refit)
    frame['refit_index'] = refit
    if not np.isfinite(frame[['y_true', 'y_pred', 'train_sd', 'outer_train_sd', 'n_train', 'n_unique_train']].to_numpy(float)).all():
        raise ValueError('Prediction values and training roles must be finite')
    if (frame[['train_sd', 'outer_train_sd', 'n_train', 'n_unique_train']] <= 0).any().any():
        raise ValueError('Training scales and bottle counts must be positive')
    if (frame.n_unique_train > frame.n_train).any():
        raise ValueError('Distinct training bottles cannot exceed training rows')
    if set(frame.target) != set(TARGETS):
        raise ValueError('All eleven declared targets are required')
    key = CONTEXT + ['model'] + PAIR_KEY
    if frame.duplicated(key).any():
        raise ValueError('Repeated prediction keys violate paired bottle roles')
    task_target = CONTEXT + ['model', 'repeat', 'fold', 'target']
    sizes = frame.groupby(task_target, sort=False).size()
    if sizes.groupby(level=list(range(len(task_target)-1))).nunique().gt(1).any():
        raise ValueError('A task has incomplete target coverage')
    roles = frame.groupby(CONTEXT + ['model', 'repeat', 'fold'], sort=False)[['n_train','n_unique_train']].nunique()
    if roles.gt(1).any().any():
        raise ValueError('Training bottle roles are inconsistent within a task')
    if frame.groupby(CONTEXT + ['model', 'repeat', 'fold'], sort=False).target.nunique().ne(11).any():
        raise ValueError('Every model task must retain all eleven targets')
    frame['squared_error'] = (frame.y_pred-frame.y_true)**2
    frame['standardized_squared_error'] = frame.squared_error/frame.train_sd**2
    frame['common_scale_squared_error'] = frame.squared_error/frame.outer_train_sd**2
    return frame


def _metric(group):
    truth, prediction = group.y_true.to_numpy(float), group.y_pred.to_numpy(float)
    error = prediction-truth
    denominator = float(np.sum((truth-truth.mean())**2))
    return dict(n_prediction_visits=len(group), n_unique_bottles=group.sample_id.nunique(),
                rmse=float(np.sqrt(np.mean(error**2))), mae=float(np.mean(np.abs(error))),
                bias=float(np.mean(error)), r2=1-float(np.sum(error**2))/denominator if denominator > 0 else np.nan,
                standardized_mse=float(group.standardized_squared_error.mean()),
                common_scale_mse=float(group.common_scale_squared_error.mean()),
                n_train=int(group.n_train.iloc[0]), n_unique_train=int(group.n_unique_train.iloc[0]),
                n_train_min=int(group.n_train.min()), n_train_max=int(group.n_train.max()),
                n_unique_train_min=int(group.n_unique_train.min()), n_unique_train_max=int(group.n_unique_train.max()))


def target_metrics(frame, keys):
    output = []
    for values, group in frame.groupby(keys, sort=False):
        if not isinstance(values, tuple):
            values = (values,)
        output.append(dict(zip(keys, values), **_metric(group)))
    return pd.DataFrame(output)


def panel_metrics(target_table, keys):
    output = []
    for values, group in target_table.groupby(keys, sort=False):
        if not isinstance(values, tuple):
            values = (values,)
        if set(group.target) != set(TARGETS) or len(group) != 11:
            raise ValueError('Equal-target panel metrics require exactly eleven targets')
        record = dict(zip(keys, values))
        record.update(macro_standardized_mse=float(group.standardized_mse.mean()),
                      macro_common_scale_mse=float(group.common_scale_mse.mean()),
                      n_targets=11, n_unique_test_bottles=int(group.n_unique_bottles.iloc[0]),
                      n_train=int(group.n_train.iloc[0]), n_unique_train=int(group.n_unique_train.iloc[0]),
                      n_train_min=int(group.n_train_min.min()) if 'n_train_min' in group else int(group.n_train.min()),
                      n_train_max=int(group.n_train_max.max()) if 'n_train_max' in group else int(group.n_train.max()),
                      n_unique_train_min=int(group.n_unique_train_min.min()) if 'n_unique_train_min' in group else int(group.n_unique_train.min()),
                      n_unique_train_max=int(group.n_unique_train_max.max()) if 'n_unique_train_max' in group else int(group.n_unique_train.max()))
        output.append(record)
    return pd.DataFrame(output)


def validate_complete_pair(before, after, expected_ids, split_records=None, require_full=True):
    if before.model.nunique() != 1 or after.model.nunique() != 1:
        raise ValueError('A paired comparison must contain one model on each side')
    a = before.sort_values(PAIR_KEY).reset_index(drop=True)
    b = after.sort_values(PAIR_KEY).reset_index(drop=True)
    if len(a) != len(b) or not a[PAIR_KEY].equals(b[PAIR_KEY]):
        raise ValueError('Paired comparison is missing matching bottle/target/outer roles')
    if not np.array_equal(a.y_true.to_numpy(), b.y_true.to_numpy()):
        raise ValueError('Paired models disagree on reference values')
    for column in ('train_sd', 'outer_train_sd', 'n_train', 'n_unique_train'):
        if not np.array_equal(a[column].to_numpy(), b[column].to_numpy()):
            raise ValueError('Paired models have unequal training roles or scales: '+column)
    if require_full:
        if set(a['repeat']) != {0,1,2} or set(a.fold) != {0,1,2,3,4}:
            raise ValueError('Conditional intervals require complete three-repeat fivefold contexts')
        if len(a) != len(expected_ids)*3*11:
            raise ValueError('Conditional intervals require all bottle visits and eleven targets')
        for repeat, group in a.groupby('repeat', sort=False):
            if set(group.sample_id) != set(expected_ids):
                raise ValueError('Each repeat must cover the complete independent bottle cohort')
            sizes = group.groupby(['sample_id','target']).size()
            if len(sizes) != len(expected_ids)*11 or not sizes.eq(1).all():
                raise ValueError('Each bottle-target must occur once per repeat')
        if a.groupby(['repeat','fold']).ngroups != 15:
            raise ValueError('Fifteen paired outer tasks are required')
        if a.n_train.lt(111).any() or a.n_train.gt(112).any() or not a.n_train.eq(a.n_unique_train).all():
            raise ValueError('Conditional intervals require complete unique 111/112-bottle training partitions')
    if split_records is not None:
        expected = {(int(s['repeat']),int(s['fold'])):set(map(str,s['test_ids'])) for s in split_records}
        for key, group in a.groupby(['repeat','fold'],sort=False):
            if key not in expected or set(group.sample_id) != expected[key]:
                raise ValueError('Prediction test roles differ from saved outer split identities')
    return a, b


def _paired_arrays(before, after):
    ids = sorted(before.sample_id.unique())
    arrays=[]
    for frame in (before, after):
        grouped = frame.groupby(['sample_id','target'])[['squared_error','standardized_squared_error']].mean()
        raw=grouped.squared_error.unstack('target').reindex(index=ids,columns=TARGETS).to_numpy(float)
        standardized=grouped.standardized_squared_error.unstack('target').reindex(index=ids,columns=TARGETS).to_numpy(float)
        arrays.append((raw,standardized))
    return ids, arrays[0][0], arrays[1][0], arrays[0][1], arrays[1][1]


def paired_descriptive_effects(frame,expected_ids,archive=False):
    targets,panels=[] ,[]
    for values,context in frame.groupby(CONTEXT,sort=False):
        for algorithm,before_name,after_name,before_model,after_model in _pairs_for_models(set(context.model),archive):
            before=context[context.model.eq(before_model)]
            after=context[context.model.eq(after_model)]
            before,after=validate_complete_pair(before,after,expected_ids,require_full=False)
            ids,raw_b,raw_a,std_b,std_a=_paired_arrays(before,after)
            meta=dict(zip(CONTEXT,values),algorithm=algorithm,before=before_name,after=after_name,
                before_model=before_model,after_model=after_model,contrast=before_name+'__to__'+after_name,
                n_unique_bottles=len(ids),n_prediction_visits=len(before)//11)
            rb,ra=np.sqrt(raw_b.mean(axis=0)),np.sqrt(raw_a.mean(axis=0))
            for j,target in enumerate(TARGETS):
                targets.append(dict(meta,target=target,rmse_before=rb[j],rmse_after=ra[j],
                    rmse_difference_after_minus_before=ra[j]-rb[j],
                    gain_pct=100*(1-ra[j]/rb[j]) if rb[j]>0 else np.nan,
                    summary_scope='Matched saved predictions; no interval or independent-replication claim'))
            common_b=float(before.common_scale_squared_error.mean())
            common_a=float(after.common_scale_squared_error.mean())
            mb,ma=float(std_b.mean()),float(std_a.mean())
            panels.append(dict(meta,standardized_mse_before=mb,standardized_mse_after=ma,
                delta_mse_after_minus_before=ma-mb,gain_pct=100*(1-ma/mb) if mb>0 else np.nan,
                common_scale_mse_before=common_b,common_scale_mse_after=common_a,
                common_scale_delta_after_minus_before=common_a-common_b,
                common_scale_gain_pct=100*(1-common_a/common_b) if common_b>0 else np.nan,
                summary_scope='Matched saved predictions; common outer-training scale retained for learning-budget comparisons'))
    return pd.DataFrame(targets),pd.DataFrame(panels)


def _pairs_for_models(models, archive):
    if archive:
        return [(algorithm,before,after,algorithm+'__'+before,algorithm+'__'+after)
                for before,after in EDGES for algorithm in ALGORITHMS]
    result=[]
    for algorithm in ALGORITHMS:
        a,b=algorithm+'__uv+raman',algorithm+'__uv+raman+nir'
        if a in models and b in models:
            result.append((algorithm,'uv+raman','uv+raman+nir',a,b))
    for algorithm,a,b in [('krr','krr__weight_zero','krr__weight_selected'),
                          ('ridge_residual','ridge_residual__zero','ridge_residual__selected')]:
        if a in models and b in models:
            result.append((algorithm,'zero','selected',a,b))
    return result


def paired_statistics(frame, expected_ids, cfg, archive=False, split_records=None):
    target_rows,macro_rows,corrected_targets,corrected_panels,statuses=[],[],[],[],[]
    seed=int(cfg.get('statistics_seed',20261104))
    draws=int(cfg.get('statistical_bootstrap_draws',2000))
    indices=bootstrap_draws(len(expected_ids),draws,seed)
    for values,context in frame.groupby(CONTEXT,sort=False):
        context_record=dict(zip(CONTEXT,values))
        pairs=_pairs_for_models(set(context.model),archive)
        if not pairs:
            statuses.append(dict(context_record,status='no_declared_paired_models',conditional_intervals=False))
            continue
        learning=str(context_record['training_size'])!='full'
        stability='stability' in str(context_record['condition']).lower() or context.n_unique_train.lt(context.n_train).any()
        for algorithm,before_name,after_name,before_model,after_model in pairs:
            meta=dict(context_record,algorithm=algorithm,before=before_name,after=after_name,
                      before_model=before_model,after_model=after_model,
                      contrast=before_name+'__to__'+after_name)
            before=context[context.model.eq(before_model)]
            after=context[context.model.eq(after_model)]
            if learning or stability:
                validate_complete_pair(before,after,expected_ids,require_full=False)
                statuses.append(dict(meta,status='learning_or_refit_descriptive_only',conditional_intervals=False))
                continue
            try:
                before,after=validate_complete_pair(before,after,expected_ids,split_records=split_records,require_full=True)
            except ValueError as error:
                statuses.append(dict(meta,status='incomplete_paired_context',reason=str(error),conditional_intervals=False))
                continue
            ids,raw_b,raw_a,std_b,std_a=_paired_arrays(before,after)
            if len(ids)!=len(expected_ids):
                raise ValueError('Bootstrap bottle count differs from expected cohort')
            result=conditional_pair(raw_b,raw_a,std_b,std_a,indices)
            for j,target in enumerate(TARGETS):
                target_rows.append(dict(meta,target=target,n_unique_bottles=len(ids),n_prediction_visits=len(before)//11,
                    rmse_before=result['rmse_before'][j],rmse_after=result['rmse_after'][j],
                    gain_pct=result['target_gain_pct'][j],conditional_ci_low_pct=result['target_gain_bounds'][0,j],
                    conditional_ci_high_pct=result['target_gain_bounds'][1,j],
                    rmse_difference_after_minus_before=result['target_rmse_difference_after_minus_before'][j],
                    rmse_difference_ci_low=result['target_rmse_difference_bounds'][0,j],
                    rmse_difference_ci_high=result['target_rmse_difference_bounds'][1,j],
                    bootstrap_draws=draws,bootstrap_seed=seed,interval_scope=CI_SCOPE,
                    gain_direction='positive_gain_means_after_improves',difference_direction='positive_difference_means_after_worsens'))
            macro_rows.append(dict(meta,n_unique_bottles=len(ids),n_targets=11,
                standardized_mse_before=result['macro_mse_before'],standardized_mse_after=result['macro_mse_after'],
                delta_mse_after_minus_before=result['macro_mse_difference_after_minus_before'],
                conditional_delta_ci_low=result['macro_mse_difference_bounds'][0],
                conditional_delta_ci_high=result['macro_mse_difference_bounds'][1],gain_pct=result['macro_gain_pct'],
                conditional_gain_ci_low_pct=result['macro_gain_bounds'][0],conditional_gain_ci_high_pct=result['macro_gain_bounds'][1],
                bootstrap_draws=draws,bootstrap_seed=seed,interval_scope=CI_SCOPE,
                delta_direction='positive_delta_means_after_worsens',gain_direction='positive_gain_means_after_improves'))
            paired=before[PAIR_KEY+['squared_error','standardized_squared_error','n_train']].merge(
                after[PAIR_KEY+['squared_error','standardized_squared_error']],on=PAIR_KEY,suffixes=('_before','_after'),validate='one_to_one')
            paired['loss_difference']=paired.squared_error_after-paired.squared_error_before
            paired['standardized_difference']=paired.standardized_squared_error_after-paired.standardized_squared_error_before
            by_fold=paired.groupby(['repeat','fold','target'],sort=False).agg(loss_difference=('loss_difference','mean'),
                standardized_difference=('standardized_difference','mean'),n_test=('sample_id','nunique'),n_train=('n_train','first')).reset_index()
            for target,group in by_fold.groupby('target',sort=False):
                corrected_targets.append(dict(meta,target=target,loss='raw_unit_mean_squared_error',
                    **corrected_cv(group.loss_difference.to_numpy(),(group.n_test/group.n_train).to_numpy())))
            panel=by_fold.groupby(['repeat','fold'],sort=False).agg(loss_difference=('standardized_difference','mean'),
                n_test=('n_test','first'),n_train=('n_train','first'))
            corrected_panels.append(dict(meta,loss='equal_target_standardized_mean_squared_error',
                **corrected_cv(panel.loss_difference.to_numpy(),(panel.n_test/panel.n_train).to_numpy())))
            statuses.append(dict(meta,status='complete_paired_conditional_analysis',conditional_intervals=True))
    target_ci,macro_ci=pd.DataFrame(target_rows),pd.DataFrame(macro_rows)
    target_cv,panel_cv=pd.DataFrame(corrected_targets),pd.DataFrame(corrected_panels)
    for table,scope,n_family in [(target_cv,'all_algorithms_times_eleven_targets',44),(panel_cv,'all_algorithms_panel_comparisons',4)]:
        if table.empty:
            continue
        table['p_holm']=np.nan
        table['planned_family_size']=0
        table['multiplicity_scope']=scope
        for values,group in table.groupby(CONTEXT+['contrast'],sort=False):
            fixed_four_family=group.contrast.iloc[0]=='uv+raman__to__uv+raman+nir'
            expected=max(n_family,len(group)) if archive or fixed_four_family else len(group)
            adjusted=holm_adjust(np.r_[group.p_uncorrected.to_numpy(),np.full(expected-len(group),np.nan)])[:len(group)]
            table.loc[group.index,'p_holm']=adjusted
            table.loc[group.index,'planned_family_size']=expected
    return dict(target_conditional_intervals=target_ci,panel_conditional_intervals=macro_ci,
                target_corrected_cv_sensitivity=target_cv,panel_corrected_cv_sensitivity=panel_cv,
                conditional_context_status=pd.DataFrame(statuses),
                draw_metadata=dict(n_bottles=len(expected_ids),draws=draws,seed=seed,
                    draw_indices_sha256=hashlib.sha256(np.ascontiguousarray(indices).tobytes()).hexdigest(),
                    pairing='One shared bottle draw index matrix for every target, model and comparison; all visits remain together',
                    intervals=CI_SCOPE,corrected_cv=CV_SCOPE))


def _concentrations_and_dimensions(root,out):
    chemistry=pd.read_csv(root/'executed/data/chemistry.csv',dtype={'sample_id':str},float_precision='round_trip')
    if list(chemistry.columns)!=['sample_id']+list(TARGETS):
        raise ValueError('Chemistry target order differs from the eleven declared endpoints')
    rows=[]
    for target in TARGETS:
        a=chemistry[target].to_numpy(float)
        q1,median,q3=np.quantile(a,[.25,.5,.75])
        rows.append(dict(target=target,n=len(a),source_unit='g/L',minimum=a.min(),maximum=a.max(),
            mean=a.mean(),sd_ddof1=a.std(ddof=1),median=median,q1=q1,q3=q3,iqr=q3-q1,
            n_zero=int((a==0).sum()),n_negative=int((a<0).sum()),
            identity_scope='Original endpoint mapping retained; ethyl_hexanoate and acetal identity require laboratory confirmation'))
    _save(out/'concentration_summary.csv',pd.DataFrame(rows))
    features={}
    for modality in ('uv','raman','nir'):
        with np.load(str(root/('executed/data/spectra_'+modality+'.npz')),allow_pickle=False) as arrays:
            features[modality]=int(arrays['spectra'].shape[1])
    dimensions=[]
    for mask in MASKS:
        p=sum(features[m] for m in mask.split('+'))
        for n,label in [(139,'whole_cohort'),(111,'outer_training'),(112,'outer_training'),(40,'learning_budget'),(70,'learning_budget')]:
            dimensions.append(dict(mask=mask,p=p,n=n,p_over_n=float(p)/n,n_scope=label,n_modalities=len(mask.split('+')),
                interpretation='Coordinate count, not effective rank or independent chemical information'))
    _save(out/'spectral_dimensions_and_sample_ratios.csv',pd.DataFrame(dimensions))
    return chemistry.sample_id.astype(str).tolist()


def _observed_ranges(table,keys,value_columns):
    if table.empty:
        return pd.DataFrame()
    rows=[]
    for values,group in table.groupby(keys,sort=False):
        if not isinstance(values,tuple):
            values=(values,)
        record=dict(zip(keys,values),n_observed_refit_groups=group.refit_index.nunique(),
                    range_scope='Observed refit or budget-draw range; not a confidence interval or independent replication')
        for column in value_columns:
            record[column+'_mean']=float(group[column].mean())
            record[column+'_min']=float(group[column].min())
            record[column+'_max']=float(group[column].max())
        rows.append(record)
    return pd.DataFrame(rows)


def _write_tables(out,tables):
    written={}
    for name,table in tables.items():
        if isinstance(table,pd.DataFrame) and len(table.columns):
            path=out/(name+'.csv')
            _save(path,table)
            written[name]=dict(path=(Path('derived')/path.name).as_posix(),rows=len(table),sha256=_sha(path))
    return written


def _selection_from_predictions(frame):
    columns=[name for name in ('effective_nir_weight','n_acquisition_modalities') if name in frame]
    if not columns:
        return pd.DataFrame(),pd.DataFrame()
    keys=CONTEXT+['model','repeat','fold']
    group=frame.groupby(keys,sort=False)
    if group[columns].nunique(dropna=False).gt(1).any().any():
        raise ValueError('A common acquisition gate must be constant across targets and bottles within a task')
    tasks=group[columns].first().reset_index()
    if 'effective_nir_weight' in tasks:
        available=tasks.effective_nir_weight.notna()
        tasks['effective_nir_present']=np.where(available,tasks.effective_nir_weight.gt(0).astype(float),np.nan)
        tasks['effective_modality_count']=np.where(available,2+tasks.effective_nir_weight.gt(0).astype(int),np.nan)
        for index,row in tasks[~available].iterrows():
            suffix=row.model.split('__',1)[-1]
            if suffix in MASKS:
                tasks.loc[index,'effective_nir_present']=float('nir' in suffix.split('+'))
                tasks.loc[index,'effective_modality_count']=len(suffix.split('+'))
    values=columns+(['effective_nir_present','effective_modality_count'] if 'effective_nir_weight' in tasks else [])
    frequency=tasks.groupby(['condition','training_size','model']+values,sort=False,dropna=False).size().reset_index(name='n_tasks')
    frequency['fraction']=frequency.n_tasks/frequency.groupby(['condition','training_size','model']).n_tasks.transform('sum')
    frequency['frequency_scope']='Dependent task frequency; no binomial confidence interval'
    return tasks,frequency


def analyze_archive(root,out,cfg):
    root,out=Path(root),Path(out)
    out.mkdir(parents=True,exist_ok=True)
    derived=out/'derived'
    derived.mkdir(exist_ok=True)
    ids=_concentrations_and_dimensions(root,derived)
    source=root/'results/baseline/benchmark/predictions.csv.gz'
    frame=read_predictions(source)
    wanted={algorithm+'__'+mask for algorithm in ALGORITHMS for mask in MASKS}
    fixed=frame[frame.model.isin(wanted)].copy()
    if set(fixed.model)!=wanted:
        raise ValueError('Archive must contain all seven masks and four algorithm families')
    fold=target_metrics(fixed,CONTEXT+['model','repeat','fold','target'])
    panel=panel_metrics(fold,CONTEXT+['model','repeat','fold'])
    if len(fold)!=4620 or len(panel)!=420:
        raise ValueError('Archive fixed-mask fold exports must contain 4620 target and 420 panel rows')
    pooled=target_metrics(frame,CONTEXT+['model','target'])
    by_repeat=target_metrics(frame,CONTEXT+['model','repeat','target'])
    repeated_panels=panel_metrics(by_repeat,CONTEXT+['model','repeat'])
    statistics=paired_statistics(fixed,ids,cfg,archive=True,split_records=_json(root/'results/baseline/splits.json'))
    tables=dict(archive_fixed_fold_target_metrics=fold,archive_fixed_fold_panel_metrics=panel,
                archive_all_strategy_target_metrics=pooled,archive_all_strategy_panel_by_repeat=repeated_panels)
    tables.update({key:value for key,value in statistics.items() if isinstance(value,pd.DataFrame)})
    evidence=root/'results/baseline/derived/stability/refit_observed_ranges.csv'
    if evidence.exists():
        table=pd.read_csv(evidence,float_precision='round_trip')
        table['inference_scope']='Five observed refit-index groups across dependent outer tasks; not a CI'
        tables['archive_five_refit_observed_ranges']=table
    for name in ('acquisition_union_by_task','acquisition_union_frequency','selection_frequency'):
        source_table=root/'results/baseline/derived'/(name+'.csv')
        if source_table.exists():
            table=pd.read_csv(source_table,float_precision='round_trip')
            if 'stage' in table:
                table=table[table.stage.isin(['benchmark','stability'])].copy()
            table['inference_scope']='Observed dependent selection frequency or task record; no binomial interval'
            tables['archive_'+name]=table
    written=_write_tables(derived,tables)
    written['concentration_summary']=dict(path='derived/concentration_summary.csv',rows=11,sha256=_sha(derived/'concentration_summary.csv'))
    written['spectral_dimensions_and_sample_ratios']=dict(path='derived/spectral_dimensions_and_sample_ratios.csv',rows=35,sha256=_sha(derived/'spectral_dimensions_and_sample_ratios.csv'))
    registry=dict(status='archived_predictions_postprocessed',model_fits_performed=0,analysis_source_sha256=_sha(Path(__file__)),
        source=source.relative_to(root).as_posix(),source_sha256=_sha(source),n_independent_bottles=len(ids),
        n_prediction_visits_per_strategy_target=417,fixed_fold_target_rows=len(fold),fixed_fold_panel_rows=len(panel),
        tables=written,statistics=statistics['draw_metadata'],
        source_results_unchanged=True,primary_outcome='Equal mean of eleven training-variance-standardized squared prediction errors',
        delta_mse_direction='after_minus_before; positive means adding NIR or applying the after strategy worsens prediction',
        target_gain_direction='100*(1-RMSE_after/RMSE_before); positive means improvement')
    _dump(out/'analysis_registry.json',registry)
    return registry


def analyze_run(root,run_dir,cfg):
    root,run_dir=Path(root),Path(run_dir)
    derived=run_dir/'derived'
    derived.mkdir(parents=True,exist_ok=True)
    source=run_dir/'predictions.csv.gz'
    if not source.exists():
        raise FileNotFoundError('The runner must consolidate completed prediction rows before analysis: '+str(source))
    ids=pd.read_csv(root/'executed/data/chemistry.csv',dtype={'sample_id':str},usecols=['sample_id']).sample_id.tolist()
    frame=read_predictions(source)
    fold=target_metrics(frame,CONTEXT+['model','repeat','fold','target'])
    panel=panel_metrics(fold,CONTEXT+['model','repeat','fold'])
    pooled=target_metrics(frame,CONTEXT+['model','target'])
    repeated=target_metrics(frame,CONTEXT+['model','repeat','target'])
    panel_repeat=panel_metrics(repeated,CONTEXT+['model','repeat'])
    splits=_json(run_dir/'splits.json') if (run_dir/'splits.json').exists() else None
    if isinstance(splits,dict):
        splits=splits.get('splits')
    statistics=paired_statistics(frame,ids,cfg,archive=False,split_records=splits)
    tasks,frequency=_selection_from_predictions(frame)
    target_effects,panel_effects=paired_descriptive_effects(frame,ids)
    target_ranges=_observed_ranges(target_effects,['condition','training_size','algorithm','contrast','target'],
        ['rmse_difference_after_minus_before','gain_pct'])
    panel_ranges=_observed_ranges(panel_effects,['condition','training_size','algorithm','contrast'],
        ['delta_mse_after_minus_before','gain_pct','common_scale_delta_after_minus_before','common_scale_gain_pct'])
    if not target_ranges.empty:
        target_ranges=target_ranges[target_ranges.n_observed_refit_groups.gt(1)].copy()
    if not panel_ranges.empty:
        panel_ranges=panel_ranges[panel_ranges.n_observed_refit_groups.gt(1)].copy()
    tables=dict(fold_target_metrics=fold,fold_panel_metrics=panel,pooled_target_metrics=pooled,
                panel_by_repeat_and_refit=panel_repeat,acquisition_by_task=tasks,acquisition_frequency=frequency,
                paired_target_effects=target_effects,paired_panel_effects=panel_effects,
                target_observed_budget_or_refit_ranges=target_ranges,panel_observed_budget_or_refit_ranges=panel_ranges)
    tables.update({key:value for key,value in statistics.items() if isinstance(value,pd.DataFrame)})
    written=_write_tables(derived,tables)
    completeness=_json(run_dir/'COMPLETENESS.json') if (run_dir/'COMPLETENESS.json').exists() else {}
    registry=dict(status='completed_rows_postprocessed',execution_status=completeness.get('status','not_recorded'),analysis_source_sha256=_sha(Path(__file__)),
        model_fits_performed=0,source=source.name,source_sha256=_sha(source),n_independent_bottles=len(ids),
        n_test_bottles_represented=int(frame.sample_id.nunique()),tables=written,statistics=statistics['draw_metadata'],
        conditional_intervals='Only complete paired full-size three-repeat fivefold contexts with distinct training bottles',
        learning_and_stability='Descriptive matched summaries; ranges and frequencies are not confidence intervals',
        delta_mse_direction='after_minus_before; positive means after worsens',
        target_gain_direction='positive gain means after improves')
    _dump(run_dir/'analysis_registry.json',registry)
    return registry
