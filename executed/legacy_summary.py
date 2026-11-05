from pathlib import Path
import hashlib
import json
import warnings
import numpy as np
import pandas as pd
TARGETS = {'acetaldehyde': 'Acetaldehyde', 'methanol': 'Methanol', 'n_propanol': 'n-Propanol', 'ethyl_acetate': 'Ethyl acetate', 'isobutanol': 'Isobutanol', 'acetal': 'Acetal', 'n_butanol': 'n-Butanol', 'isoamyl_alcohol': 'Isoamyl alcohol', 'ethyl_butyrate': 'Ethyl butyrate', 'ethyl_lactate': 'Ethyl lactate', 'ethyl_hexanoate': 'Ethyl hexanoate*'}
ALGORITHMS = ('pls', 'ridge', 'krr')
MASKS = ('uv', 'raman', 'nir', 'uv+raman', 'uv+nir', 'raman+nir', 'uv+raman+nir')
STRATEGIES = MASKS + ('select_single', 'select_subset')
MODELS = ('mean',) + tuple((a + '__' + m for a in ALGORITHMS for m in STRATEGIES))
MASK_NAMES = {'uv': 'UV', 'raman': 'Raman', 'nir': 'NIR', 'uv+raman': 'UV + Raman', 'uv+nir': 'UV + NIR', 'raman+nir': 'Raman + NIR', 'uv+raman+nir': 'All three', 'select_single': 'Training-selected single', 'select_subset': 'Training-selected subset'}
EDGES = [('uv', 'uv+raman'), ('raman', 'uv+raman'), ('uv', 'uv+nir'), ('nir', 'uv+nir'), ('raman', 'raman+nir'), ('nir', 'raman+nir'), ('uv+raman', 'uv+raman+nir'), ('uv+nir', 'uv+raman+nir'), ('raman+nir', 'uv+raman+nir')]

def _save(path, frame):
    frame.to_csv(path, index=False, encoding='utf-8-sig')

def _label(model):
    if model == 'mean':
        return 'Training mean'
    algorithm, mask = model.split('__', 1)
    return algorithm.upper() + ' / ' + MASK_NAMES[mask]

def _fmt(x, digits=5):
    if pd.isna(x):
        return 'Undefined'
    if np.isposinf(x):
        return '∞'
    if np.isneginf(x):
        return '−∞'
    return ('%.' + str(digits) + 'f') % x

def _table(headers, rows):
    return '| ' + ' | '.join(headers) + ' |\n| ' + ' | '.join(['---'] * len(headers)) + ' |\n' + ''.join(('| ' + ' | '.join(map(str, r)) + ' |\n' for r in rows))

def _metrics(g):
    y, p = (g.y_true.to_numpy(float), g.y_pred.to_numpy(float))
    e = p - y
    total = np.sum((y - np.mean(y)) ** 2)
    return dict(n_visits=len(g), n_unique_bottles=g.sample_id.nunique(), rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(abs(e))), bias=float(np.mean(e)), r2=float(1 - np.sum(e ** 2) / total) if total > 0 else np.nan, standardized_mse=float(np.mean((e / g.train_sd.to_numpy(float)) ** 2)))

def _group_metrics(df, keys):
    rows = []
    for values, g in df.groupby(keys, sort=False):
        if not isinstance(values, tuple):
            values = (values,)
        row = dict(zip(keys, values))
        row.update(_metrics(g))
        rows.append(row)
    return pd.DataFrame(rows)

def _validate_predictions(df):
    required = {'model', 'repeat', 'fold', 'bootstrap_refit', 'sample_id', 'target', 'y_true', 'y_pred', 'train_sd', 'n_train', 'n_unique_train'}
    if not required.issubset(df):
        raise ValueError('Prediction columns absent: ' + str(required - set(df)))
    if set(df.model.unique()) != set(MODELS):
        raise ValueError('A saved stage lacks the expected 28 strategies')
    if set(df.target.unique()) != set(TARGETS):
        raise ValueError('All 11 endpoints must be present')
    keys = ['model', 'repeat', 'fold', 'bootstrap_refit', 'sample_id', 'target']
    if df.duplicated(keys).any():
        raise ValueError('Duplicate prediction identities')
    if not np.isfinite(df[['y_true', 'y_pred', 'train_sd']].to_numpy(float)).all():
        raise ValueError('Nonfinite prediction or target scale')
    if (df.train_sd <= 0).any():
        raise ValueError('Target training scale must be positive')
    sizes = df.groupby(['repeat', 'fold', 'bootstrap_refit', 'model', 'target']).size()
    if not sizes.groupby(level=[0, 1, 2]).apply(lambda g: g.nunique() == 1).all():
        raise ValueError('Unequal evaluation counts across methods/endpoints')
    truth = df.groupby(['repeat', 'fold', 'bootstrap_refit', 'sample_id', 'target']).y_true.nunique()
    if not truth.eq(1).all():
        raise ValueError('Reference concentrations differ across strategies')
    global_truth = df.groupby(['sample_id', 'target']).y_true.nunique()
    if not global_truth.eq(1).all():
        raise ValueError('Reference concentrations differ across repeated evaluations')
    budget = df.groupby(['repeat', 'fold', 'bootstrap_refit', 'target'])[['train_sd', 'n_train', 'n_unique_train']].nunique()
    if not budget.eq(1).all().all():
        raise ValueError('Training budgets/scales differ across strategies')

def _gain(candidate, reference):
    if reference <= 0 or not np.isfinite(reference):
        return np.nan
    return float(100 * (1 - candidate / reference))

def _effects(df, draws, seed, do_bootstrap=True):
    rows, edges = ([], [])
    rng = np.random.RandomState(seed)
    for target in TARGETS:
        d = df[df.target.eq(target)].copy()
        d['squared_error'] = (d.y_pred - d.y_true) ** 2
        bottle = d.groupby(['sample_id', 'model']).squared_error.mean().unstack('model')
        if bottle.isna().any().any():
            raise ValueError('Paired bottle predictions are incomplete')
        n = len(bottle)
        ix = rng.randint(0, n, size=(draws, n)) if do_bootstrap and draws > 0 else None
        rmses = {m: float(np.sqrt(bottle[m].mean())) for m in MODELS}
        for algorithm in ALGORITHMS:
            candidate = algorithm + '__select_subset'
            for mask in MASKS + ('select_single',):
                reference = algorithm + '__' + mask
                gain = _gain(rmses[candidate], rmses[reference])
                lo = hi = np.nan
                if ix is not None:
                    c = np.sqrt(bottle[candidate].to_numpy()[ix].mean(axis=1))
                    r = np.sqrt(bottle[reference].to_numpy()[ix].mean(axis=1))
                    valid = r > 0
                    values = 100 * (1 - c[valid] / r[valid])
                    if len(values):
                        lo, hi = np.quantile(values, [0.025, 0.975])
                rows.append(dict(target=target, algorithm=algorithm, candidate=candidate, reference=reference, n_unique_bottles=n, rmse_candidate=rmses[candidate], rmse_reference=rmses[reference], gain_pct=gain, conditional_ci_low_pct=lo, conditional_ci_high_pct=hi, ci_scope='Bottle-cluster fixed-prediction conditional 95% interval; no refitting or multiplicity guarantee'))
            for before, after in EDGES:
                a, b = (algorithm + '__' + before, algorithm + '__' + after)
                edges.append(dict(target=target, algorithm=algorithm, before=before, after=after, rmse_before=rmses[a], rmse_after=rmses[b], gain_pct=_gain(rmses[b], rmses[a]), interpretation='Observed direct modality-addition contrast; not a chemical causal effect'))
    return (pd.DataFrame(rows), pd.DataFrame(edges))

def _bool(series):
    text = series.astype(str).str.lower()
    if not text.isin(['true', 'false', '1', '0']).all():
        raise ValueError('Invalid boolean column')
    return text.isin(['true', '1'])

def _intervals(df, point):
    keys = ['model', 'repeat', 'fold', 'sample_id', 'target']
    if df.duplicated(keys + ['interval_method']).any():
        raise ValueError('Duplicate interval identities')
    if set(df.interval_method.unique()) != {'constant_marginal', 'constant_joint'}:
        raise ValueError('Unexpected interval methods')
    if len(df) != 2 * len(point):
        raise ValueError('Incomplete interval records')
    if set(df.model.unique()) != set(MODELS) or set(df.target.unique()) != set(TARGETS):
        raise ValueError('Interval strategies/endpoints incomplete')
    truth = point[keys + ['y_true']]
    d = df.merge(truth, on=keys, how='left', validate='many_to_one')
    if d.y_true.isna().any():
        raise ValueError('Interval rows lack matching point predictions')
    if (d.upper < d.lower).any():
        raise ValueError('Reversed interval endpoints')
    expected = (d.y_true >= d.lower) & (d.y_true <= d.upper)
    if not np.array_equal(expected.to_numpy(), _bool(d.covered).to_numpy()):
        raise ValueError('Saved coverage differs from interval endpoints')
    if not np.allclose((d.upper - d.lower).to_numpy(), d.width.to_numpy(), equal_nan=False):
        raise ValueError('Saved width differs from endpoints')
    d['covered'] = expected
    d['high_true_concentration'] = _bool(d.high_true_concentration)
    rows = []
    for stratum in ['all', 'high_true_concentration']:
        subset = d if stratum == 'all' else d[d.high_true_concentration]
        for (model, method, target), g in subset.groupby(['model', 'interval_method', 'target'], sort=False):
            rows.append(dict(stratum=stratum, model=model, interval_method=method, target=target, n_visits=len(g), n_unique_bottles=g.sample_id.nunique(), coverage=g.covered.mean(), mean_width=g.width.mean(), median_width=g.width.median(), mean_standardized_width=np.mean(g.width / g.fit_sd), infinite_width_fraction=np.isinf(g.width).mean()))
    joint_rows = []
    for (model, method), g in d.groupby(['model', 'interval_method'], sort=False):
        group = g.groupby(['repeat', 'fold', 'sample_id'])
        if not group.target.nunique().eq(11).all():
            raise ValueError('Incomplete joint target vector')
        covered = group.covered.all()
        joint_rows.append(dict(model=model, interval_method=method, n_vector_visits=len(covered), n_unique_bottles=g.sample_id.nunique(), joint_coverage=covered.mean(), mean_standardized_width=np.mean(g.width / g.fit_sd)))
    return (pd.DataFrame(rows), pd.DataFrame(joint_rows))

def _plot(out, stage, metrics, effects):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = Path('<historical_environment>/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        family = 'Microsoft YaHei'
    else:
        family = 'DejaVu Sans'
    plt.rcParams.update({'font.family': family, 'axes.unicode_minus': False, 'pdf.fonttype': 42, 'font.size': 10, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 3, figsize=(17, 8), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.96, top=0.86, bottom=0.17, wspace=0.16)
    y = np.arange(11)
    for ax, algorithm in zip(axes, ALGORITHMS):
        for offset, mask, color in [(-0.2, 'select_single', '#356C8C'), (0, 'uv+raman+nir', '#8494A3'), (0.2, 'select_subset', '#BE8424')]:
            d = metrics[metrics.model.eq(algorithm + '__' + mask)].set_index('target').loc[list(TARGETS)]
            ax.scatter(d.r2, y + offset, color=color, label=MASK_NAMES[mask], s=30)
        ax.set_title(algorithm.upper())
        ax.axvline(0, color='#777777', lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(list(TARGETS.values()))
        ax.grid(axis='x', alpha=0.2)
        ax.set_xlabel('Pooled R2 / larger is better')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    axes[0].invert_yaxis()
    axes[0].legend(loc='lower left', fontsize=8)
    fig.suptitle(stage + ': equal display of eleven targets across three algorithms', fontsize=16, y=0.96)
    fig.text(0.08, 0.09, 'R2 pools saved outer-test predictions. Fitting budgets differ across stages; do not rank them directly.', fontsize=10)
    fig.text(0.08, 0.045, 'Repeated predictions do not add independent samples. *The legacy ester identity requires author confirmation.', fontsize=9)
    fig.savefig(out / (stage + '_all_11_targets.png'), dpi=180, bbox_inches='tight')
    fig.savefig(out / (stage + '_all_11_targets.pdf'), bbox_inches='tight')
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(19, 9), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.93, top=0.88, bottom=0.27, wspace=0.13)
    for ax, algorithm in zip(axes, ALGORITHMS):
        matrix = np.array([[float(metrics[(metrics.model == algorithm + '__' + mask) & metrics.target.eq(target)].iloc[0].r2) for mask in STRATEGIES] for target in TARGETS])
        art = ax.imshow(matrix, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
        for i in range(11):
            for j in range(9):
                value = matrix[i, j]
                ax.text(j, i, _fmt(value, 2), ha='center', va='center', fontsize=6.5, color='white' if abs(value) > 0.65 else '#1E2F3C')
        ax.set_xticks(np.arange(9))
        ax.set_xticklabels([MASK_NAMES[m] for m in STRATEGIES], rotation=55, ha='right', fontsize=9)
        ax.set_yticks(np.arange(11))
        ax.set_yticklabels(list(TARGETS.values()), fontsize=10)
        ax.set_title(algorithm.upper())
        ax.axvline(6.5, color='black', lw=1.3)
    bar = fig.colorbar(art, ax=axes, fraction=0.022, pad=0.025)
    bar.set_label('Pooled R2; color scale fixed from -1 to 1')
    fig.suptitle(stage + ': seven fixed masks and two training-only selectors across eleven targets', fontsize=16, y=0.97)
    fig.text(0.075, 0.1, 'The seven columns left of the divider are fixed masks; the last two are training-selected strategies. Labels retain actual R2 values; colors clip at -1.', fontsize=10)
    fig.text(0.075, 0.055, 'Original target and configuration order is retained. No target is highlighted. Interpret stages separately. *Legacy chemical identity requires confirmation.', fontsize=9)
    fig.savefig(out / (stage + '_seven_masks_and_selection.png'), dpi=180, bbox_inches='tight')
    fig.savefig(out / (stage + '_seven_masks_and_selection.pdf'), bbox_inches='tight')
    plt.close(fig)

def summarize(out_dir, cfg):
    out = Path(out_dir)
    derived = out / 'derived'
    derived.mkdir(parents=True, exist_ok=True)
    draw = int(cfg.get('summary_bootstrap_draws', 2000))
    seed = int(cfg.get('seed', 20261029))
    if draw < 0:
        raise ValueError('summary_bootstrap_draws must be nonnegative')
    report = ['# Seven fixed spectral configurations and training-only selection across eleven equally analysed targets\n', 'PLS, Ridge and RBF-KRR each use seven fixed masks, a training-selected single spectrum and a training-selected subset: 27 learned strategies plus a training-mean baseline. All eleven targets receive the same evaluation and display weight.\n', 'Every fixed mask tunes its parameters within receiving-training data. Selectors reuse the same inner candidate scores. Outer labels never guide selection. Algorithms are compared in parallel without presenting a post-hoc winner as independent validation.\n', 'Errors and interval widths use g/L. Overall standardized MSE scales each squared error with its receiving-training target SD, then weights all eleven targets equally. Raw concentration RMSE values are not averaged across targets.\n', 'This is a retrospective reanalysis of an existing cohort. Conditional bottle-bootstrap intervals hold fitted predictions fixed, exclude refitting variation and provide no multiplicity-adjusted significance claim. Refit observed ranges are not confidence intervals.\n']
    result = {'available_stages': [], 'missing_stages': [], 'files': {}, 'models_refitted_by_summary': False}
    complete_path = out / 'COMPLETENESS.json'
    complete = json.loads(complete_path.read_text(encoding='utf-8')) if complete_path.exists() else {}
    result['execution_status'] = complete.get('status', 'not_provided')
    report.append('Execution status: `' + result['execution_status'] + '`. Only the runner completeness record establishes whether all planned tasks finished. Partial saved predictions may be summarized as partial results only.\n')
    selection_rows = []
    for path in sorted((out / 'tasks').glob('*.json')):
        task = json.loads(path.read_text(encoding='utf-8')).get('payload', {})
        stage_name = task.get('stage')
        for model, selections in task.get('diagnostics', {}).get('selections', {}).items():
            if not model.endswith(('__select_single', '__select_subset')):
                continue
            for selection in selections:
                j = int(selection['target_index'])
                if j < 0 or j >= len(TARGETS):
                    raise ValueError('Invalid target index in selection record')
                selection_rows.append(dict(stage=stage_name, model=model, target=list(TARGETS)[j], mask=selection['spec']['mask']))
    selection_frequency = pd.DataFrame()
    if selection_rows:
        selection_frequency = pd.DataFrame(selection_rows).groupby(['stage', 'model', 'target', 'mask']).size().reset_index(name='n_selections')
        selection_frequency['fraction'] = selection_frequency.n_selections / selection_frequency.groupby(['stage', 'model', 'target']).n_selections.transform('sum')
        _save(derived / 'selection_frequency.csv', selection_frequency)
    elif cfg.get('archived_selection_frequency', False):
        selection_path = derived / 'selection_frequency.csv'
        if selection_path.exists():
            selection_frequency = pd.read_csv(selection_path)
            required = {'stage', 'model', 'target', 'mask', 'n_selections', 'fraction'}
            if not required.issubset(selection_frequency):
                raise ValueError('Archived selection-frequency columns are incomplete')
            if not set(selection_frequency.target).issubset(TARGETS):
                raise ValueError('Archived selection-frequency endpoints differ')
    all_data = {}
    for stage in ['benchmark', 'stability', 'reliability']:
        path = out / stage / 'predictions.csv'
        if not path.exists():
            result['missing_stages'].append(stage)
            report.append('## ' + stage + '\n\nNo saved predictions exist for this stage; missing values are not replaced with zeros.\n')
            continue
        df = pd.read_csv(path, dtype={'sample_id': str})
        _validate_predictions(df)
        all_data[stage] = df
        result['available_stages'].append(stage)
        result['files'][str(path.relative_to(out))] = hashlib.sha256(path.read_bytes()).hexdigest()
        stage_dir = derived / stage
        stage_dir.mkdir(exist_ok=True)
        metric = _group_metrics(df, ['model', 'target'])
        repeat_keys = ['model', 'repeat', 'target'] + (['bootstrap_refit'] if stage == 'stability' else [])
        repeat_metric = _group_metrics(df, repeat_keys)
        _save(stage_dir / 'target_metrics_pooled.csv', metric)
        _save(stage_dir / 'target_metrics_by_repeat.csv', repeat_metric)
        macro_keys = ['model', 'repeat'] + (['bootstrap_refit'] if stage == 'stability' else [])
        macro = repeat_metric.groupby(macro_keys, sort=False).standardized_mse.mean().reset_index(name='macro_standardized_mse')
        _save(stage_dir / 'macro_standardized_mse.csv', macro)
        effects, edges = _effects(df, draw, seed, do_bootstrap=stage != 'stability')
        _save(stage_dir / 'subset_vs_all_fixed_and_selected_single.csv', effects)
        _save(stage_dir / 'nine_modality_addition_edges.csv', edges)
        if stage == 'stability':
            per_b = []
            for b, g in df.groupby('bootstrap_refit'):
                ef, _ = _effects(g, 0, seed, False)
                ef['bootstrap_refit'] = b
                per_b.append(ef)
            refit = pd.concat(per_b, ignore_index=True)
            _save(stage_dir / 'subset_effects_by_refit.csv', refit)
            ranges = refit.groupby(['target', 'algorithm', 'candidate', 'reference'], sort=False).gain_pct.agg(n_refits='count', mean_gain_pct='mean', min_gain_pct='min', max_gain_pct='max').reset_index()
            wins = refit.assign(improved=refit.gain_pct.gt(0)).groupby(['target', 'algorithm', 'candidate', 'reference'], sort=False).improved.sum().reset_index(name='n_improved')
            ranges = ranges.merge(wins, on=['target', 'algorithm', 'candidate', 'reference'], validate='one_to_one')
            _save(stage_dir / 'refit_observed_ranges.csv', ranges)
        elif cfg.get('summary_make_figures', True):
            _plot(derived, stage, metric, effects)
        report.append('## ' + stage + '\n')
        state = complete.get('stages', {}).get(stage, {})
        report.append('Stage completeness: ' + ('Complete' if state.get('complete') is True else 'Completeness not confirmed; results may contain partial tasks') + '.\n')
        report.append('Independent bottles: %d; prediction visits per strategy and target: %s. Evaluate this stage separately from other fitting budgets.\n' % (df.sample_id.nunique(), ', '.join(map(str, sorted(df.groupby(['model', 'target']).size().unique())))))
        macro_avg = macro.groupby('model').macro_standardized_mse.mean()
        report.append(_table(['Strategy', 'Equal-weight standardized MSE across eleven targets'], [[_label(m), _fmt(macro_avg.loc[m], 6)] for m in MODELS]))
        if stage != 'stability' and cfg.get('summary_make_figures', True):
            report.append('\n![Seven masks and selection](' + stage + '_seven_masks_and_selection.png)\n')
            report.append('\n![All eleven targets](' + stage + '_all_11_targets.png)\n')
        for target, name in TARGETS.items():
            report.append('### ' + name + '\n')
            mt = metric[metric.target.eq(target)].set_index('model')
            if not selection_frequency.empty:
                sf = selection_frequency[selection_frequency.stage.eq(stage) & selection_frequency.target.eq(target)]
                report.append('Selection frequencies are descriptive, not chemical contributions or future selection probabilities.\n')
                report.append(_table(['Selection strategy', 'Selected configuration', 'Tasks', 'Fraction'], [[_label(z.model), MASK_NAMES[z.mask], int(z.n_selections), _fmt(100 * z.fraction, 2) + '%'] for z in sf.itertuples()]))
            report.append(_table(['Strategy', 'RMSE', 'MAE', 'R²'], [[_label(m), _fmt(mt.loc[m, 'rmse'], 6), _fmt(mt.loc[m, 'mae'], 6), _fmt(mt.loc[m, 'r2'], 4)] for m in MODELS]))
            er = effects[effects.target.eq(target)].set_index(['algorithm', 'reference'])
            if stage != 'stability':
                rows = []
                for algorithm in ALGORITHMS:
                    for reference in ['select_single', 'uv+raman+nir']:
                        row = er.loc[algorithm, algorithm + '__' + reference]
                        rows.append([algorithm.upper(), MASK_NAMES[reference], _fmt(row.gain_pct, 2) + '%', _fmt(row.conditional_ci_low_pct, 2) + '% to ' + _fmt(row.conditional_ci_high_pct, 2) + '%'])
                report.append('\n' + _table(['Subset-selection algorithm', 'Reference', 'RMSE reduction', 'Conditional 95% interval for fixed predictions'], rows))
            else:
                r = ranges[ranges.target.eq(target)].set_index(['algorithm', 'reference'])
                rows = []
                for algorithm in ALGORITHMS:
                    for reference in ['select_single', 'uv+raman+nir']:
                        z = r.loc[algorithm, algorithm + '__' + reference]
                        rows.append([algorithm.upper(), MASK_NAMES[reference], _fmt(z.mean_gain_pct, 2) + '%', '%d/%d' % (z.n_improved, z.n_refits), _fmt(z.min_gain_pct, 2) + '% to ' + _fmt(z.max_gain_pct, 2) + '%'])
                report.append('\n' + _table(['Algorithm', 'Reference', 'Mean gain', 'Improved groups', 'Observed range, not a confidence interval'], rows))
    path = out / 'reliability' / 'intervals.csv'
    if path.exists():
        if 'reliability' not in all_data:
            raise ValueError('Intervals exist without point predictions')
        ci = pd.read_csv(path, dtype={'sample_id': str})
        intervals, joint = _intervals(ci, all_data['reliability'])
        _save(derived / 'reliability' / 'interval_coverage_width_by_target.csv', intervals)
        _save(derived / 'reliability' / 'joint_coverage_width.csv', joint)
        result['files'][str(path.relative_to(out))] = hashlib.sha256(path.read_bytes()).hexdigest()
        report.append('## Interval reliability for individual targets and the complete eleven-target vector\n')
        report.append('The nominal level is fixed in the configuration. Marginal target coverage is distinct from simultaneous coverage. Joint coverage must be evaluated alongside width. High concentration is defined from true reference values above a fitting-set threshold for retrospective diagnosis only. Finite observed coverage below the nominal level does not itself disprove a marginal theoretical guarantee.\n')
        report.append(_table(['Strategy', 'Interval', 'Observed simultaneous coverage of eleven targets', 'Mean standardized width'], [[_label(r.model), r.interval_method, _fmt(r.joint_coverage * 100, 2) + '%', _fmt(r.mean_standardized_width, 4)] for r in joint.itertuples()]))
        for target, name in TARGETS.items():
            report.append('### ' + name + ' - intervals\n')
            it = intervals[intervals.target.eq(target)]
            rows = []
            for model in MODELS:
                for method in ['constant_marginal', 'constant_joint']:
                    for stratum in ['all', 'high_true_concentration']:
                        z = it[it.model.eq(model) & it.interval_method.eq(method) & it.stratum.eq(stratum)]
                        if z.empty:
                            continue
                        z = z.iloc[0]
                        rows.append([_label(model), method, stratum, int(z.n_unique_bottles), int(z.n_visits), _fmt(z.coverage * 100, 2) + '%', _fmt(z.mean_width, 6)])
            report.append(_table(['Strategy', 'Interval', 'Stratum', 'Unique bottles', 'Prediction visits', 'Coverage', 'Mean width'], rows))
    report.append('## Interpretation and reading order\n\nRead the seven fixed masks within each algorithm before evaluating the two training-only selectors. Compare subset selection with training-selected single spectra and fixed full fusion. The nine modality-addition edges describe predictive contrasts, not molecular specificity or causal effects.\n\nRetain all eleven targets, including unfavorable outcomes. Cross-algorithm minima are descriptive. Preprocessing, tuning and input selection belong entirely inside receiving-training data. This cohort supplies no new external samples, absolute laboratory specification limits or measured instrument costs. *The legacy ester identity still requires author confirmation.\n')
    report_path = derived / 'all_11_targets_report.md'
    report_path.write_text('\n'.join(report), encoding='utf-8')
    result['target_order'] = list(TARGETS)
    result['strategy_order'] = list(MODELS)
    result['summary_bootstrap_draws'] = draw
    (derived / 'summary_provenance.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result
