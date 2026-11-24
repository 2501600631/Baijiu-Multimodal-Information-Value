from pathlib import Path
import hashlib
import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image
from data_io import TARGETS

MASKS = ('uv', 'raman', 'nir', 'uv+raman', 'uv+nir', 'raman+nir', 'uv+raman+nir')
SHORT = {'uv': 'U', 'raman': 'R', 'nir': 'N', 'uv+raman': 'UR', 'uv+nir': 'UN', 'raman+nir': 'RN', 'uv+raman+nir': 'URN', 'select_single': 'SS', 'select_subset': 'SUB', 'panel_t000': 'P0', 'panel_t002': 'P2', 'panel_t005': 'P5', 'panel_t010': 'P10'}
ALG = {'pls': 'PLS', 'ridge': 'Ridge', 'krr': 'RBF-KRR', 'sopls': 'SO-PLS'}
ABBR = {'acetaldehyde': 'AcH', 'methanol': 'MeOH', 'n_propanol': 'nPrOH', 'ethyl_acetate': 'EA', 'isobutanol': 'iBuOH', 'acetal': 'Acetal', 'n_butanol': 'nBuOH', 'isoamyl_alcohol': 'iAmOH', 'ethyl_butyrate': 'EB', 'ethyl_lactate': 'EL', 'ethyl_hexanoate': 'EH*'}
INK = '#20343F'
TEAL = '#276C79'
ORANGE = '#B16A3D'
GRAY = '#8A9DA7'
BLUE = '#315F87'
PALETTE = {'select_single': '#B16A3D', 'select_subset': '#276C79', 'panel_t005': '#315F87', 'uv+raman+nir': '#8A9DA7'}
SCALE = LinearSegmentedColormap.from_list('spectral_score', ['#F2F6F7', '#C6DCE2', '#7EAAB9', '#356F88', '#163E57'])
COVERAGE = LinearSegmentedColormap.from_list('coverage', ['#A96035', '#EBD1BF', '#FFFFFF', '#C6DDE1', '#2F7182'])

def _read(path):
    return pd.read_csv(path, dtype={'training_size': str}) if path.exists() and path.stat().st_size else pd.DataFrame()

def _style():
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.linewidth': 0.55, 'axes.edgecolor': INK, 'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'axes.unicode_minus': False, 'savefig.facecolor': 'white'})

def _clean(ax, grid=True):
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    if grid:
        ax.grid(axis='y', color='#DFE7EA', linewidth=0.5)
        ax.set_axisbelow(True)

def _algorithms(frame):
    present = {name.split('__', 1)[0] for name in frame.model.unique() if '__' in name}
    return [name for name in ALG if name in present]

def _pick_context(frame):
    result = frame.copy()
    if 'condition' in result:
        condition = 'enhanced' if 'enhanced' in set(result.condition) else result.condition.iloc[0]
        result = result[result.condition.eq(condition)]
    if 'training_size' in result:
        size = 'full' if 'full' in set(result.training_size) else result.training_size.iloc[0]
        result = result[result.training_size.eq(size)]
    return result

def _save(fig, directory, name, assets, layouts):
    fig.canvas.draw()
    from matplotlib.text import Text
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    outside = []
    hidden_ticks = set()
    for ax in fig.axes:
        for axis, limits in [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]:
            lower, upper = sorted(limits)
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if tick.get_loc() < lower - 1e-12 or tick.get_loc() > upper + 1e-12:
                    hidden_ticks.update([tick.label1, tick.label2])
    for item in fig.findobj(Text):
        if item in hidden_ticks or not item.get_visible() or not item.get_text().strip():
            continue
        bounds = item.get_window_extent(renderer)
        if bounds.x0 < -1 or bounds.y0 < -1 or bounds.x1 > width + 1 or bounds.y1 > height + 1:
            outside.append(dict(text=item.get_text(), bounds=list(map(float, bounds.bounds))))
    layouts.append(dict(figure=name, text_outside_canvas=outside))
    for extension in ['pdf', 'svg']:
        fig.savefig(directory / (name + '.' + extension))
        assets.append((directory / (name + '.' + extension)).as_posix())
    path = directory / (name + '.tif')
    fig.savefig(path, dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    with Image.open(path) as image:
        rgb = image.convert('RGB')
    rgb.save(str(path), compression='tiff_lzw', dpi=(600, 600))
    rgb.resize((round(rgb.width / 3), round(rgb.height / 3)), Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS).save(str(directory / (name + '_preview.png')), dpi=(200, 200))
    assets.extend([path.as_posix(), (directory / (name + '_preview.png')).as_posix()])
    plt.close(fig)

def _source(directory, name, frame):
    frame.to_csv(directory / (name + '.csv'), index=False, encoding='utf-8-sig')

def make_figures(out_dir, cfg):
    out = Path(out_dir)
    derived = out / 'derived'
    directory = derived / 'figures'
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / 'Source_data'
    source.mkdir(exist_ok=True)
    targets = cfg.get('target_names', list(TARGETS))
    labels = [ABBR.get(t, t) for t in targets]
    _style()
    assets, captions, layouts, hashes = [], [], [], {}
    common = 'U, ultraviolet-visible; R, Raman; N, near-infrared; concatenated letters are included spectral blocks. SS selects a single modality separately for each target within training, and SUB selects a nonempty subset separately for each target. P0, P2, P5 and P10 select one common mask for the full target panel, using respective training-validation error tolerances of 0%, 2%, 5% and 10% relative to the smallest inner equal-target error. These are computational sensitivity settings, not established analytical noninferiority margins. AcH, acetaldehyde; MeOH, methanol; nPrOH, n-propanol; EA, ethyl acetate; iBuOH, isobutanol; nBuOH, n-butanol; iAmOH, isoamyl alcohol; EB, ethyl butyrate; EL, ethyl lactate; EH*, provisionally identified ethyl hexanoate. All targets are retained in their declared order and have equal weight. Repeated evaluations do not increase independent sample size.'
    metrics_path = derived / 'benchmark' / 'target_metrics_pooled.csv'
    metrics = _read(metrics_path)
    if not metrics.empty:
        hashes[metrics_path.relative_to(out).as_posix()] = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
        metrics = _pick_context(metrics)
        algorithms = _algorithms(metrics)
        columns = list(MASKS) + ['select_single', 'select_subset', 'panel_t005']
        height_mm = min(228, 25 + 49 * len(algorithms))
        fig, axes = plt.subplots(len(algorithms), 1, figsize=(174 / 25.4, height_mm / 25.4), squeeze=False)
        fig.subplots_adjust(left=0.12, right=0.87, top=1 - 10 / height_mm, bottom=max(0.065, 9 / height_mm), hspace=0.40)
        plotted = []
        for i, algorithm in enumerate(algorithms):
            ax = axes[i, 0]
            active = [name for name in columns if algorithm + '__' + name in set(metrics.model)]
            matrix = np.full((len(targets), len(active)), np.nan)
            for y, target in enumerate(targets):
                for x, strategy in enumerate(active):
                    rows = metrics[metrics.model.eq(algorithm + '__' + strategy) & metrics.target.eq(target)]
                    if len(rows) == 1:
                        value = float(rows.r2.iloc[0])
                        matrix[y, x] = value
                        plotted.append(dict(algorithm=algorithm, target=target, strategy=strategy, r2=value))
            art = ax.imshow(matrix, cmap=SCALE, vmin=0, vmax=1, aspect='auto')
            for y in range(len(targets)):
                for x in range(len(active)):
                    value = matrix[y, x]
                    ax.text(x, y, 'NA' if np.isnan(value) else '%.2f' % value, ha='center', va='center', fontsize=6.4, color='white' if value >= 0.65 else INK)
            ax.set_yticks(range(len(targets)))
            ax.set_yticklabels(labels)
            ax.set_xticks(range(len(active)))
            ax.set_xticklabels([SHORT[name] for name in active])
            ax.tick_params(length=0, pad=3)
            ax.set_title(chr(97 + i) + '   ' + ALG[algorithm], loc='left', pad=5, fontweight='bold')
            if len(active) > 7:
                ax.axvline(6.5, color='white', lw=1.2)
            for spine in ax.spines.values():
                spine.set_visible(False)
        bar = fig.colorbar(art, cax=fig.add_axes([0.895, 0.16, 0.018, 0.64]))
        bar.set_label('Pooled outer-test R2', fontsize=7)
        bar.ax.tick_params(labelsize=6.5)
        _source(source, 'Fig1_prediction_map', pd.DataFrame(plotted))
        _save(fig, directory, 'Fig1_prediction_map', assets, layouts)
        captions.append('Fig. 1 Outer-test prediction performance for every target, regression algorithm and principal input strategy. Every cell is the pooled coefficient of determination calculated from the saved held-out prediction and reference vectors; darker blue indicates larger values on a common zero-to-one color scale. Negative values retain their printed number at the color-scale floor, and undefined values are labelled NA. The first seven columns are independently tuned fixed masks. SS and SUB choose inputs separately for each target, whereas P5 uses one shared mask for the complete panel. All preprocessing, hyperparameters and input choices are selected within receiving-training data, and all strategies share the same outer test identities. The displayed set includes the full fixed-mask comparison and the principal selection controls; all additional panel tolerances and original-unit errors are retained in the derived CSV tables. Metrics pool repeated predictions, which are correlated visits from the same bottles rather than additional independent samples. A retrospectively favourable fixed mask or algorithm in this map is an observed ranking and not an independently validated model selected in advance')
    macro_path = derived / 'learning' / 'learning_macro_common_scale_by_repeat.csv'
    common_learning_scale = macro_path.exists()
    if not common_learning_scale:
        macro_path = derived / 'learning' / 'learning_macro_by_repeat.csv'
    budgets_path = derived / 'learning' / 'training_budgets.csv'
    macro, budgets = _read(macro_path), _read(budgets_path)
    if not macro.empty and not budgets.empty and macro.training_size.nunique() > 1:
        algorithms = _algorithms(macro)
        ncols = min(2, len(algorithms))
        nrows = int(math.ceil(len(algorithms) / ncols))
        height_mm = 65 * nrows + 25
        fig, axes = plt.subplots(nrows, ncols, figsize=(174 / 25.4, height_mm / 25.4), squeeze=False)
        fig.subplots_adjust(left=0.11, right=0.97, top=0.93, bottom=max(0.17, 22 / height_mm), wspace=0.30, hspace=0.38)
        sizes = budgets.groupby('training_size').n_train.mean().to_dict()
        plotted = []
        styles = {'select_single': '--', 'select_subset': '-', 'panel_t005': '-.', 'uv+raman+nir': ':'}
        for i, algorithm in enumerate(algorithms):
            ax = axes.flat[i]
            tick_values = set()
            for strategy, color in PALETTE.items():
                group = macro[macro.model.eq(algorithm + '__' + strategy)]
                if group.empty:
                    continue
                point = group.groupby('training_size').macro_standardized_mse.mean().reset_index()
                point['n_train_mean'] = point.training_size.map(sizes)
                point = point.sort_values('n_train_mean')
                tick_values.update(point.n_train_mean)
                ax.plot(point.n_train_mean, point.macro_standardized_mse, color=color, ls=styles[strategy], marker='o', markersize=3, linewidth=1.2)
                for row in point.itertuples():
                    plotted.append(dict(algorithm=algorithm, strategy=strategy, training_size=row.training_size, n_train_mean=row.n_train_mean, mean_macro_standardized_mse=row.macro_standardized_mse))
            ax.set_title(chr(97 + i) + '   ' + ALG[algorithm], loc='left', fontweight='bold')
            ax.set_xlabel('Fitting bottles')
            ax.set_ylabel('Equal-target standardized MSE')
            ticks = sorted(tick_values)
            if ticks:
                ax.set_xticks(ticks)
                ax.set_xticklabels(['%.0f' % value if np.isclose(value, round(value)) else '%.1f' % value for value in ticks])
            _clean(ax)
        for ax in axes.flat[len(algorithms):]:
            ax.set_visible(False)
        handles = [Line2D([0], [0], color=color, ls=styles[strategy], marker='o', markersize=3, lw=1.2, label=SHORT[strategy]) for strategy, color in PALETTE.items()]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.53, 0.02), ncol=4, frameon=False)
        _source(source, 'Fig2_learning_curves', pd.DataFrame(plotted))
        _save(fig, directory, 'Fig2_learning_curves', assets, layouts)
        scale_note = 'Every squared error is divided by its full outer-training target variance, held fixed across the training sizes of that fold. This evaluation scale uses no outer-test labels and is never supplied to model fitting or selection.' if common_learning_scale else 'The available file uses each receiving-training subset target variance, which can change the normalization across sizes; original-unit target RMSE must also be examined.'
        captions.append('Fig. 2 Controlled learning curves for four principal input rules within each learning algorithm. The ordinate is standardized mean squared error, first averaged within each target and then equally across all targets, and averaged over the saved cross-validation repeats. '+scale_note+' Smaller values indicate lower average error. The abscissa is the actual mean number of fitting bottles for the displayed training-size label; a full outer fitting set can differ by one bottle across folds. The same held-out bottles are used across training sizes, and smaller training sets are nested prefixes of the runner-defined training permutation. Every size repeats preprocessing, parameter tuning and input selection using only its available fitting bottles. Full-size predictions are reused from the benchmark rather than treated as a separate independent experiment. Curves show repeat means without confidence shading because repeats share bottles. The CSV source preserves numerical points, while target-specific curves, all fixed masks and repeat-level values remain available in the derived tables. These within-cohort learning curves do not establish performance after a production-batch, day or distillery change')
    macro_path = derived / 'benchmark' / 'macro_standardized_mse.csv'
    acquisition_path = derived / 'acquisition_union_by_task.csv'
    macro, acquisition = _read(macro_path), _read(acquisition_path)
    if not macro.empty and not acquisition.empty:
        macro = _pick_context(macro)
        acquisition = acquisition[acquisition.stage.eq('benchmark')]
        acquisition = _pick_context(acquisition) if not acquisition.empty else acquisition
        counts = acquisition.groupby('model').n_acquisition_modalities.mean().to_dict() if not acquisition.empty else {}
        algorithms = _algorithms(macro)
        ncols = min(2, len(algorithms))
        nrows = int(math.ceil(len(algorithms) / ncols))
        height_mm = 70 * nrows + 20
        fig, axes = plt.subplots(nrows, ncols, figsize=(174 / 25.4, height_mm / 25.4), squeeze=False)
        fig.subplots_adjust(left=0.11, right=0.97, top=0.93, bottom=max(0.18, 22 / height_mm), wspace=0.32, hspace=0.35)
        plotted = []
        average = macro.groupby('model').macro_standardized_mse.mean()
        for i, algorithm in enumerate(algorithms):
            ax = axes.flat[i]
            labels_at = {}
            for model, error in average.items():
                if not model.startswith(algorithm + '__'):
                    continue
                strategy = model.split('__', 1)[1]
                n = len(strategy.split('+')) if strategy in MASKS else counts.get(model)
                if n is None:
                    continue
                style = 'fixed' if strategy in MASKS else 'shared' if strategy.startswith('panel_t') else 'targetwise'
                marker, color, fill = ('o', GRAY, 'none') if style == 'fixed' else ('D', BLUE, BLUE) if style == 'shared' else ('s', ORANGE, ORANGE)
                ax.scatter([n], [error], marker=marker, s=24 if style == 'fixed' else 27, edgecolors=color, facecolors=fill, linewidths=0.8, zorder=3)
                labels_at.setdefault((round(n, 4), round(float(error), 5)), []).append(SHORT.get(strategy, strategy))
                plotted.append(dict(algorithm=algorithm, strategy=strategy, mean_acquisition_modalities=n, mean_macro_standardized_mse=float(error), strategy_scope=style))
            for (x, y), names in labels_at.items():
                ax.annotate('/'.join(names), (x, y), xytext=(-4 if x >= 2.7 else 4, 4), textcoords='offset points', fontsize=5.7, ha='right' if x >= 2.7 else 'left')
            ax.set_xlim(0.85, 3.48)
            ax.set_xticks([1, 2, 3])
            ax.set_title(chr(97 + i) + '   ' + ALG[algorithm], loc='left', fontweight='bold')
            ax.set_xlabel('Mean whole-panel modalities')
            ax.set_ylabel('Equal-target standardized MSE')
            ax.margins(y=0.18)
            _clean(ax)
        for ax in axes.flat[len(algorithms):]:
            ax.set_visible(False)
        fig.legend(handles=[Line2D([0], [0], marker='o', color=GRAY, markerfacecolor='none', ls='', label='Fixed mask'), Line2D([0], [0], marker='s', color=ORANGE, ls='', label='Target-wise selection'), Line2D([0], [0], marker='D', color=BLUE, ls='', label='Shared-panel selection')], loc='lower center', bbox_to_anchor=(0.53, 0.01), ncol=3, frameon=False)
        _source(source, 'Fig3_acquisition_tradeoff', pd.DataFrame(plotted))
        _save(fig, directory, 'Fig3_acquisition_tradeoff', assets, layouts)
        captions.append('Fig. 3 Observed prediction-error and measurement-number trade-offs when the complete target panel must be reported. Each point represents one prediction strategy within one learning algorithm. The ordinate is the benchmark repeat-mean equal-target standardized mean squared error, and the abscissa is the mean number of modalities that must actually be acquired for the entire panel in each outer fitted task. Fixed masks have an integer acquisition count. For target-wise selectors, the count is the union of the inputs selected for every target, not their average subset size. Shared-panel selectors choose one common mask, although that mask can change across fitted tasks. Filled diamonds represent the four declared panel tolerances; coincident strategy labels are combined. Open circles are fixed masks and orange squares target-wise selectors. Counts are measurement-number proxies and are not monetary costs or acquisition times. The displayed trade-off is retrospective, and no point is claimed to be clinically or analytically equivalent solely because of a small macro difference. Every target-specific error and actual task-level acquisition union remains in the CSV outputs')
    joint_path = derived / 'reliability' / 'joint_coverage_width_score.csv'
    joint = _read(joint_path)
    if not joint.empty:
        joint = _pick_context(joint)
        algorithms = _algorithms(joint)
        anchors = ['uv+raman+nir', 'select_single', 'select_subset', 'panel_t005']
        method_order = ['constant_marginal', 'adaptive_marginal', 'constant_joint', 'adaptive_joint']
        colors = ['#C6D7E1', '#79A5B4', '#5B7C96', '#214E6A']
        height_mm = 48 * len(algorithms) + 30
        fig, axes = plt.subplots(len(algorithms), 2, figsize=(174 / 25.4, height_mm / 25.4), squeeze=False)
        fig.subplots_adjust(left=0.11, right=0.98, top=1 - 10 / height_mm, bottom=max(0.10, 25 / height_mm), wspace=0.30, hspace=0.38)
        used = []
        all_width = joint.mean_standardized_width.replace([np.inf, -np.inf], np.nan).dropna()
        max_width = max(1, float(all_width.max()) * 1.08) if len(all_width) else 1
        for i, algorithm in enumerate(algorithms):
            active = [a for a in anchors if algorithm + '__' + a in set(joint.model)]
            for j, endpoint in enumerate(['joint_coverage', 'mean_standardized_width']):
                ax = axes[i, j]
                y = np.arange(len(active))
                for k, method in enumerate(method_order):
                    values = []
                    for name in active:
                        rows = joint[joint.model.eq(algorithm + '__' + name) & joint.interval_method.eq(method)]
                        value = float(rows[endpoint].iloc[0]) if len(rows) == 1 else np.nan
                        values.append(100 * value if j == 0 else value)
                        if j == 0 and len(rows) == 1:
                            used.append(rows.iloc[0].to_dict())
                    positions = y + (k - 1.5) * 0.18
                    ax.barh(positions, [value if np.isfinite(value) else np.nan for value in values], height=0.16, color=colors[k], hatch='//' if method.startswith('adaptive') else None, linewidth=0, zorder=3)
                    for position, value in zip(positions, values):
                        if np.isinf(value):
                            ax.text(0.96 * (100 if j == 0 else max_width), position, 'Inf', ha='right', va='center', color=colors[k], fontsize=5.5)
                ax.set_yticks(y)
                ax.set_yticklabels([SHORT[name] for name in active])
                ax.set_ylim(len(active) - 0.45, -0.55)
                ax.set_title(chr(97 + 2 * i + j) + '   ' + ALG[algorithm] + (' | coverage' if j == 0 else ' | width'), loc='left', fontweight='bold')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(axis='x', color='#DFE7EA', linewidth=0.5)
                ax.set_axisbelow(True)
                if j == 0:
                    ax.set_xlim(0, 100)
                    ax.set_xticks([0, 30, 60, 90])
                    ax.axvline(100 * (1 - float(cfg.get('interval_alpha', 0.1))), color=ORANGE, lw=0.8, ls='--')
                else:
                    ax.set_xlim(0, max_width)
                if i == len(algorithms) - 1:
                    ax.set_xlabel('All targets covered (%)' if j == 0 else 'Mean standardized width')
        fig.legend(handles=[Patch(facecolor=colors[k], hatch='//' if method.startswith('adaptive') else None, label=method.replace('_', ' ').capitalize()) for k, method in enumerate(method_order)], loc='lower center', bbox_to_anchor=(0.54, 0.01), ncol=2, frameon=False)
        _source(source, 'Fig4_joint_reliability', pd.DataFrame(used))
        _save(fig, directory, 'Fig4_joint_reliability', assets, layouts)
        captions.append('Fig. 4 Simultaneous target-vector coverage and standardized interval-width cost for the principal input rules. Rows correspond to learning algorithms; the left column shows the percentage of test visits at which every declared target is covered, and the right column gives full interval width divided by the corresponding fitting-set target standard deviation, averaged equally over targets and visits. Four parallel bars compare constant marginal, adaptive marginal, constant joint and adaptive joint calibration, all using identical point predictions and fitting/calibration/test identities. Hatching identifies sample-adaptive scale estimates trained exclusively from proper-training data and OOF residuals. The dashed line is the nominal joint level for the joint calibration constructions under exchangeability; it is not a simultaneous guarantee for the collection of separately calibrated marginal intervals. The display includes URN, SS, SUB and P5 to keep the mechanism comparison legible, and complete fixed-mask and panel-tolerance results are retained in the CSV tables. Higher observed coverage is not sufficient evidence of a better method without examining width and interval score. No laboratory specification tolerance is inferred from standardized widths')
    target_path = derived / 'reliability' / 'coverage_width_score_by_target.csv'
    coverage = _read(target_path)
    if not coverage.empty:
        coverage = _pick_context(coverage)
        algorithms = _algorithms(coverage)
        height_mm = min(228, 25 + 49 * len(algorithms))
        fig, axes = plt.subplots(len(algorithms), 1, figsize=(174 / 25.4, height_mm / 25.4), squeeze=False)
        fig.subplots_adjust(left=0.12, right=0.87, top=1 - 10 / height_mm, bottom=max(0.075, 9 / height_mm), hspace=0.40)
        columns = [('all', 'constant_marginal', 'All: constant'), ('all', 'adaptive_marginal', 'All: adaptive'), ('high_predicted_concentration', 'constant_marginal', 'Pred-high: constant'), ('high_predicted_concentration', 'adaptive_marginal', 'Pred-high: adaptive')]
        plotted = []
        nominal = 100 * (1 - float(cfg.get('interval_alpha', 0.1)))
        for i, algorithm in enumerate(algorithms):
            ax = axes[i, 0]
            model = algorithm + '__select_subset'
            matrix = np.full((len(targets), len(columns)), np.nan)
            for y, target in enumerate(targets):
                for x, (stratum, method, _) in enumerate(columns):
                    row = coverage[coverage.model.eq(model) & coverage.target.eq(target) & coverage.stratum.eq(stratum) & coverage.interval_method.eq(method)]
                    if len(row) == 1:
                        value = 100 * float(row.coverage.iloc[0])
                        matrix[y, x] = value
                        plotted.append(dict(algorithm=algorithm, target=target, stratum=stratum, interval_method=method, coverage_pct=value, n_visits=int(row.n_visits.iloc[0]), n_unique_bottles=int(row.n_unique_bottles.iloc[0])))
            art = ax.imshow(matrix, cmap=COVERAGE, norm=TwoSlopeNorm(vmin=max(0, nominal - 40), vcenter=nominal, vmax=100), aspect='auto')
            for y in range(len(targets)):
                for x in range(len(columns)):
                    value = matrix[y, x]
                    ax.text(x, y, 'NA' if np.isnan(value) else '%.1f' % value, ha='center', va='center', fontsize=7, color='white' if value < 60 or value > 97 else INK)
            ax.set_yticks(range(len(targets)))
            ax.set_yticklabels(labels)
            ax.set_xticks(range(len(columns)))
            ax.set_xticklabels([c[2] for c in columns], fontsize=6.5)
            ax.set_title(chr(97 + i) + '   ' + ALG[algorithm] + ' | SUB', loc='left', fontweight='bold', pad=5)
            ax.tick_params(length=0, pad=3)
            ax.axvline(1.5, color='white', lw=1.3)
            for spine in ax.spines.values():
                spine.set_visible(False)
        bar = fig.colorbar(art, cax=fig.add_axes([0.895, 0.16, 0.018, 0.64]))
        bar.set_label('Observed individual-target coverage (%)', fontsize=7)
        bar.ax.tick_params(labelsize=6.5)
        _source(source, 'Fig5_target_coverage', pd.DataFrame(plotted))
        _save(fig, directory, 'Fig5_target_coverage', assets, layouts)
        captions.append('Fig. 5 Individual-target coverage of constant and sample-adaptive marginal intervals for every target under the per-target subset strategy. The first two columns contain all-test coverage, and the last two contain coverage in a high-predicted-concentration stratum defined using a proper-training OOF prediction threshold. This threshold can be applied without knowing future reference concentrations. The printed numbers are empirical percentages; color is centered on the nominal %.1f%% marginal level, with low values toward orange and larger values toward blue-green. Values outside the color-scale limits retain their exact printed number, and NA means that the saved diagnostic stratum contains no visits rather than zero coverage. Each algorithm panel retains the same complete target order. Source tables give both repeated visit counts and distinct bottle counts, and contain the complementary true-high-concentration diagnostic and all other models and interval constructions. Sample-adaptive widths and calibration do not establish universal concentration-conditional coverage; coverage and interval efficiency must be interpreted together. This reliability experiment has a separate fitting and calibration budget from the benchmark, so its accuracy cannot be compared as a same-budget algorithm gain' % nominal)
    (directory / 'captions.md').write_text('# Figure captions\n\nShared notation. ' + common + '\n\n' + '\n\n'.join(captions) + '\n', encoding='utf-8')
    (directory / 'layout_checks.json').write_text(json.dumps(layouts, ensure_ascii=True, indent=2), encoding='utf-8')
    learning_source = derived/'learning'/('learning_macro_common_scale_by_repeat.csv' if common_learning_scale else 'learning_macro_by_repeat.csv')
    for input_path in [learning_source, budgets_path, acquisition_path, joint_path, target_path, derived/'benchmark'/'macro_standardized_mse.csv']:
        if input_path.exists():
            hashes[input_path.relative_to(out).as_posix()] = hashlib.sha256(input_path.read_bytes()).hexdigest()
    provenance = dict(models_refitted=False, target_order=targets, source_sha256=hashes, formats=['600 dpi RGB TIFF', 'vector PDF', 'editable-text SVG', '200 dpi PNG preview'], figures_created=[row['figure'] for row in layouts], no_results_fabricated=True, learning_plot_uses_common_outer_training_scale=common_learning_scale)
    (directory / 'figure_provenance.json').write_text(json.dumps(provenance, ensure_ascii=True, indent=2), encoding='utf-8')
    return [str(Path(path).relative_to(out).as_posix()) for path in assets]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--config', default='configs/full.json')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    print(json.dumps(make_figures(args.out, config), indent=2))
