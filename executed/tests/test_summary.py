from pathlib import Path
import json
import gzip
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from summarize import acquisition_and_parameters, control_effects, interval_metrics, paired_effects, stage_metrics, summarize, validate_predictions, workflow_effects

TARGET_ORDER = ['acetaldehyde', 'methanol']
STRATEGIES = ['uv', 'raman', 'nir', 'uv+raman', 'uv+nir', 'raman+nir', 'uv+raman+nir', 'select_single', 'select_subset', 'panel_t005']

def points(stage='benchmark', condition='enhanced', size='full', scale=1.0, n_train=24):
    records = []
    models = ['mean'] + ['pls__' + strategy for strategy in STRATEGIES]
    for repeat in range(2):
        for i in range(16):
            for j, target in enumerate(TARGET_ORDER):
                truth = 2 + j + i / 8
                for k, model in enumerate(models):
                    error = scale * (0.1 + 0.025 * k) * (1 + j) * (0.8 + 0.3 * np.sin(i + repeat))
                    records.append(dict(stage=stage, condition=condition, training_size=str(size), model=model, repeat=repeat, fold=i // 8, bootstrap_refit=-1, sample_id='synthetic_%02d' % i, target=target, y_true=truth, y_pred=truth + error, train_sd=1 + j, outer_train_sd=1+j, n_train=n_train, n_unique_train=n_train, n_cal=10 if stage == 'reliability' else 0))
    return pd.DataFrame(records)

def intervals(point):
    records = []
    for row in point.to_dict('records'):
        for k, method in enumerate(['constant_marginal', 'constant_joint', 'adaptive_marginal', 'adaptive_joint']):
            half = (0.25 + 0.2 * k) * row['train_sd']
            lower = row['y_pred'] - half
            upper = row['y_pred'] + half
            covered = lower <= row['y_true'] <= upper
            score = upper - lower + 20 * max(lower - row['y_true'], row['y_true'] - upper, 0)
            number = int(row['sample_id'].rsplit('_', 1)[1])
            records.append(dict(row, interval_method=method, lower=lower, upper=upper, width=upper - lower, covered=covered, fit_sd=row['train_sd'], n_fit=row['n_train'], n_cal=10, rank=10, high_true_concentration=number > 12, high_predicted_concentration=number > 11, interval_score=score))
    return pd.DataFrame(records)

def fixture(out, figures=False):
    out = Path(out)
    for stage, frame in [('benchmark', points()), ('learning', points('learning', size='8', scale=1.4, n_train=8)), ('reliability', points('reliability', n_train=16))]:
        folder = out / stage
        folder.mkdir(parents=True, exist_ok=True)
        frame.to_csv(folder / 'predictions.csv', index=False)
        if stage == 'reliability':
            intervals(frame).to_csv(folder / 'intervals.csv', index=False)
    tasks = out / 'tasks'
    tasks.mkdir(exist_ok=True)
    choices = {}
    for strategy in STRATEGIES:
        masks = ['uv', 'raman+nir'] if strategy == 'select_subset' else ['uv', 'raman'] if strategy == 'select_single' else ['uv+raman', 'uv+raman'] if strategy.startswith('panel_t') else [strategy, strategy]
        choices['pls__' + strategy] = [dict(target_index=j, mask=mask, spec=dict(algorithm='pls', mask=mask, components=2)) for j, mask in enumerate(masks)]
    diagnostic = dict(selection_by_model=choices, candidate_scores=[dict(spec=dict(algorithm='pls', components=k)) for k in [2, 5]])
    payload = dict(stage='benchmark', condition='enhanced', training_size='full', repeat=0, fold=0, bootstrap_refit=-1, diagnostics=diagnostic)
    (tasks / 'benchmark_fixture.json').write_text(json.dumps(dict(payload=payload)), encoding='utf-8')
    (out / 'COMPLETENESS.json').write_text(json.dumps(dict(status='incomplete', stages={stage: dict(complete=False) for stage in ['benchmark', 'learning', 'reliability']})), encoding='utf-8')
    cfg = dict(target_names=TARGET_ORDER, summary_bootstrap_draws=20, summary_make_figures=figures, interval_alpha=0.1, seed=5)
    return cfg

class SummaryChecks(unittest.TestCase):
    def test_target_weighting_uses_standardized_errors(self):
        data = points()
        data.loc[data.target.eq(TARGET_ORDER[0]), 'y_pred'] = data.loc[data.target.eq(TARGET_ORDER[0]), 'y_true'] + 1
        data.loc[data.target.eq(TARGET_ORDER[1]), 'y_pred'] = data.loc[data.target.eq(TARGET_ORDER[1]), 'y_true'] + 10
        data.loc[data.target.eq(TARGET_ORDER[1]), 'train_sd'] = 10
        _, _, macro = stage_metrics(data)
        np.testing.assert_allclose(macro.macro_standardized_mse, 1)

    def test_unequal_target_or_bottle_coverage_rejected(self):
        data = points()
        self.assertTrue(validate_predictions(data, TARGET_ORDER))
        with self.assertRaises(ValueError):
            validate_predictions(data[data.target.eq(TARGET_ORDER[0])], TARGET_ORDER)
        with self.assertRaises(ValueError):
            validate_predictions(data.drop(data.index[0]), TARGET_ORDER)

    def test_unmatched_model_budget_rejected(self):
        data = points()
        data.loc[data.model.eq('mean'), 'n_train'] = 25
        with self.assertRaises(ValueError):
            validate_predictions(data, TARGET_ORDER)

    def test_truth_disagreement_rejected(self):
        data = points()
        data.loc[0, 'y_true'] += 1
        with self.assertRaises(ValueError):
            validate_predictions(data, TARGET_ORDER)

    def test_fixed_prediction_intervals_are_reproducible(self):
        first, edges = paired_effects(points(), 50, 7)
        second, _ = paired_effects(points(), 50, 7)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(edges), 9 * len(TARGET_ORDER))
        self.assertTrue(first.ci_scope.str.contains('excludes refitting').all())
        self.assertTrue((first.conditional_ci_low_pct <= first.gain_pct + 1e-8).all())

    def test_whole_panel_count_uses_union(self):
        with tempfile.TemporaryDirectory(prefix='_synthetic_summary_', dir=str(ROOT)) as td:
            fixture(td)
            acquisition, _, parameters = acquisition_and_parameters(Path(td), TARGET_ORDER)
            subset = acquisition[acquisition.model.eq('pls__select_subset')].iloc[0]
            self.assertEqual(subset.n_acquisition_modalities, 3)
            self.assertEqual(subset.mean_per_target_modalities, 1.5)
            panel = acquisition[acquisition.model.eq('pls__panel_t005')].iloc[0]
            self.assertEqual(panel.n_acquisition_modalities, 2)
            self.assertEqual(panel.acquisition_union, 'uv+raman')
            self.assertTrue(parameters.at_min.all())
            self.assertFalse(parameters.at_max.any())

    def test_panel_with_different_target_masks_rejected(self):
        with tempfile.TemporaryDirectory(prefix='_synthetic_summary_', dir=str(ROOT)) as td:
            fixture(td)
            path = Path(td) / 'tasks' / 'benchmark_fixture.json'
            task = json.loads(path.read_text(encoding='utf-8'))
            task['payload']['diagnostics']['selection_by_model']['pls__panel_t005'][0]['mask'] = 'uv'
            path.write_text(json.dumps(task), encoding='utf-8')
            with self.assertRaises(ValueError):
                acquisition_and_parameters(Path(td), TARGET_ORDER)

    def test_compressed_task_diagnostics_are_read(self):
        with tempfile.TemporaryDirectory(prefix='_synthetic_summary_', dir=str(ROOT)) as td:
            fixture(td)
            path = Path(td) / 'tasks' / 'benchmark_fixture.json'
            with gzip.open(str(path)+'.gz', 'wb') as handle:
                handle.write(path.read_bytes())
            path.unlink()
            acquisition, _, _ = acquisition_and_parameters(Path(td), TARGET_ORDER)
            self.assertIn('pls__panel_t005', set(acquisition.model))

    def test_independent_modality_treatments_survive_summary(self):
        with tempfile.TemporaryDirectory(prefix='_synthetic_summary_', dir=str(ROOT)) as td:
            cfg=fixture(td)
            path=Path(td)/'tasks'/'benchmark_fixture.json'
            task=json.loads(path.read_text(encoding='utf-8'))
            chosen=task['payload']['diagnostics']['selection_by_model']['pls__select_subset'][1]
            chosen['spec']['preprocessing']={'raman':'raw','nir':'snv'}
            path.write_text(json.dumps(task),encoding='utf-8')
            summarize(td,cfg)
            record=pd.read_csv(Path(td)/'derived'/'selected_preprocessing_records.csv')
            row=record[record.model.eq('pls__select_subset') & record.target.eq(TARGET_ORDER[1])].iloc[0]
            self.assertEqual(row.raman_method,'raw')
            self.assertEqual(row.nir_method,'snv')

    def test_intervals_validate_endpoints_and_empty_strata(self):
        point = points('reliability', n_train=16)
        interval = intervals(point)
        interval['high_predicted_concentration'] = False
        per_target, joint = interval_metrics(interval, point, TARGET_ORDER, 0.1)
        empty = per_target[per_target.stratum.eq('high_predicted_concentration')]
        self.assertTrue(empty.n_visits.eq(0).all())
        self.assertTrue(empty.coverage.isna().all())
        self.assertEqual(len(joint), len(point.model.unique()) * 4)
        broken = interval.copy()
        broken.loc[0, 'covered'] = not bool(broken.loc[0, 'covered'])
        with self.assertRaises(ValueError):
            interval_metrics(broken, point, TARGET_ORDER, 0.1)
        broken = interval.copy()
        broken.loc[0, 'interval_score'] += 1
        with self.assertRaises(ValueError):
            interval_metrics(broken, point, TARGET_ORDER, 0.1)

    def test_joint_coverage_is_not_mean_marginal_coverage(self):
        point = points('reliability', n_train=16)
        point = point[point.model.eq('mean') & point['repeat'].eq(0) & point.sample_id.isin(['synthetic_00', 'synthetic_01'])].copy()
        number = point.sample_id.str.rsplit('_', n=1).str[-1].astype(int)
        miss = (number.eq(0) & point.target.eq(TARGET_ORDER[0])) | (number.eq(1) & point.target.eq(TARGET_ORDER[1]))
        point['y_pred'] = point.y_true + np.where(miss, 4, 0)
        interval = intervals(point)
        per_target, joint = interval_metrics(interval, point, TARGET_ORDER, 0.1)
        self.assertTrue(joint.joint_coverage.eq(0).all())
        self.assertTrue(per_target[per_target.stratum.eq('all')].coverage.eq(0.5).all())

    def test_workflow_and_control_comparisons_are_matched(self):
        base = points()
        legacy = points('preprocessing', 'legacy', scale=1.5)
        independent = points('preprocessing', 'independent_snv', scale=1.25)
        expanded = points('preprocessing', 'expanded_grid', scale=1.1)
        contrast = workflow_effects(dict(benchmark=base, preprocessing=pd.concat([legacy, independent, expanded], ignore_index=True)))
        self.assertEqual(len(contrast), len(base.model.unique()) * len(TARGET_ORDER) * 3)
        self.assertTrue(contrast.gain_pct.gt(0).all())
        shuffled = points('controls', 'shuffled_nir', scale=1.4)
        shuffled['bootstrap_refit'] = 0
        control = control_effects(dict(benchmark=base, controls=shuffled))
        self.assertTrue(control.gain_real_over_control_pct.gt(0).all())

    def test_summary_preserves_all_targets_and_partial_status(self):
        with tempfile.TemporaryDirectory(prefix='_synthetic_summary_', dir=str(ROOT)) as td:
            cfg = fixture(td)
            result = summarize(td, cfg)
            self.assertEqual(set(result['available_stages']), {'benchmark', 'learning', 'reliability'})
            self.assertEqual(result['execution_status'], 'incomplete')
            metrics = pd.read_csv(Path(td) / 'derived' / 'benchmark' / 'target_metrics_pooled.csv')
            self.assertEqual(set(metrics.target), set(TARGET_ORDER))
            learning = pd.read_csv(Path(td) / 'derived' / 'learning' / 'learning_macro_by_repeat.csv')
            self.assertEqual(set(learning.training_size), {'8', 'full'})
            report = (Path(td) / 'derived' / 'report.md').read_text(encoding='utf-8')
            self.assertIn('partial', report)
            self.assertIn('not a whole-panel instrument saving', report)
            self.assertIn('does not establish heteroscedasticity', report)

    def test_no_predictions_does_not_fabricate_results(self):
        with tempfile.TemporaryDirectory(prefix='_synthetic_summary_', dir=str(ROOT)) as td:
            result = summarize(td, dict(target_names=TARGET_ORDER, summary_make_figures=False))
            self.assertEqual(result['available_stages'], [])
            self.assertIn('not experimental evidence', (Path(td) / 'derived' / 'report.md').read_text(encoding='utf-8'))

    def test_learning_curves_use_one_outer_scale_across_sizes(self):
        with tempfile.TemporaryDirectory(prefix='_synthetic_summary_', dir=str(ROOT)) as td:
            cfg=fixture(td)
            full=points()
            small=full.copy()
            small['stage']='learning'
            small['training_size']='8'
            small['n_train']=8
            small['n_unique_train']=8
            small['train_sd']=10*small.train_sd
            small.to_csv(Path(td)/'learning'/'predictions.csv',index=False)
            summarize(td,cfg)
            folder=Path(td)/'derived'/'learning'
            ordinary=pd.read_csv(folder/'learning_macro_by_repeat.csv',dtype={'training_size':str})
            common=pd.read_csv(folder/'learning_macro_common_scale_by_repeat.csv',dtype={'training_size':str})
            a=ordinary[ordinary.training_size.eq('8')].set_index(['model','repeat']).macro_standardized_mse
            b=ordinary[ordinary.training_size.eq('full')].set_index(['model','repeat']).macro_standardized_mse
            np.testing.assert_allclose(a,b/100)
            a=common[common.training_size.eq('8')].set_index(['model','repeat']).macro_standardized_mse
            b=common[common.training_size.eq('full')].set_index(['model','repeat']).macro_standardized_mse
            np.testing.assert_allclose(a,b)

if __name__ == '__main__':
    unittest.main()
