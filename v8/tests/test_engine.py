from pathlib import Path
import copy
import hashlib
import inspect
import json
import sys
import unittest
from unittest import mock
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from v8 import engine


def fixture(algorithms=None, n=24):
    random = np.random.RandomState(731)
    z = random.normal(size=(n + 5, 3))
    X, E, axes = {}, {}, {}
    for modality, features in [('uv', 7), ('raman', 9), ('nir', 15)]:
        loading = random.normal(scale=0.15, size=(3, features))
        signal = np.linspace(0.1, 2.0, features)[None, :] + z.dot(loading) + random.normal(scale=0.01, size=(n + 5, features))
        X[modality], E[modality] = signal[:n], signal[n:]
        axes[modality] = np.linspace(4000, 12000, features) if modality == 'nir' else np.arange(features, dtype=float)
    Y = z[:n].dot(random.normal(size=(3, 11))) + random.normal(scale=0.03, size=(n, 11))
    ids = np.array(['s%03d' % i for i in range(n)])
    cfg = dict(profile='quick', inner_folds=2, algorithms=algorithms or ['pls', 'ridge', 'krr', 'sopls'], preprocessing_options=dict(uv=['raw'], raman=['raw'], nir=['raw']), sopls_preprocessing_options=dict(uv=['raw'], raman=['raw'], nir=['raw']), preprocessing_settings=dict(sg_window=5, sg_polyorder=2), grid=dict(pls_components=[1, 2], ridge_alphas=[0.01, 0.1], krr_alphas=[0.01, 0.1], krr_gammas=[0.1, 1], sopls_components=[1]), panel_tolerances=[0, 0.05], v8=dict(nir_weights=[0, 0.25, 1, 4], nir_interval_regions=3, balanced_seed_offset=1700000))
    task = dict(condition='nir_extended', repeat=0, fold=0, seed=314, budget_refit=0)
    return X, Y, ids, E, axes, cfg, task


def archived_fixture():
    X, Y, ids, E, axes, cfg, task = fixture(['krr'])
    cfg['preprocessing_options']['nir'] = ['raw', 'snv']
    prediction, diagnostics = engine.frozen_engine.select_and_predict(X, Y, ids, E, cfg, task['seed'], axes, return_oof=False)
    payload = dict(stage='benchmark', condition='enhanced', training_size='full', repeat=0, fold=0, fit_ids=ids.tolist(), diagnostics=diagnostics)
    wrapper = dict(signature='synthetic', payload=payload, payload_sha256=engine._payload_hash(payload))
    return X, Y, ids, E, axes, cfg, task, wrapper


class EngineTests(unittest.TestCase):
    def test_msc_uses_only_fitting_reference_and_applies_same_reference(self):
        reference = np.array([1., 2., 4., 8.])
        training = np.vstack([reference, 2 * reference + 3, 0.5 * reference - 2])
        evaluation = np.vstack([3 * reference + 10, 0.75 * reference + 4])
        at, bt, metadata = engine._msc_pair(training, evaluation)
        mean = training.mean(axis=0)
        np.testing.assert_allclose(at, np.tile(mean, (3, 1)), atol=1e-12)
        np.testing.assert_allclose(bt, np.tile(mean, (2, 1)), atol=1e-12)
        changed_a, changed_b, changed_meta = engine._msc_pair(training, evaluation + 100000)
        np.testing.assert_array_equal(at, changed_a)
        self.assertEqual(metadata['msc_reference_sha256'], changed_meta['msc_reference_sha256'])
        self.assertEqual(metadata['msc_reference_sha256'], engine.frozen_preprocessing.digest(mean))
        self.assertEqual(metadata['n_reference_rows'], 3)

    def test_msc_derivative_preserves_shape_and_training_reference(self):
        X, Y, ids, E, axes, cfg, task = fixture()
        a, b, metadata = engine._processed_pair(X['nir'], E['nir'], 'msc_sg1', axes['nir'], cfg['preprocessing_settings'])
        self.assertEqual(a.shape, X['nir'].shape)
        self.assertEqual(b.shape, E['nir'].shape)
        self.assertEqual(metadata['method'], 'msc_sg1')
        self.assertEqual(metadata['msc_reference_sha256'], engine.frozen_preprocessing.digest(X['nir'].mean(axis=0)))
        self.assertTrue(np.isfinite(a).all() and np.isfinite(b).all())

    def test_intervals_partition_by_coordinate_not_point_count(self):
        axis = np.array([0., 0.1, 0.2, 0.3, 1.1, 1.2, 1.9, 2.2, 2.5, 2.9, 3.0])
        intervals = engine._interval_indices(axis, 3)
        np.testing.assert_array_equal(intervals['region_1'], [0, 1, 2, 3])
        np.testing.assert_array_equal(intervals['region_2'], [4, 5, 6])
        np.testing.assert_array_equal(intervals['region_3'], [7, 8, 9, 10])
        np.testing.assert_array_equal(np.sort(np.concatenate([intervals['region_' + str(i)] for i in (1, 2, 3)])), np.arange(len(axis)))

    def test_extended_four_families_finite_and_fold_local_msc(self):
        X, Y, ids, E, axes, cfg, task = fixture()
        result = engine.evaluate_task(X, Y, ids, E, axes, cfg, task)
        self.assertEqual(len(result['predictions']), 8)
        for values in result['predictions'].values():
            self.assertEqual(values.shape, (5, 11))
            self.assertTrue(np.isfinite(values).all())
        records = result['diagnostics']['preprocessing_fits']
        seen = 0
        for record in records:
            metadata = record['preprocessing'].get('nir', {})
            if 'msc_reference_sha256' in metadata:
                rows = result['diagnostics']['inner_splits'][record['fold']]['train_row_indices']
                expected = engine.frozen_preprocessing.digest(X['nir'][rows].mean(axis=0))
                self.assertEqual(metadata['msc_reference_sha256'], expected)
                self.assertEqual(metadata['n_reference_rows'], len(rows))
                seen += 1
        self.assertGreater(seen, 0)
        json.dumps(result['diagnostics'], allow_nan=False)

    def test_evaluation_outliers_do_not_change_training_selection(self):
        X, Y, ids, E, axes, cfg, task = fixture(['ridge'])
        first = engine.evaluate_task(X, Y, ids, E, axes, cfg, task)
        altered = {m: E[m] + 1000 for m in E}
        second = engine.evaluate_task(X, Y, ids, altered, axes, cfg, task)
        self.assertEqual(first['diagnostics']['selections'], second['diagnostics']['selections'])
        self.assertEqual(first['diagnostics']['preprocessing_fits'], second['diagnostics']['preprocessing_fits'])
        self.assertEqual(first['diagnostics']['training_data_sha256'], second['diagnostics']['training_data_sha256'])
        self.assertNotIn('Y_eval', inspect.signature(engine.evaluate_task).parameters)
        self.assertNotIn('y_eval', inspect.signature(engine.evaluate_task).parameters)

    def test_zero_nir_weight_matches_frozen_ur_class_exactly(self):
        X, Y, ids, E, axes, cfg, task = fixture(['krr'])
        task['condition'] = 'nir_weight'
        result = engine.evaluate_task(X, Y, ids, E, axes, cfg, task)
        frozen, diagnostic = engine.frozen_engine.select_and_predict(X, Y, ids, E, cfg, task['seed'], axes, return_oof=False)
        np.testing.assert_array_equal(result['predictions']['krr__weight_zero'], frozen['krr__uv+raman'])
        self.assertEqual(result['diagnostics']['acquisition_by_model']['krr__weight_zero']['n_modalities'], 2)
        for record in result['diagnostics']['selections']['krr__weight_zero']:
            self.assertEqual(record['spec']['nir_weight'], 0)
            self.assertNotIn('nir', record['spec']['modalities'])
            self.assertNotIn('nir', record['spec']['preprocessing'])
        altered = dict(E, nir=E['nir'] * 10000 + 99)
        again = engine.evaluate_task(X, Y, ids, altered, axes, cfg, task)
        np.testing.assert_array_equal(result['predictions']['krr__weight_zero'], again['predictions']['krr__weight_zero'])

    def test_common_weight_is_shared_by_every_constituent(self):
        X, Y, ids, E, axes, cfg, task = fixture(['krr'])
        task['condition'] = 'nir_weight_learning'
        result = engine.evaluate_task(X, Y, ids, E, axes, cfg, task)
        weight = result['diagnostics']['shared_weight']['common_nir_weight']
        records = result['diagnostics']['selections']['krr__weight_selected']
        self.assertEqual({r['spec']['nir_weight'] for r in records}, {weight})
        expected = 2 if weight == 0 else 3
        self.assertEqual(result['diagnostics']['acquisition_by_model']['krr__weight_selected']['n_modalities'], expected)

    def test_only_zero_weight_has_exact_two_modality_output(self):
        X, Y, ids, E, axes, cfg, task = fixture(['krr'])
        cfg['v8']['nir_weights'] = [0]
        task['condition'] = 'nir_weight_stability'
        result = engine.evaluate_task(X, Y, ids, E, axes, cfg, task)
        np.testing.assert_array_equal(result['predictions']['krr__weight_zero'], result['predictions']['krr__weight_selected'])
        self.assertEqual(result['diagnostics']['shared_weight']['common_nir_weight'], 0)
        self.assertEqual(result['diagnostics']['acquisition_by_model']['krr__weight_selected']['modality_union'], ['uv', 'raman'])

    def test_invalid_positive_weight_classes_are_excluded_with_valid_zero_baseline(self):
        X, Y, ids, E, axes, cfg, task = fixture(['krr'])
        task['condition'] = 'nir_weight'
        original = engine._make_design
        def reject_nir(xt, xe, coordinates, spec, settings, cache=None):
            if 'nir' in spec['modalities']:
                raise ValueError('Synthetic positive-weight fit failure')
            return original(xt, xe, coordinates, spec, settings, cache)
        with mock.patch.object(engine, '_make_design', side_effect=reject_nir):
            result = engine.evaluate_task(X, Y, ids, E, axes, cfg, task)
        self.assertEqual(result['diagnostics']['shared_weight']['common_nir_weight'], 0)
        self.assertEqual(len(result['diagnostics']['shared_weight']['excluded_weights']), 3)
        np.testing.assert_array_equal(result['predictions']['krr__weight_zero'], result['predictions']['krr__weight_selected'])

    def test_unscaled_centers_without_block_rms_division(self):
        X, Y, ids, E, axes, cfg, task = fixture(['ridge'])
        spec = engine._candidates(cfg, 'unscaled')[0]
        design = engine._make_design(X, E, axes, spec, cfg)
        np.testing.assert_allclose(design['train_blocks']['uv'], X['uv'] - X['uv'].mean(axis=0))
        np.testing.assert_allclose(design['evaluation_blocks']['uv'], E['uv'] - X['uv'].mean(axis=0))
        self.assertEqual(design['preprocessing']['uv']['block_rms'], 1)
        task['condition'] = 'unscaled'
        result = engine.evaluate_task(X, Y, ids, E, axes, cfg, task)
        self.assertEqual(set(result['predictions']), {'ridge__uv+raman', 'ridge__uv+raman+nir'})

    def test_interval_candidates_select_inside_training(self):
        X, Y, ids, E, axes, cfg, task = fixture(['ridge'])
        task['condition'] = 'nir_intervals'
        result = engine.evaluate_task(X, Y, ids, E, axes, cfg, task)
        counts = {r['mask']: r['n_candidates'] for r in result['diagnostics']['candidate_counts']}
        self.assertEqual(counts['uv+raman'], 2)
        self.assertEqual(counts['uv+raman+nir'], 8)
        self.assertTrue(all(r['spec']['nir_region'] in ('full', 'region_1', 'region_2', 'region_3') for r in result['diagnostics']['selections']['ridge__uv+raman+nir']))

    def test_balanced_budget_archived_reuse_is_deterministic_and_equal(self):
        X, Y, ids, E, axes, cfg, task, archive = archived_fixture()
        cfg['profile'] = 'full'
        task['condition'] = 'balanced_budget'
        with mock.patch.object(engine, '_score', side_effect=AssertionError('Archived scores must not be fitted again')):
            first = engine.evaluate_task(X, Y, ids, E, axes, cfg, task, archive)
            second = engine.evaluate_task(X, Y, ids, E, axes, cfg, task, archive)
        self.assertTrue(first['diagnostics']['reused_archived_inner_scores'])
        self.assertEqual(first['diagnostics']['balanced_budget'], second['diagnostics']['balanced_budget'])
        allocation = first['diagnostics']['balanced_budget']['allocation']
        self.assertEqual([r['selected_candidates'] for r in allocation], [4, 4])
        for model in first['predictions']:
            np.testing.assert_array_equal(first['predictions'][model], second['predictions'][model])
        task['budget_refit'] = 1
        third = engine.evaluate_task(X, Y, ids, E, axes, cfg, task, archive)
        self.assertNotEqual(first['diagnostics']['balanced_budget']['allocation'], third['diagnostics']['balanced_budget']['allocation'])

    def test_balanced_budget_rejects_tampered_or_misaligned_archive(self):
        X, Y, ids, E, axes, cfg, task, archive = archived_fixture()
        task['condition'] = 'balanced_budget'
        damaged = copy.deepcopy(archive)
        damaged['payload']['diagnostics']['candidate_scores'][0]['inner_mse_by_target'][0] += 1
        with self.assertRaisesRegex(ValueError, 'hash'):
            engine.evaluate_task(X, Y, ids, E, axes, cfg, task, damaged)
        with self.assertRaisesRegex(ValueError, 'arrays'):
            engine.evaluate_task(X, Y + 0.01, ids, E, axes, cfg, task, archive)
        with self.assertRaisesRegex(ValueError, 'order'):
            engine.evaluate_task({m: X[m][::-1] for m in X}, Y[::-1], ids[::-1], E, axes, cfg, task, archive)
        changed_axes = dict(axes, nir=axes['nir'] + 1)
        with self.assertRaisesRegex(ValueError, 'coordinate'):
            engine.evaluate_task(X, Y, ids, E, changed_axes, cfg, task, archive)
        cfg['grid']['krr_gammas'] = [0.2]
        with self.assertRaisesRegex(ValueError, 'grid'):
            engine.evaluate_task(X, Y, ids, E, axes, cfg, task, archive)

    def test_full_balanced_requires_archive_and_quick_can_score_local_grid(self):
        X, Y, ids, E, axes, cfg, task = fixture(['ridge'])
        task['condition'] = 'balanced_budget'
        cfg['profile'] = 'full'
        with self.assertRaisesRegex(ValueError, 'require'):
            engine.evaluate_task(X, Y, ids, E, axes, cfg, task)
        cfg['profile'] = 'quick'
        result = engine.evaluate_task(X, Y, ids, E, axes, cfg, task)
        self.assertFalse(result['diagnostics']['reused_archived_inner_scores'])

    def test_failed_msc_candidates_are_excluded_without_raw_fallback_relabeling(self):
        X, Y, ids, E, axes, cfg, task = fixture(['ridge'])
        X['nir'] = np.ones_like(X['nir'])
        E['nir'] = np.ones_like(E['nir'])
        result = engine.evaluate_task(X, Y, ids, E, axes, cfg, task)
        self.assertGreater(len(result['diagnostics']['candidate_failures']), 0)
        for record in result['diagnostics']['selections']['ridge__uv+raman+nir']:
            self.assertEqual(record['spec']['preprocessing']['nir'], 'raw')
        failures = [r for r in result['diagnostics']['candidate_scores'] if r['spec']['preprocessing'].get('nir') == 'msc']
        self.assertTrue(failures and all(not any(r['valid_by_target']) for r in failures))

    def test_repeated_bottle_ids_remain_grouped_and_bad_input_rejected(self):
        X, Y, ids, E, axes, cfg, task = fixture(['ridge'])
        rows = np.r_[np.arange(len(ids)), 0, 2, 5]
        result = engine.evaluate_task({m: X[m][rows] for m in X}, Y[rows], ids[rows], E, axes, cfg, task)
        for fold in result['diagnostics']['inner_splits']:
            self.assertFalse(set(fold['train_ids']) & set(fold['validation_ids']))
        with self.assertRaises(ValueError):
            engine.evaluate_task(X, np.full_like(Y, np.nan), ids, E, axes, cfg, task)
        with self.assertRaises(ValueError):
            engine.evaluate_task(X, np.ones_like(Y), ids, E, axes, cfg, task)


if __name__ == '__main__':
    unittest.main()
