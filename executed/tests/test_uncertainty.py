from pathlib import Path
import copy
import inspect
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from uncertainty import METHODS, build_intervals, conformal_quantile


class UncertaintyChecks(unittest.TestCase):
    def data(self, n_fit=64, n_cal=34, n_test=12):
        rng = np.random.RandomState(692)
        fit_x = np.linspace(-1.5, 1.5, n_fit)
        cal_x = np.linspace(-1.4, 1.4, n_cal)
        test_x = np.linspace(-1.3, 1.3, n_test)
        def features(x):
            return {'uv': np.column_stack((x, 0.3 * x ** 2, np.sin(x))), 'raman': rng.normal(size=(len(x), 4)), 'nir': rng.normal(size=(len(x), 5))}
        X_fit, X_cal, X_test = [features(x) for x in (fit_x, cal_x, test_x)]
        def response(x):
            return np.column_stack((10 + x, 5 + 0.7 * x))
        Y_fit, Y_cal = response(fit_x), response(cal_x)
        magnitude = np.exp(0.8 * fit_x)[:, None] * np.array([[0.2, 0.15]])
        oof = {'pls__uv': Y_fit - magnitude, 'mean': np.repeat(Y_fit.mean(axis=0, keepdims=True), n_fit, axis=0)}
        cal = {'pls__uv': Y_cal - np.exp(0.8 * cal_x)[:, None] * np.array([[0.22, 0.14]]), 'mean': np.repeat(Y_fit.mean(axis=0, keepdims=True), n_cal, axis=0)}
        test = {'pls__uv': response(test_x), 'mean': np.repeat(Y_fit.mean(axis=0, keepdims=True), n_test, axis=0)}
        cfg = {'interval_alpha': 0.1, 'uncertainty': {'pca_components': 2, 'ridge_alpha': 1.0, 'floor_fraction_sd': 0.02, 'clip_multiplier': 10.0}}
        return [Y_fit, X_fit, X_cal, X_test, oof, cal, test, Y_cal, cfg]

    def test_exact_finite_sample_rank(self):
        self.assertEqual(conformal_quantile(np.arange(34.0), 0.1), (31.0, 32))
        q, rank = conformal_quantile(np.arange(4.0), 0.1)
        self.assertEqual(rank, 5)
        self.assertTrue(np.isinf(q))
        q, rank = conformal_quantile(np.column_stack((np.arange(34.0), 2 * np.arange(34.0))), 0.1)
        np.testing.assert_equal(q, [31.0, 62.0])

    def test_all_four_methods_and_shapes(self):
        out, diagnostics = build_intervals(*self.data())
        self.assertEqual(set(diagnostics['methods']), set(METHODS))
        for model, methods in out.items():
            self.assertEqual(set(methods), set(METHODS))
            for value in methods.values():
                self.assertEqual(value['lower'].shape, (12, 2))
                self.assertEqual(value['upper'].shape, (12, 2))
                self.assertEqual(value['scale'].shape, (12, 2))
                self.assertEqual(value['rank'], 32)
                self.assertTrue(np.all(value['scale'] > 0))
                self.assertTrue(np.all(value['upper'] >= value['lower']))

    def test_constant_marginal_matches_absolute_residual_calibration(self):
        args = self.data()
        out, diagnostics = build_intervals(*args)
        expected, _ = conformal_quantile(np.abs(args[7] - args[5]['pls__uv']), 0.1)
        half = (out['pls__uv']['constant_marginal']['upper'] - out['pls__uv']['constant_marginal']['lower']) / 2
        np.testing.assert_allclose(half, np.broadcast_to(expected, half.shape))

    def test_calibration_labels_do_not_fit_scale_or_group_thresholds(self):
        args = self.data()
        first, first_diagnostics = build_intervals(*args)
        args[7] = args[7] + np.array([3.0, -5.0])
        second, second_diagnostics = build_intervals(*args)
        for model in first:
            np.testing.assert_array_equal(first[model]['adaptive_joint']['scale'], second[model]['adaptive_joint']['scale'])
            self.assertEqual(first_diagnostics['models'][model]['scale_learner'], second_diagnostics['models'][model]['scale_learner'])
        self.assertEqual(first_diagnostics['predicted_high_thresholds'], second_diagnostics['predicted_high_thresholds'])
        self.assertFalse(np.array_equal(first['pls__uv']['adaptive_joint']['upper'], second['pls__uv']['adaptive_joint']['upper']))

    def test_calibration_features_do_not_refit_feature_space(self):
        args = self.data()
        first, first_diagnostics = build_intervals(*args)
        args[2]['uv'] = args[2]['uv'] * 10 + 8
        second, second_diagnostics = build_intervals(*args)
        np.testing.assert_array_equal(first['pls__uv']['adaptive_joint']['scale'], second['pls__uv']['adaptive_joint']['scale'])
        self.assertEqual(first_diagnostics['models']['pls__uv']['feature_space'], second_diagnostics['models']['pls__uv']['feature_space'])

    def test_sample_adaptive_scale_and_mean_control(self):
        out, diagnostics = build_intervals(*self.data())
        scale = out['pls__uv']['adaptive_joint']['scale']
        self.assertGreater(scale[-1, 0], scale[0, 0] * 2)
        self.assertEqual(diagnostics['models']['pls__uv']['available_modalities'], ['uv'])
        self.assertEqual(diagnostics['models']['mean']['available_modalities'], [])
        np.testing.assert_allclose(out['mean']['adaptive_joint']['scale'], np.repeat(out['mean']['adaptive_joint']['scale'][:1], 12, axis=0))

    def test_omitted_modalities_cannot_change_scale(self):
        args = self.data()
        first, _ = build_intervals(*args)
        for part in (1, 2, 3):
            args[part]['raman'] *= 1e5
            args[part]['nir'] += 1e6
        second, _ = build_intervals(*args)
        for model in first:
            np.testing.assert_array_equal(first[model]['adaptive_joint']['scale'], second[model]['adaptive_joint']['scale'])

    def test_selected_models_require_explicit_acquisition_union(self):
        args = self.data()
        for part in (4, 5, 6):
            args[part]['pls__panel_t005'] = args[part].pop('pls__uv')
        with self.assertRaisesRegex(ValueError, 'model_modalities'):
            build_intervals(*args)
        args[8]['uncertainty']['model_modalities'] = {'pls__panel_t005': ['uv']}
        out, diagnostics = build_intervals(*args)
        self.assertIn('pls__panel_t005', out)
        self.assertEqual(diagnostics['models']['pls__panel_t005']['available_modalities'], ['uv'])

    def test_joint_scores_reconstruct_from_calibration_scales(self):
        args = self.data()
        out, diagnostics = build_intervals(*args)
        scale = np.asarray(diagnostics['models']['pls__uv']['adaptive_calibration_scale'])
        scores = np.max(np.abs(args[7] - args[5]['pls__uv']) / scale, axis=1)
        expected, rank = conformal_quantile(scores, 0.1)
        self.assertAlmostEqual(out['pls__uv']['adaptive_joint']['q'], expected)
        self.assertEqual(out['pls__uv']['adaptive_joint']['rank'], rank)

    def test_predicted_high_threshold_uses_training_predictions(self):
        args = self.data()
        _, diagnostics = build_intervals(*args)
        expected = np.quantile(args[4]['pls__uv'], 0.8, axis=0)
        np.testing.assert_allclose(diagnostics['predicted_high_thresholds']['pls__uv'], expected)

    def test_perfect_training_residuals_keep_positive_scales(self):
        args = self.data()
        args[4]['pls__uv'] = args[0].copy()
        out, diagnostics = build_intervals(*args)
        self.assertTrue(np.isfinite(out['pls__uv']['adaptive_joint']['scale']).all())
        self.assertTrue(np.all(out['pls__uv']['adaptive_joint']['scale'] > 0))

    def test_infinite_quantile_is_preserved(self):
        out, diagnostics = build_intervals(*self.data(n_cal=4))
        for methods in out.values():
            for value in methods.values():
                self.assertTrue(np.isneginf(value['lower']).all())
                self.assertTrue(np.isposinf(value['upper']).all())

    def test_input_validation_and_no_test_labels(self):
        args = self.data()
        self.assertNotIn('Y_test', inspect.signature(build_intervals).parameters)
        with self.assertRaises(TypeError):
            build_intervals(*args, Y_test=np.zeros((12, 2)))
        args[4]['pls__uv'][0, 0] = np.nan
        with self.assertRaises(ValueError):
            build_intervals(*args)
        args = self.data()
        args[5].pop('mean')
        with self.assertRaises(ValueError):
            build_intervals(*args)
        args = self.data()
        args[0][:, 0] = 1
        with self.assertRaises(ValueError):
            build_intervals(*args)

    def test_repeat_determinism(self):
        first, first_diagnostics = build_intervals(*self.data())
        second, second_diagnostics = build_intervals(*self.data())
        self.assertEqual(first_diagnostics, second_diagnostics)
        for model in first:
            for method in METHODS:
                np.testing.assert_array_equal(first[model][method]['lower'], second[model][method]['lower'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
