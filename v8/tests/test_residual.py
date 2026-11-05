import unittest

import numpy as np

from executed.legacy_engine import grouped_folds
from v8.residual import MODELS, _crossfitted_base, evaluate_residual


def fixture(n_train=48, n_eval=23, seed=8721):
    rng = np.random.RandomState(seed)
    blocks = {name: rng.normal(size=(n_train, width))
              for name, width in [('uv', 4), ('raman', 3), ('nir', 3)]}
    evaluation = {name: rng.normal(size=(n_eval, values.shape[1])) for name, values in blocks.items()}
    response = rng.normal(size=(n_train, 11))
    ids = np.asarray(['b%03d' % index for index in range(n_train)])
    return blocks, response, ids, evaluation


def config(multipliers=None):
    return dict(inner_folds=3, v8=dict(residual_base_alphas=[0.1, 1.0],
                residual_correction_alphas=[0.1, 1.0],
                residual_multipliers=[0.0, 0.5, 1.0] if multipliers is None else multipliers,
                residual_subfolds=3))


def independent_ur_prediction(training, response, evaluation, alpha):
    train_columns = []
    eval_columns = []
    for name in ['uv', 'raman']:
        mean = training[name].mean(axis=0)
        centered = training[name] - mean
        scale = max(np.sqrt(np.sum(centered ** 2) / len(centered)), 1e-12)
        train_columns.append(centered / scale)
        eval_columns.append((evaluation[name] - mean) / scale)
    design = np.concatenate(train_columns, axis=1)
    eval_design = np.concatenate(eval_columns, axis=1)
    mean_response = response.mean(axis=0)
    coefficients = np.linalg.solve(design.T.dot(design) + alpha * np.eye(design.shape[1]),
                                   design.T.dot(response - mean_response))
    return eval_design.dot(coefficients) + mean_response


class ResidualCorrectionTests(unittest.TestCase):

    def test_selection_ignores_evaluation_values_and_is_deterministic(self):
        blocks, response, ids, evaluation = fixture()
        first = evaluate_residual(blocks, response, ids, evaluation, {}, config(), dict(seed=339))
        modified = {name: values * 1000 + 800 for name, values in evaluation.items()}
        second = evaluate_residual(blocks, response, ids, modified, {}, config(), dict(seed=339))
        third = evaluate_residual(blocks, response, ids, evaluation, {}, config(), dict(seed=339))
        self.assertEqual(first['diagnostics'], second['diagnostics'])
        self.assertEqual(first['diagnostics'], third['diagnostics'])
        for model in MODELS:
            self.assertEqual(first['predictions'][model].shape, (len(next(iter(evaluation.values()))), 11))
            np.testing.assert_array_equal(first['predictions'][model], third['predictions'][model])
            self.assertTrue(np.isfinite(second['predictions'][model]).all())

    def test_grouped_roles_include_duplicates_without_identity_overlap(self):
        blocks, response, ids, evaluation = fixture(n_train=30)
        index = np.concatenate([np.arange(30), np.asarray([0, 0, 2, 7, 7, 7, 12])])
        duplicated = {name: values[index] for name, values in blocks.items()}
        result = evaluate_residual(duplicated, response[index], ids[index], evaluation, {}, config(), dict(seed=72))
        diagnostics = result['diagnostics']
        for role in diagnostics['inner_splits'] + diagnostics['final_residual_crossfit_splits']:
            self.assertFalse(set(role['train_ids']) & set(role['validation_ids']))
        for inner in diagnostics['inner_residual_subfolds']:
            outer_validation = diagnostics['inner_splits'][inner['inner_fold_index']]['validation_ids']
            for role in inner['folds']:
                self.assertFalse(set(role['train_ids']) & set(role['validation_ids']))
                self.assertFalse((set(role['train_ids']) | set(role['validation_ids'])) & set(outer_validation))
        self.assertEqual(diagnostics['n_training_rows'], 37)
        self.assertEqual(diagnostics['n_distinct_training_bottles'], 30)

    def test_fixed_nuisance_oof_predictions_exclude_own_labels(self):
        blocks, response, ids, evaluation = fixture(n_train=36)
        alphas = [0.01, 0.1, 1.0, 10.0]
        before, roles = _crossfitted_base(blocks, response, ids, alphas, 3, 9102)
        held_out = np.asarray(roles[0]['validation_indices'])
        changed = response.copy()
        changed[held_out] += np.arange(1, 12)[None, :] * 1000
        after, ignored = _crossfitted_base(blocks, changed, ids, alphas, 3, 9102)
        for alpha in alphas:
            np.testing.assert_array_equal(before[alpha][held_out], after[alpha][held_out])

    def test_zero_gate_equals_independent_raw_ur_ridge_selection(self):
        blocks, response, ids, evaluation = fixture(n_train=45)
        cfg = config([0.0])
        seed = 117
        result = evaluate_residual(blocks, response, ids, evaluation, {}, cfg, dict(seed=seed))
        alphas = cfg['v8']['residual_base_alphas']
        oof = np.zeros((len(alphas),) + response.shape)
        for training, validation in grouped_folds(ids, cfg['inner_folds'], seed):
            fit = {name: values[training] for name, values in blocks.items()}
            valid = {name: values[validation] for name, values in blocks.items()}
            for index, alpha in enumerate(alphas):
                oof[index, validation] = independent_ur_prediction(fit, response[training], valid, alpha)
        chosen = ((oof - response[None, :, :]) ** 2).mean(axis=1).argmin(axis=0)
        expected = np.empty((len(evaluation['uv']), 11))
        for target, index in enumerate(chosen):
            expected[:, target] = independent_ur_prediction(blocks, response, evaluation, alphas[index])[:, target]
        np.testing.assert_allclose(result['predictions'][MODELS[0]], expected, rtol=1e-11, atol=1e-11)
        np.testing.assert_array_equal(result['predictions'][MODELS[0]], result['predictions'][MODELS[1]])
        for model in MODELS:
            self.assertEqual(result['diagnostics']['acquisition_by_model'][model]['n_modalities'], 2)
            self.assertEqual(result['diagnostics']['acquisition_by_model'][model]['modality_union'], ['uv', 'raman'])
            self.assertTrue(all(record['spec']['correction_alpha'] is None for record in result['diagnostics']['selections'][model]))

    def test_known_nir_residual_signal_improves_held_out_prediction(self):
        blocks, ignored, ids, evaluation = fixture(n_train=90, n_eval=80, seed=11)
        scales = np.linspace(0.5, 2.0, 11)
        coefficients = np.linspace(-0.4, 0.8, 11)
        response = blocks['uv'][:, :1] * coefficients + blocks['raman'][:, :1] * 0.3
        response += blocks['nir'][:, :1] * (3.0 * scales)
        truth = evaluation['uv'][:, :1] * coefficients + evaluation['raman'][:, :1] * 0.3
        truth += evaluation['nir'][:, :1] * (3.0 * scales)
        cfg = config()
        cfg['v8']['residual_base_alphas'] = [0.01, 0.1, 1.0]
        cfg['v8']['residual_correction_alphas'] = [0.01, 0.1, 1.0]
        result = evaluate_residual(blocks, response, ids, evaluation, {}, cfg, dict(seed=88))
        zero_error = np.mean((result['predictions'][MODELS[0]] - truth) ** 2 / scales[None, :] ** 2)
        selected_error = np.mean((result['predictions'][MODELS[1]] - truth) ** 2 / scales[None, :] ** 2)
        self.assertLess(selected_error, zero_error * 0.25)
        self.assertEqual(result['diagnostics']['chosen_multiplier'], 1.0)
        self.assertEqual(result['diagnostics']['acquisition_by_model'][MODELS[1]]['n_modalities'], 3)

    def test_common_gate_uses_standardized_scores_after_raw_target_tuning(self):
        blocks, response, ids, evaluation = fixture()
        response *= np.geomspace(0.01, 100, 11)[None, :]
        result = evaluate_residual(blocks, response, ids, evaluation, {}, config(), dict(seed=301))
        diagnostics = result['diagnostics']
        candidates = diagnostics['candidate_scores']
        for gate in diagnostics['shared_multiplier_scores']:
            multiplier = gate['multiplier']
            eligible = [record for record in candidates if record['spec']['multiplier'] == multiplier]
            standardized = []
            for target, candidate_id in enumerate(gate['selected_candidate_ids']):
                expected = min(eligible, key=lambda record: (record['inner_mse'][target], record['candidate_id']))
                self.assertEqual(expected['candidate_id'], candidate_id)
                standardized.append(expected['inner_standardized_mse'][target])
            self.assertAlmostEqual(gate['inner_panel_standardized_mse'], float(np.mean(standardized)), places=14)
        chosen = min(diagnostics['shared_multiplier_scores'], key=lambda record: (record['inner_panel_standardized_mse'], record['multiplier']))
        self.assertEqual(chosen['multiplier'], diagnostics['chosen_multiplier'])
        self.assertEqual(len(diagnostics['selections'][MODELS[1]]), 11)

    def test_invalid_duplicate_identity_is_rejected(self):
        blocks, response, ids, evaluation = fixture()
        ids[1] = ids[0]
        with self.assertRaises(ValueError):
            evaluate_residual(blocks, response, ids, evaluation, {}, config(), dict(seed=39))


if __name__ == '__main__':
    unittest.main()
