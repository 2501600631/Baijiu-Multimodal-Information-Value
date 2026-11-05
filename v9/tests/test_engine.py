from pathlib import Path
import ast
import copy
import hashlib
import inspect
import itertools
import sys
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / 'executed') not in sys.path:
    sys.path.insert(0, str(ROOT / 'executed'))
import sopls as original
from preprocessing import make_design, digest
from v9 import engine
from v9 import sopls_vector


def synthetic(seed=37, n=32):
    rng = np.random.RandomState(seed)
    blocks = {m: rng.normal(size=(n, 8)) for m in engine.MODALITIES}
    evaluation = {m: rng.normal(size=(7, 8)) for m in engine.MODALITIES}
    coefficients = rng.normal(size=(3, 11))
    latent = np.column_stack([blocks[m][:, 0] for m in engine.MODALITIES])
    response = latent.dot(coefficients) + rng.normal(scale=.3, size=(n, 11)) + np.arange(11)
    ids = np.array(['B%03d' % index for index in range(n)])
    axes = {m: np.linspace(1, 10, 8) for m in engine.MODALITIES}
    cfg = dict(seed=20261029, inner_folds=3, perblock=dict(components=[1, 2, 3, 5]),
               sopls_preprocessing_options={m: ['raw'] for m in engine.MODALITIES},
               preprocessing_settings={})
    task = dict(condition='sopls_perblock', seed=20261029, repeat=0, fold=0)
    return blocks, response, ids, evaluation, axes, cfg, task


class VectorKernelChecks(unittest.TestCase):
    def test_computational_body_is_exact_frozen_body_with_vector_entry_only(self):
        source = inspect.getsource(original.predict_sequential)
        expected = source.replace('def predict_sequential(training_blocks, evaluation_blocks, y, order, components):\n',
                                  'def predict_sequential(training_blocks, evaluation_blocks, y, order, components):\n    requested_components = component_vector(components, order)\n')
        expected = expected.replace('for modality in order:', 'for modality, requested in zip(order, requested_components):')
        expected = expected.replace('int(components)', 'int(requested)')
        self.assertEqual(ast.dump(ast.parse(expected)), ast.dump(ast.parse(inspect.getsource(sopls_vector.predict_sequential))))
        self.assertEqual(hashlib.sha256((ROOT / 'executed/sopls.py').read_bytes()).hexdigest(), sopls_vector.FROZEN_SOURCE_SHA256)

    def test_uniform_vectors_match_frozen_scalar_for_all_orders(self):
        X, Y, ids, E, axes, cfg, task = synthetic()
        for mask in engine.MASKS:
            for order in itertools.permutations(mask):
                for k in (1, 2, 3, 5):
                    training = {m: X[m] for m in order}
                    evaluation = {m: E[m] for m in order}
                    a, da, wa = original.predict_sequential(training, evaluation, Y[:, 0], order, k)
                    b, db, wb = sopls_vector.predict_sequential(training, evaluation, Y[:, 0], order, [k] * len(order))
                    np.testing.assert_array_equal(a, b)
                    self.assertEqual(da, db)
                    self.assertEqual(wa, wb)

    def test_heterogeneous_vector_applies_requested_rank_per_block(self):
        X, Y, ids, E, axes, cfg, task = synthetic()
        prediction, details, warnings = sopls_vector.predict_sequential(X, E, Y[:, 1], engine.MODALITIES, [1, 3, 2])
        self.assertTrue(np.isfinite(prediction).all())
        self.assertEqual([b['requested_components'] for b in details['blocks']], [1, 3, 2])
        self.assertEqual([b['components_capped'] for b in details['blocks']], [1, 3, 2])
        uniform, unused, unused2 = sopls_vector.predict_sequential(X, E, Y[:, 1], engine.MODALITIES, [1, 1, 1])
        self.assertGreater(float(np.max(np.abs(prediction - uniform))), 1e-5)

    def test_rank_cap_and_constant_response_keep_frozen_behavior(self):
        rng = np.random.RandomState(6)
        latent = rng.normal(size=(25, 2))
        x = latent.dot(rng.normal(size=(2, 7)))
        e = rng.normal(size=(4, 2)).dot(rng.normal(size=(2, 7)))
        pred, details, unused = sopls_vector.predict_sequential({'uv': x}, {'uv': e}, latent[:, 0] + .2 * latent[:, 1], ['uv'], [5])
        self.assertLessEqual(details['blocks'][0]['components_capped'], 2)
        constant, information, unused = sopls_vector.predict_sequential({'uv': x}, {'uv': e}, np.full(25, 7.), ['uv'], [5])
        np.testing.assert_array_equal(constant, np.full(4, 7.))
        self.assertTrue(information['constant_target'])

    def test_heterogeneous_known_latent_signal_is_recovered(self):
        rng = np.random.RandomState(108)
        latent = rng.normal(size=(100, 6))
        blocks = {'uv': latent[:, :1].dot(rng.normal(size=(1, 8))),
                  'raman': latent[:, 1:3].dot(rng.normal(size=(2, 8))),
                  'nir': latent[:, 3:].dot(rng.normal(size=(3, 8)))}
        response = latent.dot(np.array([2., -1., 3., .5, -2., 1.]))
        training = {m: value[:70] for m, value in blocks.items()}
        evaluation = {m: value[70:] for m, value in blocks.items()}
        heterogeneous, details, unused = sopls_vector.predict_sequential(training, evaluation, response[:70], engine.MODALITIES, [1, 2, 3])
        uniform, unused, unused2 = sopls_vector.predict_sequential(training, evaluation, response[:70], engine.MODALITIES, [1, 1, 1])
        self.assertLess(np.mean((heterogeneous - response[70:]) ** 2), 1e-18)
        self.assertGreater(np.mean((uniform - response[70:]) ** 2), 1e-4)

    def test_invalid_component_vectors_are_rejected(self):
        for values in ([1], [1, 2, 3], [1, 0], [1, -2], [1, 2.5], [1, float('nan')], [True, False], '2'):
            with self.subTest(values=values), self.assertRaises(ValueError):
                sopls_vector.component_vector(values, ['uv', 'raman'])


class StructuralSelectionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = synthetic()
        cls.result = engine.evaluate_task(*cls.inputs)

    def test_all_eleven_targets_and_both_fixed_masks_are_retained(self):
        result = self.result
        self.assertEqual(set(result['predictions']), {'sopls_perblock__uv+raman', 'sopls_perblock__uv+raman+nir'})
        for model, prediction in result['predictions'].items():
            self.assertEqual(prediction.shape, (7, 11))
            self.assertTrue(np.isfinite(prediction).all())
            selections = result['diagnostics']['selections'][model]
            self.assertEqual([r['target_index'] for r in selections], list(range(11)))
            for record in selections:
                self.assertEqual(set(record['spec']['order']), set(record['spec']['modalities']))
                self.assertEqual(len(record['spec']['order']), len(record['spec']['components_by_block']))
                self.assertTrue(set(record['spec']['components_by_block']).issubset({1, 2, 3, 5}))
                self.assertEqual(record['spec']['components'], record['spec']['components_by_block'])

    def test_all_prespecified_paths_and_greedy_stage_scores_are_logged(self):
        diag = self.result['diagnostics']
        self.assertEqual(diag['paths_per_mask'], {'uv+raman': 2, 'uv+raman+nir': 6})
        for path in diag['candidate_paths']:
            self.assertEqual(len(path['target_paths']), 11)
            for record in path['target_paths']:
                prefix = []
                for stage in record['stages']:
                    self.assertEqual(stage['fixed_prefix'], prefix)
                    self.assertEqual([candidate['k'] for candidate in stage['candidates']], [1, 2, 3, 5])
                    chosen = min(stage['candidates'], key=lambda candidate: (candidate['inner_mse'], candidate['k']))
                    self.assertEqual(stage['selected_k'], chosen['k'])
                    prefix.append(chosen['k'])
                    self.assertEqual(chosen['components_by_block'], prefix)
                    self.assertEqual(chosen['n_scored_rows'], len(self.inputs[1]))
                self.assertEqual(record['components_by_block'], prefix)
                self.assertEqual(record['inner_mse'], chosen['inner_mse'])

    def test_original_full_preprocessing_declaration_has_six_and_fiftyfour_paths(self):
        options = {'uv': ['raw'], 'raman': ['raw', 'snv', 'baseline_snv'], 'nir': ['raw', 'snv', 'sg1']}
        paths = engine._paths(options)
        self.assertEqual(sum(p['mask'] == 'uv+raman' for p in paths), 6)
        self.assertEqual(sum(p['mask'] == 'uv+raman+nir' for p in paths), 54)

    def test_later_stage_predictions_refit_full_prefix_with_local_training_data(self):
        X, Y, ids, E, axes, cfg, task = self.inputs
        record = next(path for path in self.result['diagnostics']['candidate_paths'] if path['mask'] == 'uv+raman+nir')
        candidate = record['target_paths'][0]['stages'][2]['candidates'][1]
        role = self.result['diagnostics']['inner_splits'][0]
        training, validation = np.array(role['train_row_indices']), np.array(role['validation_row_indices'])
        design = make_design({m: X[m][training] for m in engine.MODALITIES}, {m: X[m][validation] for m in engine.MODALITIES},
                             record['order'], record['preprocessing'], axes)
        prediction, details, unused = sopls_vector.predict_sequential(design['train_blocks'], design['evaluation_blocks'],
                                                                    Y[training, 0], record['order'], candidate['components_by_block'])
        expected = float(np.mean((prediction - Y[validation, 0]) ** 2))
        self.assertAlmostEqual(expected, candidate['fold_scores'][0]['mse'], places=12)
        self.assertEqual(details, candidate['fold_scores'][0]['fit_details'])

    def test_inner_preprocessing_means_use_only_inner_training_rows(self):
        X = self.inputs[0]
        roles = self.result['diagnostics']['inner_splits']
        for record in self.result['diagnostics']['preprocessing_fits']:
            if record['role'] == 'final':
                continue
            role = roles[int(record['role'])]
            training = np.array(role['train_row_indices'])
            self.assertEqual(record['train_ids'], role['train_ids'])
            self.assertEqual(record['metadata']['n_fit'], len(training))
            self.assertEqual(record['metadata']['training_mean_sha256'], digest(X[record['modality']][training].mean(axis=0)))

    def test_evaluation_spectra_cannot_change_selections_or_inner_scores(self):
        inputs = list(self.inputs)
        inputs[3] = {m: values * 20 + 100 for m, values in inputs[3].items()}
        alternative = engine.evaluate_task(*inputs)
        self.assertEqual(self.result['diagnostics']['selections'], alternative['diagnostics']['selections'])
        self.assertEqual(self.result['diagnostics']['candidate_paths'], alternative['diagnostics']['candidate_paths'])
        self.assertTrue(any(not np.array_equal(self.result['predictions'][model], alternative['predictions'][model]) for model in alternative['predictions']))

    def test_duplicate_bottle_groups_never_cross_inner_roles(self):
        inputs = list(self.inputs)
        indices = np.r_[np.arange(len(inputs[1])), [1, 4, 7, 12]]
        inputs[0] = {m: values[indices] for m, values in inputs[0].items()}
        inputs[1] = inputs[1][indices]
        inputs[2] = inputs[2][indices]
        result = engine.evaluate_task(*inputs)
        self.assertLess(result['diagnostics']['n_distinct_training_bottles'], result['diagnostics']['n_training_rows'])
        for role in result['diagnostics']['inner_splits']:
            self.assertFalse(set(role['train_ids']) & set(role['validation_ids']))

    def test_any_candidate_fit_failure_aborts_instead_of_returning_fallback(self):
        with patch.object(engine, 'predict_sequential', side_effect=RuntimeError('synthetic forced fit failure')):
            with self.assertRaisesRegex(RuntimeError, 'aborting task'):
                engine.evaluate_task(*self.inputs)

    def test_invalid_declared_transform_aborts_even_if_raw_candidate_would_work(self):
        inputs = copy.deepcopy(list(self.inputs))
        inputs[5]['sopls_preprocessing_options']['raman'] = ['raw', 'snv']
        inputs[0]['raman'][:] = 1
        with self.assertRaisesRegex(RuntimeError, 'aborting task'):
            engine.evaluate_task(*inputs)

    def test_final_path_tie_rule_is_prespecified(self):
        a = dict(path_id=5, order_rank=1, preprocessing_rank=2)
        b = dict(path_id=3, order_rank=0, preprocessing_rank=1)
        selected = dict(inner_mse=.5, components_by_block=[1, 2, 3])
        self.assertLess(engine._tie(b, selected), engine._tie(a, selected))
        larger = dict(inner_mse=.5, components_by_block=[2, 3, 3])
        self.assertLess(engine._tie(a, selected), engine._tie(b, larger))
        better = dict(inner_mse=.49, components_by_block=[5, 5, 5])
        self.assertLess(engine._tie(a, better), engine._tie(b, selected))

    def test_prefix_cache_reuses_work_and_records_actual_cost(self):
        counts = self.result['diagnostics']['fit_counts']
        self.assertGreater(counts['inner_prefix_cache_hits'], 0)
        self.assertGreater(counts['inner_requested_block_fits'], counts['inner_prefix_calls'])
        self.assertEqual(counts['final_prefix_calls'], 22)
        self.assertEqual(counts['final_requested_block_fits'], 55)
        self.assertGreater(counts['inner_requested_block_fits'], 9 * 11)

    def test_components_grid_cannot_expand_after_prespecification(self):
        inputs = copy.deepcopy(list(self.inputs))
        inputs[5]['perblock']['components'] = [1, 2, 3, 5, 10]
        with self.assertRaisesRegex(ValueError, 'exactly'):
            engine.evaluate_task(*inputs)


if __name__ == '__main__':
    unittest.main()
