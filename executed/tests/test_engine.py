from pathlib import Path
import copy
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import candidate_grid, select_and_predict, model_names, grouped_folds


def configuration(algorithms=None):
    return dict(algorithms=['pls', 'ridge', 'krr', 'sopls'] if algorithms is None else algorithms, inner_folds=3, panel_tolerances=[0, 0.02, 0.05, 0.10], preprocessing_options={'uv': ['raw'], 'raman': ['raw', 'snv'], 'nir': ['raw', 'snv']}, grid=dict(pls_components=[2], ridge_alphas=[0.1], krr_alphas=[0.1], krr_gammas=[0.3], sopls_components=[1]))


def synthetic(seed=77):
    rng = np.random.RandomState(seed)
    latent = rng.normal(size=(33, 3))
    X = {modality: latent.dot(rng.normal(size=(3, 7)))+rng.normal(scale=0.2, size=(33, 7)) for modality in ('uv', 'raman', 'nir')}
    Y = latent.dot(rng.normal(size=(3, 3)))+rng.normal(scale=0.1, size=(33, 3))+8
    ids = np.array(['synthetic_%d' % i for i in range(27)])
    return {key: value[:27] for key, value in X.items()}, Y[:27], ids, {key: value[27:] for key, value in X.items()}


class EngineChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.Y, cls.ids, cls.E = synthetic()
        cls.cfg = configuration()
        cls.prediction, cls.diagnostics = select_and_predict(cls.X, cls.Y, cls.ids, cls.E, cls.cfg, 19)

    def test_independent_snv_candidates_exist(self):
        grid = candidate_grid(self.cfg)
        triples = {tuple(spec['preprocessing'][m] for m in ('uv', 'raman', 'nir')) for spec in grid if spec['algorithm'] == 'ridge' and spec['mask'] == 'uv+raman+nir'}
        self.assertEqual(triples, {('raw', 'raw', 'raw'), ('raw', 'snv', 'raw'), ('raw', 'raw', 'snv'), ('raw', 'snv', 'snv')})

    def test_all_outputs_and_oof_values_present(self):
        self.assertEqual(set(self.prediction), set(model_names(self.cfg)))
        self.assertEqual(len(self.prediction), 53)
        for model in self.prediction:
            self.assertEqual(self.prediction[model].shape, (6, 3))
            values = np.asarray(self.diagnostics['oof_predictions'][model])
            self.assertEqual(values.shape, self.Y.shape)
            self.assertTrue(np.isfinite(values).all())

    def test_selectors_reuse_exact_fixed_predictions(self):
        for model, records in self.diagnostics['selection_by_model'].items():
            if '__select_' not in model and '__panel_' not in model:
                continue
            algorithm = model.split('__')[0]
            for record in records:
                j = record['target_index']
                reference = algorithm+'__'+record['mask']
                np.testing.assert_array_equal(self.prediction[model][:, j], self.prediction[reference][:, j])

    def test_panel_selection_obeys_global_budget(self):
        for model, panel in self.diagnostics['panel_selection'].items():
            records = self.diagnostics['selection_by_model'][model]
            masks = {record['mask'] for record in records}
            self.assertEqual(masks, {panel['common_mask']})
            self.assertLessEqual(panel['chosen_inner_standardized_mse'], panel['best_inner_standardized_mse']*(1+panel['tolerance'])+1e-12)
            eligible = [mask for mask, score in panel['all_mask_scores'].items() if score <= panel['best_inner_standardized_mse']*(1+panel['tolerance'])+1e-15]
            self.assertEqual(len(panel['common_mask'].split('+')), min(len(mask.split('+')) for mask in eligible))

    def test_acquisition_is_union_not_average(self):
        for model, records in self.diagnostics['selection_by_model'].items():
            actual = set()
            for record in records:
                actual.update(record['mask'].split('+'))
            saved = self.diagnostics['acquisition_by_model'][model]
            self.assertEqual(actual, set(saved['modality_union']))
            self.assertEqual(len(actual), saved['n_modalities'])

    def test_evaluation_perturbation_cannot_change_training_choices(self):
        cfg = configuration(['ridge'])
        first, a = select_and_predict(self.X, self.Y, self.ids, self.E, cfg, 22)
        altered = {key: values+100 for key, values in self.E.items()}
        second, b = select_and_predict(self.X, self.Y, self.ids, altered, cfg, 22)
        self.assertEqual(a['selection_by_model'], b['selection_by_model'])
        self.assertEqual(a['panel_selection'], b['panel_selection'])
        self.assertEqual(a['oof_predictions'], b['oof_predictions'])

    def test_mean_oof_uses_each_inner_training_partition(self):
        oof = np.asarray(self.diagnostics['oof_predictions']['mean'])
        for split in self.diagnostics['inner_splits']:
            fit = split['train_row_indices']
            validation = split['validation_row_indices']
            np.testing.assert_allclose(oof[validation], np.repeat(self.Y[fit].mean(axis=0, keepdims=True), len(validation), axis=0))

    def test_duplicate_id_groups_stay_together(self):
        ids = np.array(['a', 'a', 'b', 'c', 'd', 'e', 'e', 'f'])
        for train, validation in grouped_folds(ids, 3, 9):
            self.assertFalse(set(ids[train]) & set(ids[validation]))

    def test_panel_scores_are_response_scale_invariant(self):
        cfg = configuration(['ridge'])
        first, a = select_and_predict(self.X, self.Y, self.ids, self.E, cfg, 22)
        scale = np.array([1000.0, 0.01, 2.0])
        second, b = select_and_predict(self.X, self.Y*scale, self.ids, self.E, cfg, 22)
        for model, panel in a['panel_selection'].items():
            self.assertEqual(panel['common_mask'], b['panel_selection'][model]['common_mask'])
            np.testing.assert_allclose(list(panel['all_mask_scores'].values()), list(b['panel_selection'][model]['all_mask_scores'].values()), rtol=1e-10)

    def test_sopls_orders_are_candidates_not_outer_choices(self):
        grid = candidate_grid(self.cfg)
        orders = {tuple(spec['order']) for spec in grid if spec['algorithm'] == 'sopls' and spec['mask'] == 'uv+raman+nir'}
        self.assertEqual(len(orders), 6)


if __name__ == '__main__':
    unittest.main()
