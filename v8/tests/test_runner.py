from pathlib import Path
from collections import Counter
import copy
import json
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from v8 import runner
from data_io import Dataset, TARGETS


def synthetic_dataset(n=36):
    random = np.random.RandomState(2026)
    latent = random.normal(size=(n, 4))
    blocks, axes = {}, {}
    for name, features in [('uv', 9), ('raman', 11), ('nir', 15)]:
        blocks[name] = np.linspace(0.2, 1.8, features)[None, :] + latent.dot(random.normal(scale=0.2, size=(4, features))) + random.normal(scale=0.03, size=(n, features))
        axes[name] = np.linspace(4000., 12000., features) if name == 'nir' else np.arange(features, dtype=float)
    ids = np.array(['synthetic_%03d' % index for index in range(n)])
    response = latent.dot(random.normal(size=(4, 11))) + random.normal(scale=0.05, size=(n, 11))
    values = pd.DataFrame(response, index=pd.Index(ids, name='sample_id'), columns=list(TARGETS))
    return Dataset(ids, blocks, axes, values, pd.DataFrame(index=values.index), dict(synthetic=True, n_matched_bottles=n))


def config(profile='quick'):
    return json.loads((ROOT/'configs'/('v8_' + profile + '.json')).read_text(encoding='utf-8'))


def task(condition):
    return dict(stage='v8', condition=condition, repeat=0, fold=0, training_size='full', budget_refit=-1, bootstrap_refit=-1, seed=20261029)


def synthetic_splits(repeats, folds, n=139):
    partitions = np.array_split(np.arange(n), folds)
    result = []
    for repeat in range(repeats):
        for fold, test in enumerate(partitions):
            train = np.setdiff1d(np.arange(n), test)
            result.append(dict(repeat=repeat, fold=fold, tr=train, te=test))
    return result


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = synthetic_dataset()
        cls.cfg = config()
        cls.split = dict(repeat=0, fold=0, tr=np.arange(27), te=np.arange(27, 36))
        cls.weight_task = task('nir_weight')
        cls.residual_task = task('residual')
        cls.weight_payload = runner.evaluate(cls.dataset, cls.cfg, cls.split, cls.weight_task)
        cls.residual_payload = runner.evaluate(cls.dataset, cls.cfg, cls.split, cls.residual_task)

    def validate_weight(self, payload):
        return runner.validate_payload(payload, self.dataset, self.cfg, self.split, self.weight_task)

    def test_synthetic_weight_and_residual_roundtrip_validate_all_eleven_targets(self):
        for payload, item in [(self.weight_payload, self.weight_task), (self.residual_payload, self.residual_task)]:
            self.assertTrue(runner.validate_payload(payload, self.dataset, self.cfg, self.split, item))
            frame = pd.DataFrame(payload['predictions'])
            self.assertEqual(len(frame), 9 * 11 * 2)
            self.assertEqual(set(frame.target), set(self.dataset.target_names))
            self.assertEqual(set(frame.sample_id), set(self.dataset.ids[self.split['te']]))
            self.assertFalse(set(payload['fit_ids']) & set(payload['test_ids']))
            zero = frame[frame.model.str.endswith('__weight_zero') | frame.model.str.endswith('__zero')]
            self.assertTrue(zero.effective_nir_weight.eq(0).all())
            self.assertTrue(zero.n_acquisition_modalities.eq(2).all())
            self.assertTrue(np.isfinite(frame[['y_pred', 'y_true']].to_numpy()).all())

    def test_full_and_quick_plan_cover_all_requested_conditions(self):
        full = runner.plan_tasks(config('full'), synthetic_splits(3, 5))
        quick = runner.plan_tasks(config('quick'), synthetic_splits(1, 2))
        self.assertEqual(len(full), 255)
        self.assertEqual(len(quick), 16)
        self.assertEqual(Counter(t['condition'] for t in full), Counter(unscaled=15, nir_extended=15, nir_intervals=15, balanced_budget=75, nir_weight=15, residual=15, nir_weight_learning=30, nir_weight_stability=75))
        self.assertEqual(Counter(t['condition'] for t in quick), Counter({condition:2 for condition in runner.CONDITIONS}))
        self.assertEqual(len({runner.task_name(t) for t in full}), 255)
        self.assertEqual({t['budget_refit'] for t in full if t['condition']=='balanced_budget'}, set(range(5)))
        self.assertEqual({t['training_size'] for t in full if t['condition']=='nir_weight_learning'}, {'40', '70'})
        for item in full:
            self.assertEqual(item['seed'], 20261029 + 10000 * item['repeat'] + 100 * item['fold'])

    def test_learning_prefix_and_bootstrap_roles_are_deterministic(self):
        dataset = synthetic_dataset(139)
        split = synthetic_splits(3, 5)[0]
        forty = dict(task('nir_weight_learning'), training_size='40')
        seventy = dict(task('nir_weight_learning'), training_size='70')
        fit40, test40 = runner.roles(dataset, split, forty)
        fit70, test70 = runner.roles(dataset, split, seventy)
        np.testing.assert_array_equal(fit40, fit70[:40])
        np.testing.assert_array_equal(test40, test70)
        np.testing.assert_array_equal(test40, split['te'])
        bootstrap = dict(task('nir_weight_stability'), budget_refit=0, bootstrap_refit=0)
        first, outer = runner.roles(dataset, split, bootstrap)
        second, unused = runner.roles(dataset, split, bootstrap)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), len(split['tr']))
        self.assertLess(len(np.unique(first)), len(first))
        self.assertFalse(set(first) & set(outer))

    def test_synthetic_grouped_bootstrap_payload_validates(self):
        item = dict(task('nir_weight_stability'), budget_refit=0, bootstrap_refit=0)
        payload = runner.evaluate(self.dataset, self.cfg, self.split, item)
        self.assertTrue(runner.validate_payload(payload, self.dataset, self.cfg, self.split, item))
        self.assertLess(len(set(payload['fit_ids'])), len(payload['fit_ids']))
        for fold in payload['diagnostics']['inner_splits']:
            self.assertFalse(set(fold['train_ids']) & set(fold['validation_ids']))

    def test_cache_hash_signature_and_semantic_truth_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix='_synthetic_cache_', dir=str(ROOT)) as folder:
            path = Path(folder)/'task.json.gz'
            wrapper = dict(signature='synthetic_signature', payload_sha256=runner.digest_object(self.weight_payload), payload=self.weight_payload)
            runner.atomic_json(path, wrapper)
            loaded = runner.checked_cache(path, 'synthetic_signature', self.dataset, self.cfg, self.split, self.weight_task)
            self.assertEqual(loaded['fit_ids'], self.weight_payload['fit_ids'])
            with self.assertRaisesRegex(ValueError, 'signature'):
                runner.checked_cache(path, 'wrong_signature', self.dataset, self.cfg, self.split, self.weight_task)
            altered = copy.deepcopy(wrapper)
            altered['payload']['predictions'][0]['y_pred'] += 1
            runner.atomic_json(path, altered)
            with self.assertRaisesRegex(ValueError, 'hash'):
                runner.checked_cache(path, 'synthetic_signature', self.dataset, self.cfg, self.split, self.weight_task)
            altered['payload']['predictions'][0]['y_true'] += 0.01
            altered['payload_sha256'] = runner.digest_object(altered['payload'])
            runner.atomic_json(path, altered)
            with self.assertRaisesRegex(ValueError, 'reference'):
                runner.checked_cache(path, 'synthetic_signature', self.dataset, self.cfg, self.split, self.weight_task)

    def test_row_metadata_and_training_scales_are_rejected_when_changed(self):
        for field, value in [('seed', -1), ('n_train', 1), ('n_unique_train', 1), ('train_sd', 999.), ('outer_train_sd', 999.)]:
            payload = copy.deepcopy(self.weight_payload)
            payload['predictions'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate_weight(payload)

    def test_wrong_bottle_roles_target_order_and_prediction_keys_are_rejected(self):
        payload = copy.deepcopy(self.weight_payload)
        payload['fit_ids'][0] = payload['test_ids'][0]
        with self.assertRaises(ValueError):
            self.validate_weight(payload)
        payload = copy.deepcopy(self.weight_payload)
        payload['targets'] = payload['targets'][::-1]
        with self.assertRaises(ValueError):
            self.validate_weight(payload)
        payload = copy.deepcopy(self.weight_payload)
        payload['predictions'][-1] = copy.deepcopy(payload['predictions'][0])
        with self.assertRaises(ValueError):
            self.validate_weight(payload)

    def test_inner_roles_reject_outer_identity_and_repeated_validation_fold(self):
        payload = copy.deepcopy(self.weight_payload)
        payload['diagnostics']['inner_splits'][0]['validation_ids'].append(payload['test_ids'][0])
        with self.assertRaises(ValueError):
            self.validate_weight(payload)
        payload = copy.deepcopy(self.weight_payload)
        payload['diagnostics']['inner_splits'][1] = copy.deepcopy(payload['diagnostics']['inner_splits'][0])
        with self.assertRaises(ValueError):
            self.validate_weight(payload)

    def test_zero_baseline_cannot_be_relabelled_as_positive_nir_gate(self):
        payload = copy.deepcopy(self.weight_payload)
        for row in payload['predictions']:
            if row['model'] == 'krr__weight_zero':
                row['effective_nir_weight'] = 1.
                row['n_acquisition_modalities'] = 3
        with self.assertRaises(ValueError):
            self.validate_weight(payload)

    def test_inner_selection_target_records_are_complete(self):
        payload = copy.deepcopy(self.weight_payload)
        model = payload['model_names'][0]
        payload['diagnostics']['selections'][model].pop()
        with self.assertRaises(ValueError):
            self.validate_weight(payload)

    def test_point_tasks_reject_nonzero_calibration_count(self):
        payload = copy.deepcopy(self.weight_payload)
        payload['predictions'][0]['n_cal'] = 9
        with self.assertRaises(ValueError):
            self.validate_weight(payload)

    def test_nonfinite_numbers_cannot_be_frozen_into_completed_cache(self):
        for value in (float('nan'), float('inf'), float('-inf')):
            with self.assertRaises(ValueError):
                runner.digest_object(dict(result=value))
        payload = copy.deepcopy(self.weight_payload)
        payload['predictions'][0]['y_pred'] = float('nan')
        with self.assertRaises(ValueError):
            self.validate_weight(payload)

    def test_partial_completion_never_claims_all_requested_tasks(self):
        plan = [self.weight_task, self.residual_task]
        lookup = {(0, 0):self.split}
        with tempfile.TemporaryDirectory(prefix='_synthetic_completion_', dir=str(ROOT)) as folder:
            out = Path(folder)
            (out/'tasks').mkdir()
            runner.atomic_json(out/'tasks'/runner.task_name(self.weight_task), dict(signature='synthetic', payload_sha256=runner.digest_object(self.weight_payload), payload=self.weight_payload))
            report = runner.consolidate(out, plan, 'synthetic', self.dataset, self.cfg, lookup)
            self.assertEqual(report['status'], 'incomplete')
            self.assertEqual(report['completed_tasks'], 1)
            self.assertFalse(report['stages']['residual']['complete'])
            runner.atomic_json(out/'tasks'/runner.task_name(self.residual_task), dict(signature='synthetic', payload_sha256=runner.digest_object(self.residual_payload), payload=self.residual_payload))
            report = runner.consolidate(out, plan, 'synthetic', self.dataset, self.cfg, lookup)
            self.assertEqual(report['status'], 'all_requested_stages_complete')
            self.assertEqual(report['prediction_rows'], 2 * 9 * 11 * 2)
            self.assertFalse(report['quick_results_are_formal_evidence'])
            self.assertFalse(report['external_validation'])

    def test_recorded_source_and_result_paths_are_protected(self):
        for name in ('executed', 'results', 'figures', 'v8'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'immutable'):
                runner.main(['--profile', 'quick', '--stage', 'prepare', '--out', str(ROOT/name/'synthetic_output')])


if __name__ == '__main__':
    unittest.main()
