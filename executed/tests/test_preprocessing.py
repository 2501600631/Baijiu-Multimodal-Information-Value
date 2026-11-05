from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing import transform_block, make_design


class PreprocessingChecks(unittest.TestCase):
    def test_nonuniform_linear_signal_has_correct_derivative(self):
        axis = np.array([1.0, 1.1, 1.8, 2.0, 3.5, 5.0, 6.1, 8.0, 9.0])
        values = np.vstack([3*axis+2, -2*axis+40])
        result, meta = transform_block(values, 'sg1', axis, {'sg_window': 7})
        np.testing.assert_allclose(result[0], 3.0, atol=1e-12)
        np.testing.assert_allclose(result[1], -2.0, atol=1e-12)
        self.assertTrue(meta['interpolated'])
        self.assertEqual(meta['derivative_order'], 1)

    def test_training_statistics_are_independent_of_evaluation_values(self):
        rng = np.random.RandomState(31)
        train = {'uv': rng.normal(size=(12, 9)), 'raman': rng.normal(size=(12, 9))}
        first = {'uv': rng.normal(size=(3, 9)), 'raman': rng.normal(size=(3, 9))}
        second = {key: value+10000 for key, value in first.items()}
        a = make_design(train, first, ('uv', 'raman'), {'uv': 'raw', 'raman': 'snv'})
        b = make_design(train, second, ('uv', 'raman'), {'uv': 'raw', 'raman': 'snv'})
        np.testing.assert_array_equal(a['train'], b['train'])
        self.assertEqual(a['preprocessing'], b['preprocessing'])
        np.testing.assert_allclose(np.mean(np.sum(a['train_blocks']['uv']**2, axis=1)), 1.0)

    def test_preprocessing_does_not_modify_source(self):
        rng = np.random.RandomState(9)
        source = rng.normal(size=(4, 21))
        original = source.copy()
        for method in ('raw', 'snv', 'baseline_snv', 'sg1'):
            transformed, meta = transform_block(source, method)
            self.assertTrue(np.isfinite(transformed).all())
            np.testing.assert_array_equal(source, original)

    def test_baseline_correction_preserves_peak_shape(self):
        axis = np.linspace(0, 1, 101)
        peak = np.exp(-((axis-0.5)/0.04)**2)
        baseline = 2+3*axis+axis**2
        source = (baseline+peak)[None, :]
        raw, unused = transform_block(source, 'snv', axis)
        corrected, unused = transform_block(source, 'baseline_snv', axis, {'baseline_lambda': 10000, 'baseline_iterations': 20})
        self.assertGreater(np.corrcoef(corrected[0], peak)[0, 1], np.corrcoef(raw[0], peak)[0, 1]+0.6)

    def test_rejects_nonmonotonic_axis(self):
        with self.assertRaises(ValueError):
            transform_block(np.ones((3, 4)), 'raw', [0, 2, 1, 3])

    def test_rejects_constant_snv(self):
        with self.assertRaises(ValueError):
            transform_block(np.ones((3, 9)), 'snv')


if __name__ == '__main__':
    unittest.main()
