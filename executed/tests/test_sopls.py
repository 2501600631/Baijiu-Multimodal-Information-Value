from pathlib import Path
import sys
import unittest
import numpy as np
from sklearn.cross_decomposition import PLSRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sopls import predict_sequential


class SequentialModelChecks(unittest.TestCase):
    def test_single_block_matches_pls(self):
        rng = np.random.RandomState(17)
        x = rng.normal(size=(35, 8))
        evaluation = rng.normal(size=(8, 8))
        y = x[:, 0]*3-x[:, 1]+rng.normal(scale=0.1, size=len(x))
        prediction, details, messages = predict_sequential({'uv': x}, {'uv': evaluation}, y, ('uv',), 2)
        mean, sd = y.mean(), y.std(ddof=1)
        expected = PLSRegression(n_components=2, scale=False, max_iter=1000, tol=1e-7).fit(x, (y-mean)/sd).predict(evaluation).reshape(-1)*sd+mean
        np.testing.assert_allclose(prediction, expected, atol=1e-10)
        self.assertEqual(details['total_components_active'], 2)

    def test_complementary_block_recovers_unavailable_signal(self):
        rng = np.random.RandomState(101)
        latent = rng.normal(size=(100, 2))
        uv = latent[:, :1].dot(np.array([[1.0, 2.0, -1.0]]))
        nir = latent[:, :1].dot(np.array([[2.0, 0.1, -1.0]]))+latent[:, 1:].dot(np.array([[0.1, 2.0, 1.0]]))
        y = 2*latent[:, 0]+3*latent[:, 1]
        training = {'uv': uv[:70], 'nir': nir[:70]}
        evaluation = {'uv': uv[70:], 'nir': nir[70:]}
        single, unused, messages = predict_sequential({'uv': training['uv']}, {'uv': evaluation['uv']}, y[:70], ('uv',), 1)
        combined, details, messages = predict_sequential(training, evaluation, y[:70], ('uv', 'nir'), 1)
        self.assertLess(np.mean((combined-y[70:])**2), np.mean((single-y[70:])**2)*0.01)
        self.assertEqual(details['total_components_active'], 2)

    def test_duplicate_block_remains_finite(self):
        rng = np.random.RandomState(42)
        x = rng.normal(size=(25, 4))
        evaluation = rng.normal(size=(7, 4))
        y = x[:, 0]
        prediction, details, messages = predict_sequential({'uv': x, 'nir': x.copy()}, {'uv': evaluation, 'nir': evaluation.copy()}, y, ('uv', 'nir'), 4)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertLess(np.mean((prediction-evaluation[:, 0])**2), 1e-12)

    def test_constant_response_returns_training_mean(self):
        prediction, details, unused = predict_sequential({'uv': np.eye(5)}, {'uv': np.ones((2, 5))}, np.full(5, 9.0), ('uv',), 2)
        np.testing.assert_array_equal(prediction, [9, 9])
        self.assertTrue(details['constant_target'])


if __name__ == '__main__':
    unittest.main()
