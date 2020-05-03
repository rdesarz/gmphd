import numpy as np
from unittest import TestCase
from gmphd.gaussian_component import GaussianComponent
from gmphd.postprocessing import *

gaussian_mean = np.array([1., 1., 0.5, 0.5]).transpose()
gaussian_cov = np.array([[1., 0., 0., 0.],
                         [0., 1., 0., 0.],
                         [0., 0., 1., 0.],
                         [0., 0., 0., 1.]])
gaussian_weight = 1.
truncation_threshold = 0.1


class TestPostprocessing(TestCase):
    def test_pruning(self):
        posterior_intensity = list()
        posterior_intensity.append(GaussianComponent(gaussian_mean, gaussian_cov, gaussian_weight))
        posterior_intensity.append(GaussianComponent(gaussian_mean, gaussian_cov, truncation_threshold))
        posterior_intensity.append(GaussianComponent(gaussian_mean, gaussian_cov, 0.01))

        posterior_intensity = prune(posterior_intensity, truncation_threshold)

        self.assertEqual(len(posterior_intensity), 2)
        self.assertAlmostEqual(posterior_intensity[0].weight, 1.)
        self.assertAlmostEqual(posterior_intensity[1].weight, truncation_threshold)

