from unittest import TestCase
import numpy as np

from gmphd.gaussian_component import GaussianComponent
from gmphd.update import *
import gmphd.clutter_model as clutter_model

# Parameters for the tests
gaussian_mean = np.array([1., 1., 0.5, 0.5]).transpose()
gaussian_cov = np.array([[1., 0., 0., 0.],
                         [0., 1., 0., 0.],
                         [0., 0., 1., 0.],
                         [0., 0., 0., 1.]])
gaussian_weight = 1.
prob_detection = 0.95
measurement_model = np.array([[1., 0., 0., 0.],
                              [0., 1., 0., 0.]])
measurement_noise = np.array([[1., 0.],
                              [0., 1.]])
measurement_1 = np.array([2., 2.]).transpose()
measurement_2 = np.array([1., 1.]).transpose()
measurements = [measurement_1, measurement_2]

clutter_model = clutter_model.Constant(1.)


class TestUpdate(TestCase):
    def test_non_detected_update_components(self):
        gaussian_components = list()
        gaussian_components.append(GaussianComponent(gaussian_mean, gaussian_cov, gaussian_weight))

        gaussian_components = compute_non_detection_intensity(gaussian_components, prob_detection)

        self.assertTrue(np.allclose(gaussian_components[0].mean, [1, 1, 0.5, 0.5]))
        self.assertTrue(np.allclose(gaussian_components[0].covariance, np.array([[1., 0., 0., 0.],
                                                                                 [0., 1., 0., 0.],
                                                                                 [0., 0., 1., 0.],
                                                                                 [0., 0., 0., 1.]])))
        self.assertAlmostEqual(gaussian_components[0].weight, 1 - prob_detection)

    def test_compute_kalman_gain(self):
        component = GaussianComponent(gaussian_mean, gaussian_cov, gaussian_weight)

        innovation_covariance = compute_innovation_cov(component, measurement_model, measurement_noise)

        kalman_gain = compute_kalman_gain(component, measurement_model, innovation_covariance)

        self.assertTrue(np.allclose(kalman_gain, np.array([[0.5, 0.],
                                                           [0., 0.5],
                                                           [0., 0.],
                                                           [0., 0.]])))

    def test_compute_measurement_estimate(self):
        component = GaussianComponent(gaussian_mean, gaussian_cov, gaussian_weight)

        measurement_estimate = compute_meas_estimate(component, measurement_model)

        self.assertTrue(np.allclose(measurement_estimate, np.array([1., 1.]).transpose()))

    def test_compute_innovation_covariance(self):
        component = GaussianComponent(gaussian_mean, gaussian_cov, gaussian_weight)

        innovation_cov = compute_innovation_cov(component, measurement_model, measurement_noise)

        self.assertTrue(np.allclose(innovation_cov, np.array([[2., 0.],
                                                              [0., 2.]])))

    def test_compute_updated_covariance(self):
        component = GaussianComponent(gaussian_mean, gaussian_cov, gaussian_weight)
        kalman_gain = compute_kalman_gain(component, measurement_model, measurement_noise)

        updated_cov = compute_updated_gauss_cov(component, measurement_model, kalman_gain)

        self.assertTrue(np.allclose(updated_cov, np.array([[0., 0., 0., 0.],
                                                           [0., 0., 0., 0.],
                                                           [0., 0., 1., 0.],
                                                           [0., 0., 0., 1.]])))

    def test_compute_updated_gaussian(self):
        component = GaussianComponent(gaussian_mean, gaussian_cov, gaussian_weight)
        update_components = compute_update_components(component, measurement_model, measurement_noise)

        component = compute_updated_gaussian(component, measurement_1, update_components, prob_detection)

        self.assertTrue(np.allclose(component.mean, [1.5, 1.5, 0.5, 0.5]))
        self.assertTrue(np.allclose(component.covariance, update_components.updated_cov))
        self.assertAlmostEqual(component.weight,
                               1. * prob_detection * multivariate_normal(update_components.meas_estimate,
                                                                         update_components.innovation_cov).pdf(
                                   measurement_1))

    def test_compute_updated_intensity(self):
        intensity = list()
        intensity.append(GaussianComponent(gaussian_mean, gaussian_cov, gaussian_weight))

        intensity = update(intensity, measurements, measurement_model, measurement_noise, prob_detection, clutter_model)
