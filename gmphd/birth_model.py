import numpy as np
from numpy import float64
from gmphd.gaussian_component import GaussianComponent


class BirthModel:
    def __init__(self, width, depth):
        self.width = width
        self.depth = depth

    def __call__(self):
        birth_intensity = list()
        mean = np.array([5., 5., 0.5, 0.5], dtype=float64).transpose()
        covariance = np.array([[1., 0., 0., 0.],
                               [0., 1., 0., 0.],
                               [0., 0., 1., 0.],
                               [0., 0., 0., 1.]], dtype=float64)
        weight = float64(0.1)
        birth_intensity.append(GaussianComponent(mean=mean, covariance=covariance, weight=weight))
        return birth_intensity
