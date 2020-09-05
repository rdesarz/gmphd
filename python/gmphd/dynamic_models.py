import numpy as np


class ConstantVelocity:
    def __init__(self, get_delta_t):
        self.get_delta_t = get_delta_t

    def __call__(self):
        return np.array([[1, 0., self.get_delta_t, 0.],
                         [0., 1., 0., self.get_delta_t],
                         [0., 0., 1., 0.],
                         [0., 0., 0., 1.]], dtype=np.float64)
