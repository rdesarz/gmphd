import numpy as np


class ConstantVelocityProcessNoise:
    def __init__(self, process_noise_std_dev, get_delta_t):
        self.get_delta_t = get_delta_t
        self.process_noise_std_dev = process_noise_std_dev

    def __call__(self):
        return np.array([[pow(self.get_delta_t, 4) / 4., 0., pow(self.get_delta_t, 3) / 2, 0.],
                         [0., pow(self.get_delta_t, 4) / 4, 0., pow(self.get_delta_t, 3) / 2],
                         [pow(self.get_delta_t, 3) / 2., 0., pow(self.get_delta_t, 2), 0.],
                         [0., pow(self.get_delta_t, 3) / 2, 0., pow(self.get_delta_t, 2)]], dtype=np.float64).dot(
            pow(self.process_noise_std_dev, 2))
