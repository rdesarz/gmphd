from unittest import TestCase
from gmphd.birth_model import *


class TestBirthModel(TestCase):
    def test_side_fov_birth_model(self):
        birth_model = SidesOfFovBirthModel(5.0, 6.0)

        birth_intensity = birth_model()

        self.assertEqual(len(birth_intensity), 12)
        self.assertTrue(np.allclose(birth_intensity[0].mean, [2.5, 0., -0.2, 0.1]))
        self.assertTrue(np.allclose(birth_intensity[1].mean, [2.5, 1., -0.2, 0.1]))
        self.assertTrue(np.allclose(birth_intensity[6].mean, [-2.5, 0., 0.2, 0.1]))

    def test_side_fov_birth_model_each_half_meter(self):
        birth_model = SidesOfFovBirthModel(5.0, 6.0, 0.5)

        birth_intensity = birth_model()

        self.assertEqual(len(birth_intensity), 24)
        self.assertTrue(np.allclose(birth_intensity[0].mean, [2.5, 0., -0.2, 0.1]))
        self.assertTrue(np.allclose(birth_intensity[1].mean, [2.5, 0.5, -0.2, 0.1]))
        self.assertTrue(np.allclose(birth_intensity[12].mean, [-2.5, 0., 0.2, 0.1]))
