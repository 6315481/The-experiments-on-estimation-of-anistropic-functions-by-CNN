import unittest
import numpy as np
from numpy.testing import assert_array_equal
from functions import anistrophic_decrease, mix_decrease

class test_functions(unittest.TestCase):

    def test_anistrophic(self):
        sigma = np.array([1, 2])
        num = 10
        dim = 2
        X = np.random.uniform(size=[num, dim])
        phi_0 = np.sqrt(2) * np.cos(2 * np.pi * X[:, 0])
        phi_1 = np.sqrt(2) * np.cos(2 * np.pi * X[:, 1])

        y = phi_0 / (2 ** sigma[0]) + phi_0 * phi_1 / (2 ** sigma[0] + 2 ** sigma[1])
        assert_array_equal(y, anistrophic_decrease(X, sigma))

    def test_mix(self):
        sigma = np.array([1, 2])
        num = 10
        dim = 2
        X = np.random.uniform(size=[num, dim])
        phi_0 = np.sqrt(2) * np.cos(2 * np.pi * X[:, 0])
        phi_1 = np.sqrt(2) * np.cos(2 * np.pi * X[:, 1])

        y = phi_0 / (2 ** sigma[0]) + phi_0 * phi_1 / (2 ** (sigma[0] + sigma[1]))
        assert_array_equal(y, mix_decrease(X, sigma))

    