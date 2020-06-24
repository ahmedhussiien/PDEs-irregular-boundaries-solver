import unittest
from PDE_solver.app import PDE_irregular_boundaries

class PDE_solver_tests(unittest.TestCase):

    def test_regular_boundaries(self):
        boundary_points = {(0.0, 0) : 62.5, (0.6, 0) : 50, (1.2, 0): 50, (1.8, 0): 50, (2.4, 0): 75, (0, 0.6) : 75, (2.4, 0.6): 100, (0, 1.2) : 75, (2.4, 1.2): 100,
        (0, 1.8) : 75, (2.4, 1.8): 100, (0, 2.4) : 75, (2.4, 2.4): 100, (0, 3) : 187.5, (0.6, 3) : 300, (1.2, 3): 300, (1.8, 3): 300, (2.4, 3): 200}

        Results = PDE_irregular_boundaries(boundary_points = boundary_points, h = 0.6, k = 0.6, uyy_coeff_fn = "1",
                                    uxx_coeff_fn = "1", eqn_fn = "0")

        assert Results == {(0.6, 0.6): 73.892, (1.2, 0.6): 77.544, (1.8, 0.6): 82.983, (0.6, 2.4): 173.355, (1.2, 2.4): 198.512, (1.8, 2.4): 182.446, (0.6, 1.8): 119.907, (1.2, 1.8): 138.248, (1.8, 1.8): 131.271, (0.6, 1.2): 93.025, (1.2, 1.2): 103.302, (1.8, 1.2): 104.389}

    def test_irregular_boundaries(self):
        boundary_points = {(-1, 0.0): 4, (-0.5, 0.0): 8, (0.0,0.0): 8, (0.5, 0.0): 8, (1.0, 0.0): 8, (1.5, 0.0): 8, (2.0, 0.0): 4,
                   (-0.833, 0.5): 0, (1.5, 0.5) : 0, 
                   (-0.6667, 1.0):0, (1.0,1.0): 0, 
                   (-0.5, 1.5): 2, (0.0, 1.5): 4, (0.5, 1.5): 2}

        Results = PDE_irregular_boundaries(boundary_points = boundary_points, h = 0.5, k = 0.5, uyy_coeff_fn = "1", 
                                    uxx_coeff_fn = "1", eqn_fn = "2")

        assert Results == {(-0.5, 0.5): 2.739, (0.0, 0.5): 4.296, (0.5, 0.5): 4.209, (1.0, 0.5): 2.927, (-0.5, 1.0): 1.043, (0.0, 1.0): 2.738,(0.5, 1.0): 2.112}

    def test_irregular_boundaries_2(self):
        boundary_points = {(0.5, 0.6) : 62.5, (0.6, 0.6) : 50, (1.2, 0.6): 50, (1.8, 0.6): 50, (2.2, 0.6): 75, (0.5, 1.2) : 75, (2.2, 1.2): 100, (0.5, 1.8) : 75, (2.2, 1.8): 100, (0.5, 2.4) : 75, (2.2, 2.4): 100, (0.5, 2.8) : 187.5, (0.6, 2.8) : 2.800, (1.2, 2.8): 2.800, (1.8, 2.8): 2.800, (2.2, 2.8): 200}

        Results = PDE_irregular_boundaries(boundary_points = boundary_points, h = 0.6, k = 0.6, uyy_coeff_fn = "1", 
                                    uxx_coeff_fn = "1", eqn_fn = "0")

        assert Results == {(0.6, 1.2): 71.906, (1.2, 1.2): 65.778, (1.8, 1.2): 77.318,(0.6, 2.4): 62.092, (1.2, 2.4): 39.618, (1.8, 2.4): 54.295, (0.6, 1.8): 72.496, (1.2, 1.8): 63.887, (1.8, 1.8): 77.655}
 
if __name__ == '__main__':
    unittest.main()