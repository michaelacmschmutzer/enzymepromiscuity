import unittest
import numpy as np
import promiscuity as pr
import warnings

"""Tests for the script in promiscuity.py"""

# suppress warnings
warnings.filterwarnings("ignore")

# Preparations (test cases etc)
synergism2d = np.array([
             [-3,  1,  0],
             [-1,  2, -1],
             [ 1, -3,  0],
             [ 2, -1, -1]
            ])

tradeoff2d = [ np.array([ [ 1,  4, -1] ]),
               np.array([ [ 4,  1, -1] ])
             ]

testfs = {(0, 1): 'unconstrained', (0, 2): 'weakantagonism', (1, 2): 'antagonism'}

def testnonlinobj(x, ignored):
    return -np.product(x)

class TestProm(unittest.TestCase):

    def test_solve_linear(self):
        sol, pos = pr.solve_linear([-1,-1], synergism2d)
        sol1, pos1 = pr.solve_linear([-1,1], synergism2d)
        self.assertAlmostEqual(sol, -2, places=6)
        self.assertAlmostEqual(pos[0], 1, places=6)
        self.assertAlmostEqual(pos[1], 1, places=6)
        self.assertAlmostEqual(sol1, -0.4, places=6)
        self.assertAlmostEqual(pos1[0], 0.6, places=6)
        self.assertAlmostEqual(pos1[1], 0.2, places=6)

    def test_solve_nonlinear(self):
        sol, pos = pr.solve_nonlinear(testnonlinobj, synergism2d)
        self.assertAlmostEqual(sol, -1, places=6)
        self.assertAlmostEqual(pos[0], 1, places=6)
        self.assertAlmostEqual(pos[1], 1, places=6)

    def test_variability_analysis(self):
        fit = -1
        point = np.array([0.659, 1.0, 0.0])
        constraints = np.array([[-1.,  0.,  0.,  0.],
                                [ 0., -1.,  0.,  0.],
                                [ 0.,  0., -1.,  0.],
                                [ 0.,  1.,  0., -1.],
                                [ 1.,  0.,  0., -1.],
                                [ 1.,  0.,  4., -4.],
                                [ 4.,  0.,  1., -4.],
                                [ 0.,  1.,  4., -1.]
                               ])
        objective = np.array([  0, -1,  0])
        ignored = [True, False, True]
        minmax = pr.variability_analysis(fit, point, constraints, objective=objective, problem="linear")
        self.assertAlmostEqual(minmax[0,0], 1.0, places=3)
        self.assertAlmostEqual(minmax[0,1], 0.0, places=3)
        self.assertAlmostEqual(minmax[1,0], 1.0, places=3)
        self.assertAlmostEqual(minmax[1,1], 1.0, places=3)
        self.assertAlmostEqual(minmax[2,0], 0.0, places=3)
        self.assertAlmostEqual(minmax[2,1], 0.0, places=3)
        minmax = pr.variability_analysis(fit, point, constraints, problem="nonlinear", ignored=ignored)
        self.assertAlmostEqual(minmax[0,0], 1.0, places=3)
        self.assertAlmostEqual(minmax[0,1], 0.0, places=3)
        self.assertAlmostEqual(minmax[1,0], 1.0, places=3)
        self.assertAlmostEqual(minmax[1,1], 1.0, places=3)
        self.assertAlmostEqual(minmax[2,0], 0.0, places=3)
        self.assertAlmostEqual(minmax[2,1], 0.0, places=3)

    def test_return_optimal(self):
        # Return only the solutions with larger fitness
        fit, pos = pr.return_optimal([-1,-0.5], [[0.2,0.3],[0.1,0.5]])
        self.assertTrue(len(fit)==1)
        self.assertTrue(len(pos)==1)
        self.assertAlmostEqual(fit[0], -1, places=6)
        self.assertAlmostEqual(pos[0][0], 0.2, places=6)
        self.assertAlmostEqual(pos[0][1], 0.3, places=6)
        fit, pos = pr.return_optimal([-1,-1], [[0.2,0.3],[0.1,0.5]])
        self.assertTrue(len(fit)==2)
        self.assertTrue(len(pos)==2)
        self.assertAlmostEqual(fit[0], -1, places=6)
        self.assertAlmostEqual(fit[1], -1, places=6)
        self.assertAlmostEqual(pos[0][0], 0.2, places=6)
        self.assertAlmostEqual(pos[0][1], 0.3, places=6)
        self.assertAlmostEqual(pos[1][0], 0.1, places=6)
        self.assertAlmostEqual(pos[1][1], 0.5, places=6)
        # Return equivalent solutions only once
        fit, pos = pr.return_optimal([-1,-1,-1], [[0.2,0.3], [0.2,0.3],[0.1,0.5]])
        self.assertTrue(len(fit)==2)
        self.assertTrue(len(pos)==2)
        self.assertAlmostEqual(fit[0], -1, places=6)
        self.assertAlmostEqual(fit[1], -1, places=6)
        self.assertAlmostEqual(pos[0][0], 0.2, places=6)
        self.assertAlmostEqual(pos[0][1], 0.3, places=6)
        self.assertAlmostEqual(pos[1][0], 0.1, places=6)
        self.assertAlmostEqual(pos[1][1], 0.5, places=6)

    def test_solve_concave(self):
        sol, pos = pr.solve_concave([[-1,-1], [-1,-1]], tradeoff2d, problem="linear")
        self.assertAlmostEqual(sol[0], -1, places=6)
        self.assertAlmostEqual(sol[1], -1, places=6)
        self.assertAlmostEqual(pos[0][0], 1, places=6)
        self.assertAlmostEqual(pos[0][1], 0, places=6)
        self.assertAlmostEqual(pos[1][0], 0, places=6)
        self.assertAlmostEqual(pos[1][1], 1, places=6)
        sol, pos = pr.solve_concave([testnonlinobj, testnonlinobj], tradeoff2d, problem="nonlinear")
        self.assertAlmostEqual(sol[0], -0.0625, places=6)
        self.assertAlmostEqual(sol[1], -0.0625, places=6)
        self.assertAlmostEqual(pos[0][0], 0.5, places=6)
        self.assertAlmostEqual(pos[0][1], 0.125, places=6)
        self.assertAlmostEqual(pos[1][0], 0.125, places=6)
        self.assertAlmostEqual(pos[1][1], 0.5, places=6)

if __name__ == "__main__":
    unittest.main()
