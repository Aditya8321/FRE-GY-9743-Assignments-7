"""Independent shape, integral, and library-integration checks for HW7."""
import unittest
import warnings

import numpy as np
from numpy.testing import assert_allclose

from fixedincomelib import qfCreate1DInterpolator, qfInterpolate1D, qfInterpolate1DIntegral
from fixedincomelib.utilities.numerics import Interpolator1DMonotoneConvex, ExtrapMethod


def make(x, y):
    return Interpolator1DMonotoneConvex(x, y, ExtrapMethod.FLAT)


def quadrature(curve, a, b):
    """Gauss-Legendre quadrature, split at every knot (no primitive reuse)."""
    if b < a:
        return -quadrature(curve, b, a)
    edges = [a, *[x for x in curve.axis1 if a < x < b], b]
    z, w = np.polynomial.legendre.leggauss(8)
    result = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        points = (lo + hi) / 2 + (hi - lo) * z / 2
        result += (hi - lo) / 2 * np.dot(w, [curve.interpolate(t) for t in points])
    return result


class MonotoneConvexTests(unittest.TestCase):
    def test_known_quadratic_and_integral(self):
        # This data set independently produces f(x)=x^2 on [0,2].
        c = make([0, 1, 2], [0, 1, 4])
        for x in np.linspace(0, 2, 31):
            self.assertAlmostEqual(c.interpolate(x), x*x, places=13)
        self.assertAlmostEqual(c.integrate(0.2, 1.7), (1.7**3 - 0.2**3)/3, places=13)
        self.assertAlmostEqual(c.integrate(-1, 3), 8/3 + 4, places=13)

    def test_decreasing_convex(self):
        c = make([0, 1, 2], [4, 1, 0])
        for x in np.linspace(0, 2, 31):
            self.assertAlmostEqual(c.interpolate(x), (2-x)**2, places=13)
        self.assertAlmostEqual(c.integrate(0, 2), 8/3, places=13)

    def test_one_node_constant_everywhere(self):
        c = make([2.5], [0.0005])
        for x in [-100, 0, 2.5, 10, 100]:
            self.assertEqual(c.interpolate(x), 0.0005)
        for a, b in [(-10, 12), (4, -3), (2, 2), (5, 9)]:
            self.assertAlmostEqual(c.integrate(a, b), 0.0005*(b-a), places=14)

    def test_two_nodes_and_flat_extrapolation(self):
        c = make([1, 3], [0.02, 0.04])
        assert_allclose([c.interpolate(x) for x in [-1, 1, 2, 3, 5]],
                        [0.02, 0.02, 0.03, 0.04, 0.04], rtol=0, atol=1e-15)
        self.assertAlmostEqual(c.integrate(-1, 5), 0.18, places=14)

    def test_equal_slopes_and_plateau(self):
        for y in [[0, 1, 2], [0.02, 0.02, 0.02], [0, 0, 1], [1, 0, 0]]:
            c = make([0, 1, 2], y)
            self.assertTrue(c.shape_compatible)
            grid = np.linspace(0, 2, 401)
            v = np.array([c.interpolate(t) for t in grid])
            self.assertGreaterEqual(np.min(np.diff(v, n=2)), -1e-14)
            self.assertGreaterEqual(np.min(v), min(y)-1e-14)
            self.assertLessEqual(np.max(v), max(y)+1e-14)

    def test_irregular_grid_exact_nodes_and_continuity(self):
        x = np.array([0.2, 0.7, 1.8, 2, 4.7])
        y = np.array([0.01, 0.011, 0.015, 0.017, 0.05])
        c = make(x, y)
        assert_allclose([c.interpolate(t) for t in x], y, rtol=0, atol=1e-15)
        for t in x:
            self.assertAlmostEqual(c.interpolate(t-1e-10), c.interpolate(t+1e-10), places=10)

    def test_random_compatible_curves_shape_and_no_overshoot(self):
        rng = np.random.default_rng(9743)
        for direction in [1, -1]:
            for _ in range(60):
                x = np.cumsum(rng.uniform(0.03, 2, 9))
                s = np.sort(direction * rng.uniform(0, 0.07, 8))
                y = np.r_[0.01, 0.01 + np.cumsum(s*np.diff(x))]
                c = make(x, y)
                self.assertTrue(c.shape_compatible)
                # Test the actual sampled curve, including cross-knot convexity.
                grid = np.linspace(x[0], x[-1], 401)
                v = np.array([c.interpolate(t) for t in grid])
                self.assertGreaterEqual(np.min(direction*np.diff(v)), -1e-13)
                self.assertGreaterEqual(np.min(np.diff(v, n=2)), -1e-13)
                for i in range(len(x)-1):
                    local = [c.interpolate(t) for t in np.linspace(x[i], x[i+1], 11)]
                    self.assertGreaterEqual(min(local), min(y[i:i+2])-1e-13)
                    self.assertLessEqual(max(local), max(y[i:i+2])+1e-13)
                # Arbitrary Jensen inequalities, not only grid second differences.
                a, b = rng.uniform(x[0], x[-1], (2, 30))
                lam = rng.uniform(0, 1, 30)
                for p, q, w in zip(a, b, lam):
                    self.assertLessEqual(c.interpolate(w*p+(1-w)*q),
                                         w*c.interpolate(p)+(1-w)*c.interpolate(q)+1e-13)

    def test_integral_against_independent_quadrature(self):
        c = make([0.2, 0.7, 1.8, 2, 4.7], [0.01, 0.011, 0.015, 0.017, 0.05])
        rng = np.random.default_rng(7)
        for a, b in rng.uniform(-2, 7, (80, 2)):
            self.assertAlmostEqual(c.integrate(a, b), quadrature(c, a, b), places=13)
            m = (a+b)/2
            self.assertAlmostEqual(c.integrate(a,b), c.integrate(a,m)+c.integrate(m,b), places=13)
            self.assertEqual(c.integrate(a,b), -c.integrate(b,a))
        self.assertEqual(c.integrate(2,2), 0)

    def test_integral_derivative(self):
        c = make([0, 1, 2, 4], [0.02, 0.021, 0.023, 0.031])
        for t in [-1, 0.3, 1, 1.5, 2, 3.5, 5]:
            eps = 1e-5
            derivative = (c.integrate(0,t+eps)-c.integrate(0,t-eps))/(2*eps)
            self.assertAlmostEqual(derivative, c.interpolate(t), places=8)

    def test_incompatible_data_preserved_with_explicit_warning(self):
        for y in [[0, 2, 3], [0, 2, 1], [2, 0, 1]]:
            with self.assertWarnsRegex(RuntimeWarning, 'piecewise-linear fallback'):
                c = make([0, 1, 2], y)
            self.assertFalse(c.shape_compatible)
            self.assertIsNotNone(c.fallback_reason)
            for t in np.linspace(-1, 3, 31):
                self.assertAlmostEqual(c.interpolate(t), np.interp(t, [0,1,2], y))
            self.assertAlmostEqual(c.integrate(-1,3), quadrature(c,-1,3), places=13)

    def test_negative_ifrs_not_clipped(self):
        c = make([0, 1, 2], [-0.03, -0.02, 0.01])
        self.assertTrue(c.shape_compatible)
        self.assertEqual(c.interpolate(0), -0.03)
        self.assertGreater(np.exp(-c.integrate(0,1)), 1)

    def test_invalid_inputs(self):
        cases = [([], []), ([0,0], [1,2]), ([1,0], [1,2]),
                 ([0,1], [1]), ([[0,1]], [[1,2]]),
                 ([0,np.nan], [1,2]), ([0,1], [1,np.inf])]
        for x, y in cases:
            with self.subTest(x=x,y=y), self.assertRaises(ValueError):
                make(x,y)
        with self.assertRaisesRegex(ValueError, 'FLAT'):
            Interpolator1DMonotoneConvex([0,1],[1,2],ExtrapMethod.LINEAR)
        c = make([0,1],[1,2])
        for bad in [np.nan, np.inf, -np.inf, [0,1]]:
            with self.assertRaises(ValueError): c.interpolate(bad)
            with self.assertRaises(ValueError): c.integrate(0,bad)

    def test_input_copies(self):
        x, y = np.array([0.,1.,2.]), np.array([0.,1.,4.])
        c = make(x,y)
        x[1], y[1] = 100, 100
        self.assertEqual(c.interpolate(1), 1)

    def test_public_factory_and_optional_risk_contract(self):
        c = qfCreate1DInterpolator([0,1,2], [0,1,4], 'MONOTONE_CONVEX', 'FLAT')
        self.assertIsInstance(c, Interpolator1DMonotoneConvex)
        self.assertEqual(qfInterpolate1D(1.5,c), 2.25)
        self.assertAlmostEqual(qfInterpolate1DIntegral(0,2,c), 8/3)
        with self.assertRaises(NotImplementedError): c.gradient_wrt_ordinate(1)
        with self.assertRaises(NotImplementedError): c.gradient_of_integrated_value_wrt_ordinate(0,2)


if __name__ == '__main__':
    unittest.main()
