# Tests for singularity_eos.pte_solve (single-cell PTE solvers).
# Run with PYTHONPATH pointing at the directory holding the built module, e.g.
#   PYTHONPATH=build-python/python python test/python_pte.py -v
import unittest

import numpy as np
from numpy.testing import assert_allclose

import singularity_eos as se

SOLVERS = ["rhoT", "pt", "pt_analytic", "pt_cyclic", "cyclic_rhoe", "dual", "dual_kink"]


def ideal_pte_exact(gm1, cv, Y, rho, sie):
    """Closed form for ideal gases: e = Cv T, P = gm1 rho_m Cv T.
    T = e / sum(Y Cv); P = T sum(Y gm1 Cv) / tau_cell."""
    gm1, cv, Y = map(np.asarray, (gm1, cv, Y))
    T = sie / np.sum(Y * cv)
    P = T * np.sum(Y * gm1 * cv) * rho
    rho_m = P / (gm1 * cv * T)
    vfrac = Y * rho / rho_m
    return P, T, rho_m, vfrac


def numpy_pte(eos_params, Y, rho, sie):
    """Independent nested-bisection solve for ideal / stiffened gases.
    eos_params: list of (gm1, Cv, Pinf, q); rho_m(P,T) = (P+Pinf)/(gm1 Cv T),
    e_m = Cv T + Pinf/rho_m + q.  For fixed T the volume closure sum(Y/rho_m) = 1/rho is
    monotone in P; the outer bisection on T makes the mixture energy match."""
    Y = np.asarray(Y)
    tau = 1.0 / rho
    pmin = -min(pinf for (_, _, pinf, _) in eos_params)

    def p_of_T(T):
        lo, hi = pmin * (1 + 1e-12) + 1e-300, 1e30
        for _ in range(300):
            P = np.sqrt(abs(lo + 1e-30) * hi) if lo > 0 else 0.5 * (lo + hi)
            f = sum(y * g * cv * T / (P + pinf) for y, (g, cv, pinf, _) in zip(Y, eos_params)) - tau
            lo, hi = (P, hi) if f > 0 else (lo, P)
        return 0.5 * (lo + hi)

    def e_of_T(T, P):
        e = 0.0
        for y, (g, cv, pinf, q) in zip(Y, eos_params):
            r = (P + pinf) / (g * cv * T)
            e += y * (cv * T + pinf / r + q)
        return e

    lo, hi = 1e-3, 1e8
    for _ in range(300):
        T = np.sqrt(lo * hi)
        if e_of_T(T, p_of_T(T)) > sie:
            hi = T
        else:
            lo = T
    T = np.sqrt(lo * hi)
    return p_of_T(T), T


class TestPTE(unittest.TestCase):
    def check_equilibrium(self, eoss, res, rho, sie, Y, tol=1e-6):
        n = len(eoss)
        self.assertTrue(res.converged)
        vf, r, e = map(np.array, (res.vfrac, res.rho, res.sie))
        # volume closure, mass fractions, total energy
        self.assertAlmostEqual(vf.sum(), 1.0, delta=tol)
        assert_allclose(vf * r, np.asarray(Y) * rho, rtol=tol)
        assert_allclose(np.sum(vf * r * e) / rho, sie, rtol=tol)
        # common P and T, evaluated through the EOS itself
        P = [eoss[m].PressureFromDensityInternalEnergy(r[m], e[m]) for m in range(n)]
        T = [eoss[m].TemperatureFromDensityInternalEnergy(r[m], e[m]) for m in range(n)]
        assert_allclose(P, res.pressure, rtol=tol)
        assert_allclose(T, res.temperature, rtol=tol)

    def test_two_ideal_gases_closed_form(self):
        eoss = [se.IdealGas(0.4, 717.0), se.IdealGas(0.67, 3100.0)]
        Y = [0.3, 0.7]
        rho, sie = 1.3, 2.0e5
        P, T, rho_m, vfrac = ideal_pte_exact([0.4, 0.67], [717.0, 3100.0], Y, rho, sie)
        for solver in SOLVERS:
            with self.subTest(solver=solver):
                res = se.pte_solve(eoss, rho, sie, Y, solver=solver, tol=1e-12)
                self.assertTrue(res.converged, res)
                assert_allclose(res.pressure, P, rtol=1e-8)
                assert_allclose(res.temperature, T, rtol=1e-8)
                assert_allclose(res.rho, rho_m, rtol=1e-7)
                assert_allclose(res.vfrac, vfrac, rtol=1e-7)
                self.check_equilibrium(eoss, res, rho, sie, Y)

    def test_three_materials_ideal_and_stiff_vs_numpy(self):
        params = [(0.4, 717.0, 0.0, 0.0), (0.8, 1500.0, 3.0e7, 1.0e5),
                  (1.2, 4000.0, 6.0e7, -2.0e5)]
        eoss = [se.IdealGas(0.4, 717.0), se.StiffGas(0.8, 1500.0, 3.0e7, 1.0e5),
                se.StiffGas(1.2, 4000.0, 6.0e7, -2.0e5)]
        Y = [0.2, 0.5, 0.3]
        rho, sie = 20.0, 1.2e6
        P, T = numpy_pte(params, Y, rho, sie)
        for solver in SOLVERS:
            with self.subTest(solver=solver):
                res = se.pte_solve(eoss, rho, sie, Y, solver=solver, tol=1e-12)
                self.assertTrue(res.converged, res)
                assert_allclose(res.pressure, P, rtol=1e-7)
                assert_allclose(res.temperature, T, rtol=1e-7)
                self.check_equilibrium(eoss, res, rho, sie, Y)

    def test_mirrors_cpp_pte_ideal(self):
        # test/test_pte_ideal.cpp: 5 ideal gases IdealGas((m+1)/6, m+1), rho = sie = 1,
        # equal volume fractions; RhoT and PT solvers must agree.
        n = 5
        gm1 = [(m + 1) / (n + 1) for m in range(n)]
        cv = [float(m + 1) for m in range(n)]
        eoss = [se.IdealGas(g, c) for g, c in zip(gm1, cv)]
        Y = [1.0 / n] * n
        a = se.pte_solve(eoss, 1.0, 1.0, Y, solver="rhoT", tol=1e-14)
        b = se.pte_solve(eoss, 1.0, 1.0, Y, solver="pt", tol=1e-14)
        self.assertTrue(a.converged and b.converged)
        assert_allclose(a.rho, b.rho, atol=1e-5)
        P, T, rho_m, _ = ideal_pte_exact(gm1, cv, Y, 1.0, 1.0)
        assert_allclose(a.rho, rho_m, rtol=1e-8)
        assert_allclose(a.pressure, P, rtol=1e-8)
        assert_allclose(a.temperature, T, rtol=1e-8)

    def test_gruneisen_and_vinet_with_ideal_gas(self):
        gru = se.Gruneisen(3940.0, 1.489, 0.0, 0.0, 2.02, 0.47, 8.93, 298.0, 0.0, 3.9e6)
        vin = se.Vinet(8.93, 298.0, 1.4e12, 4.0, 0.0, 3.9e6, 0.0, 0.0, np.zeros(39))
        gas = se.IdealGas(0.4, 717.0)
        Y, rho = [0.99, 0.01], 4.0
        for solid in (gru, vin):
            sie = 0.5 * (solid.InternalEnergyFromDensityTemperature(8.93, 400.0)
                         + gas.InternalEnergyFromDensityTemperature(1e-3, 400.0))
            res = se.pte_solve([solid, gas], rho, sie, Y, solver="rhoT")
            self.check_equilibrium([solid, gas], res, rho, sie, Y, tol=1e-5)

    def test_errors(self):
        eos = [se.IdealGas(0.4, 717.0)]
        with self.assertRaises(ValueError):
            se.pte_solve(eos, 1.0, 1.0, [0.5, 0.5])
        with self.assertRaises(ValueError):
            se.pte_solve(eos, 1.0, 1.0, [1.0], solver="nope")
        with self.assertRaises(TypeError):
            se.pte_solve([object()], 1.0, 1.0, [1.0])


if __name__ == "__main__":
    unittest.main()
