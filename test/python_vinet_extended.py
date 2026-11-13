# Compares singularity_eos.Vinet (with the solid-pool extension) to the A0 oracle
# solidpool_oracle.eos.Vinet over a (rho, e) grid spanning the regular, tension-extension
# and cold-floor regions. Needs the oracle importable (PYTHONPATH containing
# phases/a0-oracle) and numpy; skipped otherwise. Run with
#   PYTHONPATH=build-python/python:<a0-oracle> python test/python_vinet_extended.py -v
import unittest

import numpy as np

import singularity_eos as se

try:
    from pfc.sim.solidpool_oracle import Vinet as OracleVinet
except ImportError:  # pragma: no cover
    OracleVinet = None

MBCC = 1e12
CU = dict(rho0=8.93, T0=298.0, B0=1.3448466 * MBCC, BP0=4.956, A0=5.19245e-05,
          Cv0=0.383e-05 * MBCC, E0=0.0, S0=5.05e-04 * MBCC)
TOL = 1e-12


def rel(a, b, scale=0.0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), scale)))


@unittest.skipIf(OracleVinet is None, "solidpool_oracle not importable")
class TestVinetExtended(unittest.TestCase):
    def build(self, d):
        dd = np.zeros(39)
        dd[: len(d)] = d
        ref = OracleVinet(**CU, d=d)
        eos = se.Vinet(*[CU[k] for k in ("rho0", "T0", "B0", "BP0", "A0", "Cv0", "E0", "S0")],
                       dd, ref.k_min, ref.t_floor)
        return ref, eos

    def grid(self, ref):
        rho = ref.rho_lim * np.array([0.1, 0.3, 0.6, 0.9, 0.999, 1.0001, 1.01, 1.1])
        rho = np.concatenate([rho, [7.0, 8.0, 8.93, 9.5, 11.0]])
        tf = ref.t_floor
        temps = np.array([0.01, 0.1, 0.5, 0.99]) * tf
        temps = np.concatenate([temps, [1.5 * tf, 5.0, 50.0, 298.0, 900.0, 5000.0]])
        r, t = np.meshgrid(rho, temps, indexing="ij")
        r, t = r.ravel().copy(), t.ravel().copy()
        e = np.asarray(ref.internal_energy(r, t))
        return r, t, e

    def test_series_and_copper(self):
        for d in ([], [0.3, -0.2, 0.05]):
            ref, eos = self.build(d)
            self.assertLess(abs(eos.rho_lim / ref.rho_lim - 1), 1e-14)
            r, t, e = self.grid(ref)
            n = r.size
            out = np.zeros(n)
            kw = dict(rtol=0)
            # the oracle's in_extension flags; the grid must contain every region
            tension = ref.in_tension_extension(r)
            cold = ref.in_cold_extension(r, e)
            self.assertTrue(tension.any() and cold.any() and (~tension & ~cold).any())
            self.assertTrue((tension & cold).any())

            eos.TemperatureFromDensityInternalEnergy(r, e, out)
            # T = T0 + (e - e_ref)/Cv0 cancels e_ref (up to 1e12 erg/g) against e, so the
            # 1e-16 rounding differences between libm and numpy in e_ref are amplified for
            # T << e/Cv0: measure T against max(T, |e|/Cv0).
            t_ref = np.asarray(ref.temperature(r, e))
            err_t = float(np.max(np.abs(out - t_ref)
                                 / np.maximum(np.abs(t_ref), np.abs(e) / CU["Cv0"])))
            eos.PressureFromDensityInternalEnergy(r, e, out)
            p_ref = np.asarray(ref.pressure(r, e))
            # p crosses zero: scale by the larger of |p| and k_min-sized pressure noise
            err_p = rel(out, p_ref, scale=1e-6 * ref.B0)
            eos.PressureFromDensityTemperature(r, t, out)
            err_pt = rel(out, p_ref, scale=1e-6 * ref.B0)
            eos.InternalEnergyFromDensityTemperature(r, t, out)
            err_e = rel(out, e, scale=CU["Cv0"] * CU["T0"])
            eos.BulkModulusFromDensityInternalEnergy(r, e, out)
            err_b = rel(out, ref.isentropic_bulk_modulus(r, e))
            eos.PressureFromDensityInternalEnergy(r, e, out)
            eos.InternalEnergyFromDensityPressure(r, p_ref, out)
            ok = np.isfinite(ref.internal_energy_from_pressure(r, p_ref))
            # (rho, p) -> e is ill-conditioned for T << t_floor: T = T0 + (p - p_ref)/(A0 B0)
            # cancels p_ref ~ 1e11 against a few-mK thermal pressure, so rounding of p_ref
            # (1e-16 relative) is amplified. Check those states (T < 0.6 t_floor) at 1e-8. Energies are measured
            # against max(|e|, Cv0 T0), the model's energy unit.
            e_p_ref = np.asarray(ref.internal_energy_from_pressure(r, p_ref))
            deep = t < 0.6 * ref.t_floor
            err_ep = rel(out[ok & ~deep], e_p_ref[ok & ~deep],
                         scale=CU["Cv0"] * CU["T0"])
            err_ep_deep = rel(out[ok & deep], e_p_ref[ok & deep],
                              scale=CU["Cv0"] * CU["T0"])
            self.assertLess(err_ep_deep, 1e-8)
            print(f"   deep-cold e(p) max rel err {err_ep_deep:.2e}")
            print(f"d={d}: max rel err T {err_t:.2e} p(e) {err_p:.2e} p(T) {err_pt:.2e} "
                  f"e(T) {err_e:.2e} B_S {err_b:.2e} e(p) {err_ep:.2e}")
            for name, err in (("T", err_t), ("p", err_p), ("p_T", err_pt), ("e", err_e),
                              ("B_S", err_b), ("e_p", err_ep)):
                self.assertLess(err, TOL, name)

    def test_plain_constructor_unchanged(self):
        eos = se.Vinet(*[CU[k] for k in ("rho0", "T0", "B0", "BP0", "A0", "Cv0", "E0", "S0")],
                       np.zeros(39))
        self.assertEqual(eos.rho_lim, 0.0)
        self.assertEqual(eos.k_min, 0.0)

    def test_bad_parameters(self):
        args = [CU[k] for k in ("rho0", "T0", "B0", "BP0", "A0", "Cv0", "E0", "S0")]
        with self.assertRaises(ValueError):
            se.Vinet(*args, np.zeros(39), 2.0 * CU["B0"], 0.1)
        with self.assertRaises(ValueError):
            se.Vinet(*args, np.zeros(39), -1.0, 0.1)


if __name__ == "__main__":
    unittest.main()
