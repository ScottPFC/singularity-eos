//------------------------------------------------------------------------------
// Tests of the solid-pool Vinet extension (tension continuation below the spinodal and
// cold-floor entropy continuation). Gold values in vinet_extended_gold.hpp come from
// solidpool_oracle.eos.Vinet (see test/vinet_extended_gold.hpp header).
//------------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <limits>
#ifndef CATCH_CONFIG_FAST_COMPILE
#define CATCH_CONFIG_FAST_COMPILE
#include <catch2/catch_test_macros.hpp>
#endif

#include <singularity-eos/base/constants.hpp>
#include <singularity-eos/eos/eos.hpp>
#include <test/vinet_extended_gold.hpp>

using singularity::Vinet;
using EOS = singularity::Variant<Vinet>;

namespace {
constexpr Real Mbcc = 1e12;
constexpr Real rho0 = 8.93, T0 = 298.0, B0 = 1.3448466 * Mbcc, BP0 = 4.956,
               A0 = 5.19245e-05, Cv0 = 0.383e-05 * Mbcc, E0 = 0.0, S0 = 5.05e-04 * Mbcc;
const Real d_zero[39] = {};
Real d_series[39] = {0.3, -0.2, 0.05};

Vinet MakeCu(const bool series) {
  return Vinet(rho0, T0, B0, BP0, A0, Cv0, E0, S0, series ? d_series : d_zero,
               0.02 * B0, 1.0e-3 * T0);
}
// |a - g| <= tol * max(|g|, scale)
bool Close(const Real a, const Real g, const Real scale, const Real tol = 1e-11) {
  return std::abs(a - g) <= tol * std::max(std::abs(g), scale);
}
Real maxerr = 0.0, maxerr_cold = 0.0;
bool CloseTrack(const Real a, const Real g, const Real scale, const Real tol = 1e-11) {
  Real &m = (tol > 1e-11) ? maxerr_cold : maxerr;
  m = std::max(m, std::abs(a - g) / std::max(std::abs(g), scale));
  return Close(a, g, scale, tol);
}
} // namespace

TEST_CASE("Vinet extension matches the oracle", "[VinetExtended][gold]") {
  using namespace vinet_ext_gold;
  for (const bool series : {false, true}) {
    const Vinet host = MakeCu(series);
    const EOS eos = EOS(host);
    const Gold *g = series ? cuseries : cu;
    const int n = series ? sizeof(cuseries) / sizeof(Gold) : sizeof(cu) / sizeof(Gold);
    REQUIRE(CloseTrack(host.RhoLim(), series ? cuseries_rho_lim : cu_rho_lim, 1.0, 1e-13));
    REQUIRE(host.KMin() == (series ? cuseries_k_min : cu_k_min));
    REQUIRE(host.TFloor() == (series ? cuseries_t_floor : cu_t_floor));
    for (int i = 0; i < n; i++) {
      INFO("series=" << series << " state " << i << " rho=" << g[i].rho);
      const Real rho = g[i].rho;
      // T = T0 + (e - e_ref)/Cv0 cancels e_ref (up to 1e12), amplifying 1e-16 libm
      // differences from the oracle: measure T against max(T, |e|/Cv0).
      const Real pscale = 1e-4 * B0, escale = 1e-3 * Cv0 * T0;
      CHECK(CloseTrack(eos.TemperatureFromDensityInternalEnergy(rho, g[i].e), g[i].T,
                       std::abs(g[i].e) / Cv0));
      CHECK(CloseTrack(eos.PressureFromDensityInternalEnergy(rho, g[i].e), g[i].p,
                       pscale));
      CHECK(CloseTrack(eos.EntropyFromDensityInternalEnergy(rho, g[i].e), g[i].s, S0));
      CHECK(CloseTrack(eos.BulkModulusFromDensityInternalEnergy(rho, g[i].e), g[i].bs,
                       pscale));
      // Quantities that depend on T^-1 or T^-2 (and e(p), which cancels p_ref against the
      // thermal pressure) amplify the oracle/libm rounding differences in e_ref for
      // T < t_floor, so those get a looser tolerance.
      const Real tc = (g[i].T_in < host.TFloor()) ? 1e-8 : 1e-11;
      CHECK(CloseTrack(eos.SpecificHeatFromDensityInternalEnergy(rho, g[i].e), g[i].cv,
                       Cv0 * 1e-6, tc));
      CHECK(CloseTrack(eos.InternalEnergyFromDensityTemperature(rho, g[i].T_in),
                       g[i].e_from_T, escale));
      CHECK(CloseTrack(eos.EntropyFromDensityTemperature(rho, g[i].T_in), g[i].s_from_T,
                       S0));
      CHECK(CloseTrack(eos.PressureFromDensityTemperature(rho, g[i].T_in), g[i].p,
                       pscale));
      Real sie_guess = 0.0;
      eos.InternalEnergyFromDensityPressure(rho, g[i].p, sie_guess);
      CHECK(CloseTrack(sie_guess, g[i].e_from_p, escale, tc));
      // FillEos with (rho, e) input
      Real r = rho, T = 0, e = g[i].e, P = 0, cv = 0, bmod = 0;
      eos.FillEos(r, T, e, P, cv, bmod,
                  singularity::thermalqs::temperature | singularity::thermalqs::pressure |
                      singularity::thermalqs::specific_heat |
                      singularity::thermalqs::bulk_modulus);
      CHECK(CloseTrack(T, g[i].T, std::abs(g[i].e) / Cv0));
      CHECK(CloseTrack(P, g[i].p, pscale));
      CHECK(CloseTrack(cv, g[i].cv, Cv0 * 1e-6, tc));
      CHECK(CloseTrack(bmod, g[i].bs, pscale));
      // Gruneisen: V dp/de = A0 B0/(rho Cv)
      CHECK(CloseTrack(eos.GruneisenParamFromDensityInternalEnergy(rho, g[i].e),
                       A0 * B0 / (rho * g[i].cv), 1e-3, tc));
    }
  }
  printf("Vinet extension vs oracle: max rel err %.2e (T >= t_floor), %.2e (cold floor)\n",
         maxerr, maxerr_cold);
  CHECK(maxerr < 1e-11);
  CHECK(maxerr_cold < 1e-8);
}

TEST_CASE("Vinet extension is off by default and for zero parameters",
          "[VinetExtended]") {
  const Vinet plain(rho0, T0, B0, BP0, A0, Cv0, E0, S0, d_zero);
  const Vinet off(rho0, T0, B0, BP0, A0, Cv0, E0, S0, d_zero, 0.0, 0.0);
  REQUIRE(plain.RhoLim() == 0.0);
  for (const Real rho : {8.0, 9.5}) {
    for (const Real e : {2.0e8, 4.6e9}) {
      CHECK(plain.PressureFromDensityInternalEnergy(rho, e) ==
            off.PressureFromDensityInternalEnergy(rho, e));
      CHECK(plain.EntropyFromDensityInternalEnergy(rho, e) ==
            off.EntropyFromDensityInternalEnergy(rho, e));
    }
  }
  // Extension changes nothing in the regular region
  const Vinet ext = MakeCu(false);
  for (const Real rho : {7.0, 8.0, 9.5}) {
    const Real e = plain.InternalEnergyFromDensityTemperature(rho, 500.0);
    CHECK(Close(ext.PressureFromDensityInternalEnergy(rho, e),
                plain.PressureFromDensityInternalEnergy(rho, e), 1e-4 * B0, 1e-13));
    CHECK(Close(ext.EntropyFromDensityInternalEnergy(rho, e),
                plain.EntropyFromDensityInternalEnergy(rho, e), S0, 1e-13));
  }
  CHECK_THROWS(Vinet(rho0, T0, B0, BP0, A0, Cv0, E0, S0, d_zero, 2.0 * B0, 0.0));
}

TEST_CASE("Vinet extension: spinodal and C1/C2 joint", "[VinetExtended]") {
  const Vinet eos = MakeCu(false);
  const Real rl = eos.RhoLim();
  const Real T = 450.0;
  // K_T reaches k_min at rho_lim and is flat (= k_min) beyond
  CHECK(Close(eos.TBulkModulusFromDensityTemperature(rl * (1 + 1e-12), T), eos.KMin(),
              1.0, 1e-9));
  for (const Real f : {0.99, 0.5, 0.1}) {
    CHECK(Close(eos.TBulkModulusFromDensityTemperature(rl * f, T), eos.KMin(), 1.0, 1e-12));
  }
  // the spinodal does not depend on T
  CHECK(eos.TBulkModulusFromDensityTemperature(rl * 1.0001, 10.0) ==
        eos.TBulkModulusFromDensityTemperature(rl * 1.0001, 3000.0));
  // p continuous, dp/drho continuous (C1), e, de/drho, d2e/drho2 continuous (C2)
  const Real h = 1e-7 * rl;
  auto p = [&](Real rho) { return eos.PressureFromDensityTemperature(rho, T); };
  auto e = [&](Real rho) { return eos.InternalEnergyFromDensityTemperature(rho, T); };
  CHECK(Close(p(rl - 1e-12 * rl), p(rl + 1e-12 * rl), B0, 1e-10));
  CHECK(Close(e(rl - 1e-12 * rl), e(rl + 1e-12 * rl), Cv0 * T0, 1e-10));
  const Real dp_lo = (p(rl) - p(rl - h)) / h, dp_hi = (p(rl + h) - p(rl)) / h;
  CHECK(std::abs(dp_lo - dp_hi) < 1e-5 * std::abs(dp_hi));
  // 1st derivative of e at fixed T: (p - T A0 B0)/rho^2 (thermodynamic identity)
  const Real de_lo = (e(rl) - e(rl - h)) / h, de_hi = (e(rl + h) - e(rl)) / h;
  CHECK(std::abs(de_lo - de_hi) < 1e-5 * std::abs(de_hi));
  const Real h2 = 1e-3 * rl;
  const Real d2_lo = (e(rl) - 2 * e(rl - h2) + e(rl - 2 * h2)) / (h2 * h2);
  const Real d2_hi = (e(rl + 2 * h2) - 2 * e(rl + h2) + e(rl)) / (h2 * h2);
  // C2 across the joint: the one-sided second differences differ only by O(h) terms
  CHECK(std::abs(d2_lo - d2_hi) < 5e-2 * std::abs(d2_hi));
  // p keeps falling through the extension (monotone, finite)
  CHECK(p(0.5 * rl) < p(rl));
  CHECK(p(0.1 * rl) < p(0.5 * rl));
  // the extension is analytic: p = p_lim + k_min ln(rho/rho_lim) + A0 B0 (T - T0)
  CHECK(Close(p(0.4 * rl),
              vinet_ext_gold::cu_p_lim + eos.KMin() * std::log(0.4) + A0 * B0 * (T - T0),
              B0, 1e-12));
}

TEST_CASE("Vinet extension: thermodynamic consistency", "[VinetExtended]") {
  for (const bool series : {false, true}) {
    const Vinet eos = MakeCu(series);
    const Real rl = eos.RhoLim(), tf = eos.TFloor();
    struct S {
      Real rho, T;
    };
    for (const S s : {S{10.0, 800.0}, S{8.5, 300.0}, S{rl * 1.01, 100.0},
                      S{rl * 0.6, 700.0}, S{rl * 0.3, 20.0}, S{8.5, 0.5 * tf},
                      S{rl * 0.7, 0.2 * tf}, S{9.2, 0.05 * tf}}) {
      INFO("series=" << series << " rho=" << s.rho << " T=" << s.T);
      const Real rho = s.rho;
      const Real e = eos.InternalEnergyFromDensityTemperature(rho, s.T);
      auto T = [&](Real r, Real ee) { return eos.TemperatureFromDensityInternalEnergy(r, ee); };
      auto P = [&](Real r, Real ee) { return eos.PressureFromDensityInternalEnergy(r, ee); };
      auto Sf = [&](Real r, Real ee) { return eos.EntropyFromDensityInternalEnergy(r, ee); };
      CHECK(Close(T(rho, e), s.T, 1e-6, 1e-12));
      const Real Tc = T(rho, e), p = P(rho, e);
      // (ds/de)_rho = 1/T and (ds/dv)_e = p/T. Step in e scaled by the local heat capacity.
      const Real de = 1e-5 * eos.SpecificHeatFromDensityInternalEnergy(rho, e) * Tc;
      const Real dsde = (Sf(rho, e + de) - Sf(rho, e - de)) / (2 * de);
      CHECK(std::abs(dsde * Tc - 1.0) < 1e-6);
      const Real v = 1.0 / rho, dv = 1e-6 * v;
      const Real dsdv =
          (Sf(1.0 / (v + dv), e) - Sf(1.0 / (v - dv), e)) / (2 * dv);
      CHECK(std::abs(dsdv * Tc / p - 1.0) < 1e-5);
      // isentropic bulk modulus: rho c^2 = rho (dp/drho)_s, with de = p/rho^2 drho
      const Real hr = 1e-5 * rho;
      auto along = [&](Real r) {
        // integrate e along the isentrope with RK2 from (rho, e)
        const Real dr = r - rho;
        const Real k1 = P(rho, e) / (rho * rho);
        const Real em = e + 0.5 * dr * k1;
        const Real k2 = P(rho + 0.5 * dr, em) / ((rho + 0.5 * dr) * (rho + 0.5 * dr));
        return P(r, e + dr * k2);
      };
      const Real dpdrho_s = (along(rho + hr) - along(rho - hr)) / (2 * hr);
      const Real bs = eos.BulkModulusFromDensityInternalEnergy(rho, e);
      CHECK(std::abs(rho * dpdrho_s / bs - 1.0) < 1e-5);
      CHECK(bs > 0.0);
      CHECK(Tc > 0.0);
      // entropy concavity in e: s'' = -1/(T^2 Cv) < 0
      CHECK(Sf(rho, e + de) + Sf(rho, e - de) - 2 * Sf(rho, e) < 0.0);
      // (rho,T) and (rho,e) paths agree on the (rho,p) inversion
      CHECK(Close(eos.InternalEnergyFromDensityPressure(rho, p), e, Cv0 * T0, 1e-9));
    }
  }
}

TEST_CASE("Vinet extension: cold floor", "[VinetExtended]") {
  const Vinet eos = MakeCu(false);
  const Real tf = eos.TFloor();
  const Real rho = 9.0;
  const Real ef = eos.InternalEnergyFromDensityTemperature(rho, tf);
  // T continuous (C1 in e) across the floor
  CHECK(Close(eos.TemperatureFromDensityInternalEnergy(rho, ef * (1 + 1e-9) + 1.0), tf, 1.0,
              1e-6));
  // very cold energies keep T > 0 and s finite, decreasing in e
  Real last = std::numeric_limits<Real>::infinity();
  for (const Real du : {1e3, 1e5, 1e7, 1e9, 1e11}) {
    const Real T = eos.TemperatureFromDensityInternalEnergy(rho, ef - du);
    const Real s = eos.EntropyFromDensityInternalEnergy(rho, ef - du);
    CHECK(T > 0.0);
    CHECK(T < last);
    CHECK(std::isfinite(s));
    last = T;
  }
  // s and T are C1 at the floor: s(e) slope 1/tf on both sides
  const Real de = 1e-6 * Cv0 * tf;
  const Real s_lo = eos.EntropyFromDensityInternalEnergy(rho, ef - de);
  const Real s_hi = eos.EntropyFromDensityInternalEnergy(rho, ef + de);
  CHECK(std::abs((s_hi - s_lo) / (2 * de) * tf - 1.0) < 1e-5);
  // Cv is reduced below the floor, Cv0 above it
  CHECK(eos.SpecificHeatFromDensityTemperature(rho, 2 * tf) == Cv0);
  CHECK(Close(eos.SpecificHeatFromDensityTemperature(rho, 0.5 * tf), 4.0 * Cv0, 1.0, 1e-14));
  // below the T = 0 pressure floor there is no energy
  const Real pmin = eos.PressureFromDensityTemperature(rho, 0.0);
  CHECK(std::isnan(eos.InternalEnergyFromDensityPressure(rho, pmin - 1e9)));
}
