//------------------------------------------------------------------------------
// Tests of the thermodynamically consistent Steinberg Gruneisen (solid-pool fork, D-A1-17):
// e = e_c(rho) + Cv T with e_c the 0 K isentrope of P(rho, e), and the bounded tensile
// branch. Gold values come from test/gruneisen_consistent_oracle.py (mpmath, independent
// quadrature of the isentrope ODE).
//------------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#ifndef CATCH_CONFIG_FAST_COMPILE
#define CATCH_CONFIG_FAST_COMPILE
#include <catch2/catch_test_macros.hpp>
#endif

#include <singularity-eos/base/constants.hpp>
#include <singularity-eos/eos/eos.hpp>

using singularity::Gruneisen;
using EOS = singularity::Variant<Gruneisen>;

namespace {
// Steinberg copper
constexpr Real C0 = 3.94e5, s1 = 1.489, s2 = 0, s3 = 0, G0 = 2.02, b = 0.47, rho0 = 8.93,
               T0 = 298.0, P0 = 0.0, Cv = 0.383e-5 * 1e12;
Gruneisen MakeCu() { return Gruneisen(C0, s1, s2, s3, G0, b, rho0, T0, P0, Cv); }

bool Close(const Real a, const Real g, const Real scale, const Real tol) {
  return std::abs(a - g) <= tol * std::max(std::abs(g), scale);
}

// Composite 8-point Gauss-Legendre of f over [a, b] with n panels
template <typename F>
Real Quad(const F &f, const Real a, const Real b, const int n) {
  constexpr Real x[4] = {0.1834346424956498049394761, 0.5255324099163289858177390,
                         0.7966664774136267395915539, 0.9602898564975362316835609};
  constexpr Real w[4] = {0.3626837833783619829651504, 0.3137066458778872873379622,
                         0.2223810344533744705443560, 0.1012285362903762591525314};
  Real sum = 0;
  const Real h = (b - a) / n;
  for (int k = 0; k < n; k++) {
    const Real c = a + (k + 0.5) * h, hh = 0.5 * h;
    for (int i = 0; i < 4; i++) sum += hh * w[i] * (f(c - hh * x[i]) + f(c + hh * x[i]));
  }
  return sum;
}

struct ColdGold {
  Real rho, e_c, p_c;
};
// gruneisen_consistent_oracle.py
constexpr ColdGold cold_gold[] = {
    {1e-3, 2.36310701197972938e+14, -2.36374979051314758e+11},
    {3.0, 4.43372073292954788e+10, -2.23610001521239563e+11},
    {6.0, 7.89098758698543644e+09, -2.10840767579850525e+11},
    {7.0, 2.97390161923949528e+09, -1.95538222261892731e+11},
    {8.0, -6.01425062315396518e+07, -1.32345868627665192e+11},
    {8.93, -1.14133999999999976e+09, -2.05881757239999962e+10},
    {9.5, -9.73749681308932185e+08, 7.63016346542524414e+10},
    {11.0, 2.45281523568327951e+09, 4.49274869984356201e+11},
    {14.0, 2.20553552823065300e+10, 1.83586281179976538e+12},
    {20.0, 1.17898380187406372e+11, 9.68173804816089648e+12}};
constexpr Real gold_K0 = 1.33987904692352002e+12, gold_rho_lim = 6.29581564441650698e+00,
               gold_p_lim = -2.09581654524158173e+11,
               gold_p_inf = -2.36379235462628571e+11;
} // namespace

TEST_CASE("Consistent Gruneisen cold curve matches the oracle", "[GruneisenEOS][gold]") {
  const Gruneisen eos = MakeCu();
  // K0 comes from a one-sided mpmath derivative: ~1e-11
  CHECK(Close(eos.ColdBulkModulusAtRho0(), gold_K0, 0, 1e-10));
  CHECK(Close(eos.RhoLim(), gold_rho_lim, 0, 1e-10));
  CHECK(Close(eos.KMin(), 0.02 * gold_K0, 0, 1e-10));
  CHECK(Close(eos.TensionLimit(), gold_p_inf, 0, 1e-10));
  CHECK(eos.MinimumPressure() == eos.TensionLimit());
  for (const auto &g : cold_gold) {
    INFO("rho " << g.rho);
    Real e_c, p_c;
    eos.ColdCurve(g.rho, e_c, p_c);
    CHECK(Close(e_c, g.e_c, 1e7, 1e-11));
    CHECK(Close(p_c, g.p_c, 1e8, 1e-11));
    // T = 0 is the cold curve
    CHECK(Close(eos.InternalEnergyFromDensityTemperature(g.rho, 0.0), e_c, 1e7, 1e-15));
    CHECK(Close(eos.PressureFromDensityTemperature(g.rho, 0.0), p_c, 1e8, 1e-11));
    CHECK(eos.MinInternalEnergyFromDensity(g.rho) == e_c);
  }
  // Reference state unchanged: e = 0 is T0 at rho0, P(rho0, 0) = P0
  CHECK(Close(eos.TemperatureFromDensityInternalEnergy(rho0, 0.0), T0, 0, 1e-14));
  CHECK(Close(eos.PressureFromDensityInternalEnergy(rho0, 0.0), P0, 1.0, 1e-14));
}

TEST_CASE("Consistent Gruneisen: e_c is the isentrope of P(rho, e)", "[GruneisenEOS]") {
  // e_c(r2) - e_c(r1) = int p_c / rho^2 to round-off, on each smooth branch
  const Gruneisen eos = MakeCu();
  const Real rl = eos.RhoLim();
  const std::vector<std::pair<Real, Real>> ranges = {
      {1e-6, 1e-3}, {1e-3, 0.5 * rl}, {0.5 * rl, rl}, {rl, 0.9 * rho0}, {0.9 * rho0, rho0},
      {rho0, 1.05 * rho0}, {1.05 * rho0, 12.0}, {12.0, 20.0}, {20.0, 26.0}};
  Real worst = 0;
  for (const auto &rg : ranges) {
    auto f = [&](const Real r) {
      Real e_c, p_c;
      eos.ColdCurve(r, e_c, p_c);
      return p_c / (r * r);
    };
    // p_c / rho^2 is steep at low rho: integrate in log rho
    const Real integral =
        Quad([&](const Real l) { return f(std::exp(l)) * std::exp(l); },
             std::log(rg.first), std::log(rg.second), 64);
    Real e1, e2, p;
    eos.ColdCurve(rg.first, e1, p);
    eos.ColdCurve(rg.second, e2, p);
    const Real err = std::abs((e2 - e1) - integral) /
                     std::max({std::abs(e1), std::abs(e2), Cv * T0});
    INFO("range " << rg.first << " " << rg.second << " err " << err);
    CHECK(err < 1e-13);
    worst = std::max(worst, err);
  }
  WARN("isentrope worst relative error " << worst);
  // p_c is C1 at rho0 and rho_lim
  for (const Real r : {rho0, rl}) {
    const Real h = 1e-7 * r;
    Real em, pm, ep, pp, e0, p0;
    eos.ColdCurve(r - h, em, pm);
    eos.ColdCurve(r, e0, p0);
    eos.ColdCurve(r + h, ep, pp);
    const Real left = (p0 - pm) / h, right = (pp - p0) / h;
    INFO("joint " << r << " left " << left << " right " << right);
    CHECK(Close(left, right, 0, 1e-5));
  }
}

TEST_CASE("Consistent Gruneisen: Maxwell relations", "[GruneisenEOS][Maxwell]") {
  const Gruneisen eos = MakeCu();
  const EOS veos = EOS(eos);
  Real worst_e = 0, worst_s = 0, worst_pt = 0;
  const Real rl = eos.RhoLim();
  const std::vector<std::pair<Real, Real>> ranges = {
      {0.2, 0.5 * rl}, {0.5 * rl, rl}, {rl, rho0}, {rho0, 12.0}, {12.0, 20.0}};
  for (const Real T : {1.0, 300.0, 3000.0, 3.0e4}) {
    for (const auto &rg : ranges) {
      // (de/drho)_T = (P - T (dP/dT)_rho) / rho^2, integrated over rho
      const Real dT = 1e-3 * T;
      auto dPdT = [&](const Real r) {
        return (veos.PressureFromDensityTemperature(r, T + dT) -
                veos.PressureFromDensityTemperature(r, T - dT)) /
               (2 * dT);
      };
      // P is linear in T at fixed rho: the central difference is exact up to round-off
      for (const Real r : {rg.first, rg.second}) {
        const Real exact = r * veos.GruneisenParamFromDensityTemperature(r, T) * Cv;
        const Real err = std::abs(dPdT(r) - exact) /
                         (std::abs(veos.PressureFromDensityTemperature(r, T)) / dT);
        worst_pt = std::max(worst_pt, err);
        CHECK(err < 1e-13);
      }
      auto integrand = [&](const Real r) {
        const Real exact_dPdT = r * veos.GruneisenParamFromDensityTemperature(r, T) * Cv;
        return (veos.PressureFromDensityTemperature(r, T) - T * exact_dPdT) / (r * r);
      };
      const Real de = veos.InternalEnergyFromDensityTemperature(rg.second, T) -
                      veos.InternalEnergyFromDensityTemperature(rg.first, T);
      const Real err_e =
          std::abs(de - Quad(integrand, rg.first, rg.second, 64)) /
          std::max(std::abs(veos.InternalEnergyFromDensityTemperature(rg.second, T)),
                   Cv * T0);
      worst_e = std::max(worst_e, err_e);
      INFO("T " << T << " range " << rg.first << " " << rg.second);
      CHECK(err_e < 1e-12);
      // (dS/drho)_T = -(dP/dT)_rho / rho^2
      const Real ds = veos.EntropyFromDensityTemperature(rg.second, T) -
                      veos.EntropyFromDensityTemperature(rg.first, T);
      const Real ids = -Quad(
          [&](const Real r) {
            return r * veos.GruneisenParamFromDensityTemperature(r, T) * Cv / (r * r);
          },
          rg.first, rg.second, 64);
      const Real err_s = std::abs(ds - ids) / std::max(std::abs(ds), Cv);
      worst_s = std::max(worst_s, err_s);
      CHECK(err_s < 1e-12);
    }
  }
  // (de/dS)_rho = T, (dS/dT)_rho = Cv/T
  for (const Real r : {1.0, 6.0, 8.0, 8.93, 10.0, 15.0}) {
    for (const Real T : {10.0, 300.0, 1e4}) {
      const Real e = veos.InternalEnergyFromDensityTemperature(r, T);
      CHECK(Close(veos.TemperatureFromDensityInternalEnergy(r, e), T, 0, 1e-9));
      const Real dT = 1e-4 * T;
      const Real dS = veos.EntropyFromDensityTemperature(r, T + dT) -
                      veos.EntropyFromDensityTemperature(r, T - dT);
      CHECK(Close(dS / (2 * dT), Cv / T, 0, 1e-7));
    }
  }
  WARN("Maxwell worst: (de/drho)_T " << worst_e << ", (dS/drho)_T " << worst_s
                                      << ", (dP/dT)_rho " << worst_pt);
}

TEST_CASE("Consistent Gruneisen: bounded tension and stability", "[GruneisenEOS]") {
  const Gruneisen eos = MakeCu();
  const Real pmin = eos.MinimumPressure();
  // Tension of order the copper spall/cohesive scale, not -rho0 C0^2
  CHECK(pmin > -0.3e12);
  CHECK(pmin < -0.1e12);
  int bad = 0;
  for (int i = 0; i <= 400; i++) {
    const Real r = 1e-8 * std::pow(0.99 * eos.MaximumDensity() / 1e-8, i / 400.0);
    for (const Real T : {0.0, 1.0, 300.0, 1e4}) {
      const Real P = eos.PressureFromDensityTemperature(r, T);
      const Real h = 1e-6 * r;
      const Real KT = r * (eos.PressureFromDensityTemperature(r + h, T) -
                           eos.PressureFromDensityTemperature(r - h, T)) /
                      (2 * h);
      const Real e = eos.InternalEnergyFromDensityTemperature(r, T);
      const Real BS = eos.BulkModulusFromDensityInternalEnergy(r, e);
      if (!(P >= pmin) || !(KT > 0) || !(BS > 0)) {
        bad++;
        UNSCOPED_INFO("rho " << r << " T " << T << " P " << P << " K_T " << KT << " B_S "
                             << BS);
      }
    }
  }
  CHECK(bad == 0);
  // The tail tends to the bound, and the thermal pressure vanishes with rho
  CHECK(Close(eos.PressureFromDensityTemperature(1e-10, 1e4), pmin, 0, 1e-9));
}

TEST_CASE("Consistent Gruneisen: Hugoniot is the Us-up fit", "[GruneisenEOS][Hugoniot]") {
  const Gruneisen eos = MakeCu();
  Real worst = 0, T_prev = T0;
  for (int i = 1; i <= 60; i++) {
    const Real up = 0.05e5 * i; // 0.05 .. 3 km/s
    const Real Us = C0 + s1 * up;
    const Real rho = rho0 * Us / (Us - up);
    const Real e = 0.5 * up * up; // from e = 0 (T0) at rho0, P0 = 0
    const Real P = eos.PressureFromDensityInternalEnergy(rho, e);
    const Real err = std::abs(P - rho0 * Us * up) / (rho0 * Us * up);
    worst = std::max(worst, err);
    CHECK(err < 1e-13);
    // Shock heating: T_H above T0 and rising with shock strength
    const Real T = eos.TemperatureFromDensityInternalEnergy(rho, e);
    CHECK(T > T_prev);
    T_prev = T;
    // Shock speed is above the upstream sound speed and the post-shock flow is subsonic
    const Real c = std::sqrt(eos.BulkModulusFromDensityInternalEnergy(rho, e) / rho);
    CHECK(Us - up < c);
  }
  WARN("Hugoniot worst relative error " << worst << ", T_H(3 km/s) " << T_prev);
}

TEST_CASE("Consistent Gruneisen: (P,T) entry points", "[GruneisenEOS][PT]") {
  const Gruneisen eos = MakeCu();
  const EOS veos = EOS(eos);
  for (const Real r : {1e-4, 0.5, 4.0, 6.0, 8.0, 8.95, 9.5, 12.0, 20.0}) {
    for (const Real T : {1.0, 300.0, 1e4}) {
      const Real P = veos.PressureFromDensityTemperature(r, T);
      Real rho, sie, drdP, drdT, dedP, dedT;
      veos.DensityEnergyDerivativesFromPressureTemperature(P, T, nullptr, rho, sie, drdP,
                                                           drdT, dedP, dedT);
      INFO("rho " << r << " T " << T << " P " << P << " rho_out " << rho);
      // The inversion is exact to round-off in P; rho to round-off times the condition
      // number |P| / (rho (dP/drho)_T), which is large in the flat tension tail.
      const Real cond = std::abs(P) * drdP / rho;
      CHECK(Close(veos.PressureFromDensityTemperature(rho, T), P, 1e8, 1e-14));
      CHECK(Close(rho, r, 0, 1e-14 * std::max(cond, 10.0)));
      CHECK(Close(veos.InternalEnergyFromDensityTemperature(rho, T),
                  veos.InternalEnergyFromDensityTemperature(r, T), Cv * T0,
                  1e-14 * std::max(cond, 100.0) * std::max(1.0, std::abs(sie) / (Cv * T0))));
      // finite differences of the inversion (relative density step 1e-6), where the
      // inversion is well conditioned
      if (cond < 1e4) {
        const Real dP = 1e-6 * rho / drdP, dT = 1e-4 * T;
        Real r1, e1, r2, e2;
        veos.DensityEnergyFromPressureTemperature(P + dP, T, nullptr, r1, e1);
        veos.DensityEnergyFromPressureTemperature(P - dP, T, nullptr, r2, e2);
        CHECK(Close((r1 - r2) / (2 * dP), drdP, 0, 1e-5));
        CHECK(Close((e1 - e2) / (2 * dP), dedP, 1e-12, 1e-5));
        veos.DensityEnergyFromPressureTemperature(P, T + dT, nullptr, r1, e1);
        veos.DensityEnergyFromPressureTemperature(P, T - dT, nullptr, r2, e2);
        CHECK(Close((r1 - r2) / (2 * dT), drdT, 1e-12, 1e-5));
        CHECK(Close((e1 - e2) / (2 * dT), dedT, 0, 1e-5));
      }
      // Maxwell in (P,T): (dh/dP)_T = v - T (dv/dT)_P, h = e + P v
      const Real v = 1 / rho, dvdP = -drdP / (rho * rho), dvdT = -drdT / (rho * rho);
      const Real lhs = dedP + P * dvdP + v;
      CHECK(Close(lhs, v - T * dvdT, v, 1e-10));
      // e(rho, P) closed form
      Real sie_p = 0;
      veos.InternalEnergyFromDensityPressure(r, P, sie_p, nullptr);
      CHECK(Close(sie_p, veos.InternalEnergyFromDensityTemperature(r, T), Cv * T0, 1e-9));
    }
  }
  // Pressure beyond the bounds is clamped, not an abort
  Real rho, sie;
  veos.DensityEnergyFromPressureTemperature(2 * eos.MinimumPressure(), 300.0, nullptr, rho,
                                            sie);
  CHECK(rho > 0);
  CHECK(rho < 1e-6);
  CHECK(eos.MaximumPressureAtTemperature(300.0) ==
        eos.PressureFromDensityTemperature(eos.MaximumDensity(), 300.0));
}
