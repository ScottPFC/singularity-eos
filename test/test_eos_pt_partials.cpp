//------------------------------------------------------------------------------
// Exact (P,T) partials: DensityEnergyDerivativesFromPressureTemperature against
// Richardson central differences of DensityEnergyFromPressureTemperature, the Maxwell
// relation they must satisfy, and the (P,T) -> rho inversions they are built on.
//------------------------------------------------------------------------------

#include <cmath>
#include <limits>

#include <ports-of-call/portability.hpp>
#include <singularity-eos/base/indexable_types.hpp>
#include <singularity-eos/eos/eos.hpp>

#include <catch2/catch_test_macros.hpp>

using singularity::DavisProducts;
using singularity::DavisReactants;
using singularity::IdealGas;
using singularity::ScaledEOS;
using singularity::ShiftedEOS;
using singularity::StiffGas;
using singularity::SubMixtureEOS;
using singularity::UnitSystem;

namespace {

struct PT {
  Real rho, sie, drho_dP, drho_dT, de_dP, de_dT;
};

template <typename E, typename L>
PT Exact(const E &eos, const Real P, const Real T, L &&lambda) {
  PT r;
  eos.DensityEnergyDerivativesFromPressureTemperature(
      P, T, lambda, r.rho, r.sie, r.drho_dP, r.drho_dT, r.de_dP, r.de_dT);
  return r;
}

// Richardson-extrapolated central difference (h and h/2): O(h^4) truncation error.
template <typename F>
Real Richardson(const F &f, const Real x, const Real rel_h) {
  const Real h = rel_h * std::abs(x);
  const Real d1 = (f(x + h) - f(x - h)) / (2.0 * h);
  const Real d2 = (f(x + 0.5 * h) - f(x - 0.5 * h)) / h;
  return (4.0 * d2 - d1) / 3.0;
}

template <typename E, typename L>
PT Differenced(const E &eos, const Real P, const Real T, L &&lambda) {
  auto rho_at = [&](const Real p, const Real t) {
    Real rho, sie;
    eos.DensityEnergyFromPressureTemperature(p, t, lambda, rho, sie);
    return rho;
  };
  auto sie_at = [&](const Real p, const Real t) {
    Real rho, sie;
    eos.DensityEnergyFromPressureTemperature(p, t, lambda, rho, sie);
    return sie;
  };
  constexpr Real h = 1.0e-4;
  PT r;
  eos.DensityEnergyFromPressureTemperature(P, T, lambda, r.rho, r.sie);
  r.drho_dP = Richardson([&](Real p) { return rho_at(p, T); }, P, h);
  r.drho_dT = Richardson([&](Real t) { return rho_at(P, t); }, T, h);
  r.de_dP = Richardson([&](Real p) { return sie_at(p, T); }, P, h);
  r.de_dT = Richardson([&](Real t) { return sie_at(P, t); }, T, h);
  return r;
}

// |a - b| relative to a scale; `scale` covers derivatives that vanish exactly (ideal gas
// de/dP).
bool Close(const Real a, const Real b, const Real scale, const Real tol) {
  return std::abs(a - b) <= tol * scale;
}

// Maxwell: (de/dP)_T = -T (dtau/dT)_P - P (dtau/dP)_T with tau = 1/rho, i.e.
//   rho^2 de_dP = T drho_dT + P drho_dP.
// The residual is scaled by the two terms' magnitudes, so a consistent EOS reports
// round-off.
Real MaxwellResidual(const PT &r, const Real P, const Real T) {
  const Real lhs = r.rho * r.rho * r.de_dP;
  const Real rhs = T * r.drho_dT + P * r.drho_dP;
  const Real scale = std::abs(T * r.drho_dT) + std::abs(P * r.drho_dP) + std::abs(lhs);
  return std::abs(lhs - rhs) / scale;
}

template <typename E, typename L>
void CheckPartials(const E &eos, const Real P, const Real T, L &&lambda,
                   const Real fd_tol) {
  const PT x = Exact(eos, P, T, lambda);
  const PT d = Differenced(eos, P, T, lambda);
  INFO("P = " << P << " T = " << T << " rho = " << x.rho << " sie = " << x.sie);
  CHECK(Close(x.rho, d.rho, std::abs(d.rho), 1.0e-13));
  CHECK(Close(x.sie, d.sie, std::abs(d.sie), 1.0e-13));
  INFO("drho_dP " << x.drho_dP << " vs " << d.drho_dP);
  CHECK(Close(x.drho_dP, d.drho_dP, std::abs(d.drho_dP), fd_tol));
  INFO("drho_dT " << x.drho_dT << " vs " << d.drho_dT);
  CHECK(Close(x.drho_dT, d.drho_dT, std::abs(d.drho_dT), fd_tol));
  // de/dP is identically 0 for an ideal gas: judge it against the c_v-sized scale T de_dT
  // / P.
  const Real e_scale = std::abs(d.de_dP) + std::abs(T * d.de_dT / P);
  INFO("de_dP " << x.de_dP << " vs " << d.de_dP);
  CHECK(Close(x.de_dP, d.de_dP, e_scale, fd_tol));
  INFO("de_dT " << x.de_dT << " vs " << d.de_dT);
  CHECK(Close(x.de_dT, d.de_dT, std::abs(d.de_dT), fd_tol));
  INFO("Maxwell residual " << MaxwellResidual(x, P, T));
  CHECK(MaxwellResidual(x, P, T) < 1.0e-10);
}

// Clayton et al. (arXiv:2606.27726) Table 6.1 gases and Composition B reactants (Z = 0)
// in cgs, and the PBX 9501-type products of test_eos_davis.cpp.  The stiffened gas is
// probed only above its P_inf = 1e8 barye: below it rho(P) is dominated by P_inf and a
// relative-step difference in P cancels away the digits it is meant to measure.
constexpr Real kGPa = 1.0e10;
constexpr Real kKJg = 1.0e10;
IdealGas Air() { return IdealGas(0.0014 / 0.001 - 1.0, 0.001 * kKJg); }
StiffGas Water() {
  return StiffGas(0.0085 / 0.0012 - 1.0, 0.0012 * kKJg, 0.01 * kGPa, 0.0);
}
DavisReactants CompB(const Real Z = 0.0, const Real P0 = 0.0) {
  return DavisReactants(1.71, 0.0, P0, 298.15, 2.434e5, 2.367, 1.024, 0.8437, Z, 0.8484,
                        0.001072 * kKJg);
}
DavisProducts Products() {
  return DavisProducts(0.798311, 0.58, 1.35, 2.66182, 0.75419, 3.2e10, 0.001072e10);
}

} // namespace

SCENARIO("Exact (P,T) partials match differences and satisfy Maxwell", "[PTPartials]") {
  singularity::NullIndexer lambda;
  GIVEN("An ideal gas") {
    const auto eos = Air();
    for (Real P : {1.0e2, 1.0e6, 1.0e11})
      for (Real T : {1.0, 300.0, 3.0e4})
        CheckPartials(eos, P, T, lambda, 1.0e-8);
  }
  GIVEN("A stiffened gas") {
    const auto eos = Water();
    for (Real P : {1.0e9, 1.0e10, 1.0e12})
      for (Real T : {1.0, 300.0, 3.0e4})
        CheckPartials(eos, P, T, lambda, 1.0e-8);
  }
  GIVEN("Davis reactants, on both the compressed and the expanded branch") {
    const auto eos = CompB();
    for (Real P : {1.0e7, 1.0e9, 1.0e11})
      for (Real T : {300.0, 1500.0})
        CheckPartials(eos, P, T, lambda, 1.0e-7);
  }
  GIVEN("Davis products") {
    const auto eos = Products();
    for (Real P : {1.0e8, 1.0e10, 1.0e11})
      for (Real T : {1000.0, 4000.0})
        CheckPartials(eos, P, T, lambda, 1.0e-7);
  }
  GIVEN("The modifiers, by the chain rule through the wrapped EOS") {
    const auto scaled = ScaledEOS<StiffGas>(Water(), 1.0 / 1.489499);
    const auto shifted = ShiftedEOS<StiffGas>(Water(), 3.0e9);
    const auto scaled_shifted =
        ScaledEOS<ShiftedEOS<StiffGas>>(ShiftedEOS<StiffGas>(Water(), -2.0e9), 0.6);
    // SI-like code units: rho in kg/m^3, sie in J/kg, T in K.
    const auto units = UnitSystem<StiffGas>(Water(), 1.0e-3, 1.0e4, 1.0);
    for (Real P : {1.0e9, 1.0e11}) {
      for (Real T : {50.0, 3000.0}) {
        CheckPartials(scaled, P, T, lambda, 1.0e-8);
        CheckPartials(shifted, P, T, lambda, 1.0e-8);
        CheckPartials(scaled_shifted, P, T, lambda, 1.0e-8);
        CheckPartials(units, P, T, lambda, 1.0e-8);
      }
    }
  }
  GIVEN("A sub-mixture group reading its scale from the lambda") {
    // The D+T group as FLASH builds it: base scale 1 (D) and 1/1.489499 (T), S_eff from
    // the cell.
    const Real s_t = 1.0 / 1.489499;
    const auto group = SubMixtureEOS<StiffGas>(Water(), 1.0, 0, s_t, 1.0);
    Real lam[1];
    for (Real s_eff : {1.0, 0.4 + 0.6 * s_t, s_t}) {
      lam[0] = s_eff;
      for (Real P : {1.0e9, 1.0e11})
        for (Real T : {50.0, 3000.0})
          CheckPartials(group, P, T, lam, 1.0e-8);
    }
  }
}

SCENARIO("Davis (P,T) inversions round-trip to round-off", "[PTPartials][Davis]") {
  // Judged on RhoTPartials' P(rho, T), the function the inversion roots, to within a few
  // ulps of rho: near rho0 one ulp of density moves P by ~1e-8 of a 1 kbar state, so a
  // relative test in P alone would ask for more than the arithmetic holds.  The (rho, e)
  // route, PressureFromDensityTemperature, cancels e - Es and resolves even less on a
  // dilute state.
  singularity::NullIndexer lambda;
  constexpr Real kEps = std::numeric_limits<Real>::epsilon();
  GIVEN("Davis reactants and products") {
    const auto r = CompB();
    const auto p = Products();
    // The T = 3e4 states at low P are hot and dilute, far below the 1e-5 g/cc floor of
    // the fixed root-find window the inversion used to have.
    for (Real P : {1.0e3, 1.0e8, 1.0e11}) {
      for (Real T : {300.0, 3.0e4}) {
        Real rho, sie, Pr, P_rho, P_T, e_rho, e_T;
        r.DensityEnergyFromPressureTemperature(P, T, lambda, rho, sie);
        r.RhoTPartials(rho, T, Pr, P_rho, P_T, e_rho, e_T);
        INFO("reactants P = " << P << " T = " << T << " rho = " << rho
                              << " P(rho,T) = " << Pr);
        CHECK(std::abs(Pr - P) <= 1.0e-13 * P + 4.0 * kEps * rho * std::abs(P_rho));
        p.DensityEnergyFromPressureTemperature(P, T, lambda, rho, sie);
        p.RhoTPartials(rho, T, Pr, P_rho, P_T, e_rho, e_T);
        INFO("products P = " << P << " T = " << T << " rho = " << rho
                             << " P(rho,T) = " << Pr);
        CHECK(std::abs(Pr - P) <= 1.0e-13 * P + 4.0 * kEps * rho * std::abs(P_rho));
      }
    }
  }
}
