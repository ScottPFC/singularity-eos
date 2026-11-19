//------------------------------------------------------------------------------
// © 2021-2025. Triad National Security, LLC. All rights reserved.  This
// program was produced under U.S. Government contract 89233218CNA000001
// for Los Alamos National Laboratory (LANL), which is operated by Triad
// National Security, LLC for the U.S.  Department of Energy/National
// Nuclear Security Administration. All rights in the program are
// reserved by Triad National Security, LLC, and the U.S. Department of
// Energy/National Nuclear Security Administration. The Government is
// granted for itself and others acting on its behalf a nonexclusive,
// paid-up, irrevocable worldwide license in this material to reproduce,
// prepare derivative works, distribute copies to the public, perform
// publicly and display publicly, and to permit others to do so.
//------------------------------------------------------------------------------

#ifndef _SINGULARITY_EOS_EOS_GRUNEISEN_HPP_
#define _SINGULARITY_EOS_EOS_GRUNEISEN_HPP_

#include <cmath>
#include <cstdio>

#include <limits>

#include <singularity-eos/base/constants.hpp>
#include <singularity-eos/base/math_utils.hpp>
#include <singularity-eos/base/robust_utils.hpp>
#include <singularity-eos/base/root-finding-1d/root_finding.hpp>
#include <singularity-eos/eos/eos_base.hpp>

namespace singularity {

using namespace eos_base;

// COMMENT: This is meant to be an implementation of the Steinberg version of
// the Gruneisen EOS which should correspond to eostype(3) in xRAGE and
// /[...]/eos/gruneisen in FLAG
//
// THERMODYNAMICALLY CONSISTENT COMPLETION (solid-pool fork, D-A1-17). The upstream model
// completed P(rho, e) with T = T0 + e / Cv, so (de/drho)_T = 0 where Maxwell requires
// (P - T (dP/dT)_rho) / rho^2. Here the model is a Mie-Gruneisen solid with a constant Cv:
//
//   P(rho, e) = p_c(rho) + rho Gamma(rho) (e - e_c(rho)),   e(rho, T) = e_c(rho) + Cv T,
//
// where e_c is the 0 K isentrope, de_c/drho = p_c / rho^2, through e_c(rho0) = -Cv T0 (so
// e = 0 is T = T0 at rho0, as before). Gamma(rho) is the upstream Steinberg Gamma.
//  * Compression (rho >= rho0): p_c = P_H + rho Gamma (e_c - E_H), so P(rho, e) is the
//    upstream P_H + rho Gamma (e - E_H) unchanged (same expression). With rho Gamma =
//    G0 rho0 + b (rho - rho0) the isentrope ODE is linear with integrating factor
//    theta(eta) = (1 - eta)^-b exp((G0 - b) eta), eta = 1 - rho0/rho:
//      e_c = theta(eta) [e_c(rho0) + int_0^eta (P_H - rho Gamma E_H) / (rho0 theta) deta'].
//    The integral is stored at nodes clustered toward the Hugoniot pole and completed by
//    8-point Gauss-Legendre from the nearest node (relative error ~1e-15).
//  * Expansion (rho < rho0): the upstream branch P_H = P0 + C0^2 (rho - rho0), E_H = 0 has
//    no spinodal: its isentrope bottoms at about -rho0 C0^2 / (1 + G0) (-0.46 TPa for Cu).
//    The 0 K curve is instead a Vinet (Rose) curve anchored at rho0,
//      p_c = p_c0 + 3 K0 (1 - x) x^-2 exp(eta_v (1 - x)),  x = (rho0/rho)^(1/3),
//    with K0 the compression-side 0 K bulk modulus at rho0 (p_c is C1 at rho0) and
//    K0' = max(4 s1 - 1, 2) (eta_v = 1.5 (K0' - 1)). As for Vinet (solid-pool extension), below
//    rho_lim, where rho dp_c/drho falls to k_min (default 0.02 K0), the curve continues
//    with p C1 and e C2 at the joint; here with rho dp_c/drho = k_min rho / rho_lim, so
//      p_c = p_inf + k_min rho / rho_lim,   p_inf = p_lim - k_min,
//    the tension is bounded below by p_inf (MinimumPressure) and the cold sound speed is
//    constant. Gamma = G0, so the thermal pressure G0 rho Cv T vanishes as rho -> 0.
// Above rho_max every quantity is evaluated at rho_max (as upstream).
// Entropy: S = Cv ln(T/T0) - Cv ln theta(rho) (S = 0 at rho0, T0), theta(rho) = (rho/rho0)^G0
// in expansion.
class Gruneisen : public EosBase<Gruneisen> {
 public:
  Gruneisen() = default;
  PORTABLE_INLINE_FUNCTION
  Gruneisen(const Real C0, const Real s1, const Real s2, const Real s3, const Real G0,
            const Real b, const Real rho0, const Real T0, const Real P0, const Real Cv,
            const Real rho_max,
            const MeanAtomicProperties &AZbar = MeanAtomicProperties())
      : _C0(C0), _s1(s1), _s2(s2), _s3(s3), _G0(G0), _b(b), _rho0(rho0), _T0(T0), _P0(P0),
        _Cv(Cv), _rho_max(rho_max), _AZbar(AZbar) {
    // Warn user when provided rho_max is greater than the computed rho_max
#ifndef NDEBUG
    const Real computed_rho_max = ComputeRhoMax(s1, s2, s3, rho0);
    if (rho_max > RHOMAX_SAFETY * computed_rho_max) {
      printf(
          "WARNING: Provided rho_max, %g, is greater than the computed rho_max of %g.\n",
          rho_max, computed_rho_max);
      printf("         States beyond %g g/cm^3 are unphysical (i.e. imaginary sound "
             "speeds).\n",
             computed_rho_max);
    }
#endif
    InitializeColdCurve(-1.0);
  }
  // Constructor when rho_max isn't specified automatically determines _rho_max
  PORTABLE_INLINE_FUNCTION
  Gruneisen(const Real C0, const Real s1, const Real s2, const Real s3, const Real G0,
            const Real b, const Real rho0, const Real T0, const Real P0, const Real Cv,
            const MeanAtomicProperties &AZbar = MeanAtomicProperties())
      : _C0(C0), _s1(s1), _s2(s2), _s3(s3), _G0(G0), _b(b), _rho0(rho0), _T0(T0), _P0(P0),
        _Cv(Cv), _rho_max(RHOMAX_SAFETY * ComputeRhoMax(s1, s2, s3, rho0)),
        _AZbar(AZbar) {
    InitializeColdCurve(-1.0);
  }
  // Solid-pool constructor with the tension parameter: k_min in (0, K0) sets where the
  // expansion cold curve leaves the Vinet shape (rho dp_c/drho = k_min); k_min < 0 selects
  // the default 0.02 K0. rho_max <= 0 selects the computed rho_max.
  PORTABLE_INLINE_FUNCTION
  Gruneisen(const Real C0, const Real s1, const Real s2, const Real s3, const Real G0,
            const Real b, const Real rho0, const Real T0, const Real P0, const Real Cv,
            const Real rho_max, const Real k_min,
            const MeanAtomicProperties &AZbar = MeanAtomicProperties())
      : _C0(C0), _s1(s1), _s2(s2), _s3(s3), _G0(G0), _b(b), _rho0(rho0), _T0(T0), _P0(P0),
        _Cv(Cv),
        _rho_max(rho_max > 0 ? rho_max : RHOMAX_SAFETY * ComputeRhoMax(s1, s2, s3, rho0)),
        _AZbar(AZbar) {
    InitializeColdCurve(k_min);
  }
  static PORTABLE_INLINE_FUNCTION Real ComputeRhoMax(const Real s1, const Real s2,
                                                     const Real s3, const Real rho0);
  PORTABLE_INLINE_FUNCTION
  void CheckParams() const {
    PORTABLE_ALWAYS_REQUIRE(_T0 >= 0, "Non-negative reference temperature required");
    PORTABLE_ALWAYS_REQUIRE(_rho0 >= 0, "Non-negative reference density required");
    PORTABLE_ALWAYS_REQUIRE(_C0 >= 0, "Non-negative Hugoniot intercept required");
    PORTABLE_ALWAYS_REQUIRE(_Cv >= 0, "Non-negative heat capacity required");
    PORTABLE_ALWAYS_REQUIRE(_rho_max > _rho0,
                            "Maximum density must be greater than reference");
    PORTABLE_ALWAYS_REQUIRE(_K0 > 0, "Positive 0 K bulk modulus at rho0 required");
    PORTABLE_ALWAYS_REQUIRE(_k_min > 0 && _k_min < _K0, "0 < k_min < K0 required");
    _AZbar.CheckParams();
  }
  PORTABLE_INLINE_FUNCTION Real
  MaxStableDensityAtTemperature(const Real temperature) const;
  Gruneisen GetOnDevice() { return *this; }

  // Cold (0 K isentrope) curve and its parameters
  PORTABLE_FORCEINLINE_FUNCTION Real KMin() const { return _k_min; }
  PORTABLE_FORCEINLINE_FUNCTION Real RhoLim() const { return _rho_lim; }
  PORTABLE_FORCEINLINE_FUNCTION Real ColdBulkModulusAtRho0() const { return _K0; }
  // Lower bound of the pressure for T >= 0: the cold curve's limit as rho -> 0
  PORTABLE_FORCEINLINE_FUNCTION Real TensionLimit() const { return _p_lim - _k_min; }
  PORTABLE_INLINE_FUNCTION void ColdCurve(const Real rho, Real &e_c, Real &p_c) const {
    Real dp;
    ColdCurveAll(std::min(rho, _rho_max), e_c, p_c, dp, false);
  }

  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real TemperatureFromDensityInternalEnergy(
      const Real rho, const Real sie,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const {
    return robust::ratio(sie - ColdEnergy(std::min(rho, _rho_max)), _Cv);
  }
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real InternalEnergyFromDensityTemperature(
      const Real rho, const Real temp,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const {
    return ColdEnergy(std::min(rho, _rho_max)) + _Cv * temp;
  }
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real
  PressureFromDensityTemperature(const Real rho, const Real temperature,
                                 Indexer_t &&lambda = static_cast<Real *>(nullptr)) const;
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real PressureFromDensityInternalEnergy(
      const Real rho, const Real sie,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const;
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real MinInternalEnergyFromDensity(
      const Real rho, Indexer_t &&lambda = static_cast<Real *>(nullptr)) const;
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real
  EntropyFromDensityTemperature(const Real rho, const Real temperature,
                                Indexer_t &&lambda = static_cast<Real *>(nullptr)) const;
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real EntropyFromDensityInternalEnergy(
      const Real rho, const Real sie,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const;
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real SpecificHeatFromDensityTemperature(
      const Real rho, const Real temperature,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const {
    return _Cv;
  }
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real SpecificHeatFromDensityInternalEnergy(
      const Real rho, const Real sie,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const {
    return _Cv;
  }
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real BulkModulusFromDensityTemperature(
      const Real rho, const Real temperature,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const;
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real BulkModulusFromDensityInternalEnergy(
      const Real rho, const Real sie,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const;
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real GruneisenParamFromDensityTemperature(
      const Real rho, const Real temperature,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const {
    return Gamma(std::min(rho, _rho_max));
  }
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real GruneisenParamFromDensityInternalEnergy(
      const Real rho, const Real sie,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const {
    return Gamma(std::min(rho, _rho_max));
  }
  // Closed form: e = e_c + (P - p_c) / (rho Gamma)
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION void InternalEnergyFromDensityPressure(
      const Real rho_in, const Real P, Real &sie,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const {
    const Real rho = std::min(rho_in, _rho_max);
    Real e_c, p_c;
    ColdCurve(rho, e_c, p_c);
    sie = e_c + robust::ratio(P - p_c, rho * Gamma(rho));
  }
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION void
  FillEos(Real &rho, Real &temp, Real &energy, Real &press, Real &cv, Real &bmod,
          const unsigned long output,
          Indexer_t &&lambda = static_cast<Real *>(nullptr)) const;
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION void
  ValuesAtReferenceState(Real &rho, Real &temp, Real &sie, Real &press, Real &cv,
                         Real &bmod, Real &dpde, Real &dvdt,
                         Indexer_t &&lambda = static_cast<Real *>(nullptr)) const;
  // Generic functions provided by the base class. These contain e.g. the vector
  // overloads that use the scalar versions declared here
  SG_ADD_DEFAULT_MEAN_ATOMIC_FUNCTIONS(_AZbar)
  SG_ADD_BASE_CLASS_USINGS(Gruneisen)

  static constexpr unsigned long PreferredInput() { return _preferred_input; }
  PORTABLE_INLINE_FUNCTION void PrintParams() const {
    static constexpr char s1[]{"Gruneisen Params: "};
    printf("%s C0:%e s1:%e s2:%e s3:%e\n  G0:%e b:%e rho0:%e T0:%e\n  P0:%eCv:%e "
           "rho_max:%e\n  k_min:%e K0:%e rho_lim:%e p_lim:%e\n",
           s1, _C0, _s1, _s2, _s3, _G0, _b, _rho0, _T0, _P0, _Cv, _rho_max, _k_min, _K0,
           _rho_lim, _p_lim);
    _AZbar.PrintParams();
  }

  // Report minimum values of density and temperature
  PORTABLE_FORCEINLINE_FUNCTION
  Real MaximumDensity() const { return _rho_max; }
  // For T >= 0 the pressure is above the cold curve, which is bounded below by its rho -> 0
  // limit.
  PORTABLE_FORCEINLINE_FUNCTION
  Real MinimumPressure() const { return TensionLimit(); }
  PORTABLE_INLINE_FUNCTION
  Real MaximumPressureAtTemperature(const Real T) const {
    return PressureFromDensityTemperature(MaxStableDensityAtTemperature(T), T);
  }

  template <typename Indexer_t>
  PORTABLE_INLINE_FUNCTION void
  DensityEnergyFromPressureTemperature(const Real press, const Real temp,
                                       Indexer_t &&lambda, Real &rho, Real &sie) const;
  // Exact (P,T) partials (overrides the EosBase central-difference default)
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION void DensityEnergyDerivativesFromPressureTemperature(
      const Real press, const Real temp, Indexer_t &&lambda, Real &rho, Real &sie,
      Real &drho_dP, Real &drho_dT, Real &de_dP, Real &de_dT) const;
  inline void Finalize() {}
  static std::string EosType() { return std::string("Gruneisen"); }
  static std::string EosPyType() { return EosType(); }

 private:
  Real _C0, _s1, _s2, _s3, _G0, _b, _rho0, _T0, _P0, _Cv, _rho_max;
  MeanAtomicProperties _AZbar;
  // Cold curve. Expansion: Vinet shape to rho_lim, then the bounded tail.
  Real _e_c0 = 0, _p_c0 = 0, _K0 = 0, _eta_v = 0, _k_min = 0;
  Real _rho_lim = 0, _p_lim = 0, _e_lim = 0;
  // Compression: cumulative isentrope integral at nodes in eta
  static constexpr int NNODE_MAX = 96;
  int _n_nodes = 0;
  Real _eta_nodes[NNODE_MAX] = {}, _int_nodes[NNODE_MAX] = {};
  // static constexpr const char _eos_type[] = {"Gruneisen"};
  PORTABLE_INLINE_FUNCTION
  Real Gamma(const Real rho_in) const {
    const Real rho = std::min(rho_in, _rho_max);
    return rho < _rho0 ? _G0 : _G0 * _rho0 / rho + _b * (1 - _rho0 / rho);
  }
  PORTABLE_INLINE_FUNCTION
  Real dPres_drho_e(const Real rho, const Real sie) const;
  // Upstream (rho >= rho0) branch of (dP/drho)_e
  PORTABLE_INLINE_FUNCTION
  Real dPres_drho_e_Compression(const Real rho, const Real sie) const;
  // Hugoniot (rho >= rho0)
  PORTABLE_INLINE_FUNCTION void Hugoniot(const Real eta, Real &P_H, Real &E_H) const {
    using namespace math_utils;
    const Real s = _s1 * eta + _s2 * pow<2>(eta) + _s3 * pow<3>(eta);
    P_H = _P0 + pow<2>(_C0) * _rho0 * eta / pow<2>(1 - s);
    E_H = (P_H + _P0) * eta / _rho0 / 2.;
  }
  // ln theta: rho Gamma / rho^2 = d ln theta / drho, theta(rho0) = 1
  PORTABLE_INLINE_FUNCTION Real LogTheta(const Real rho) const {
    if (rho < _rho0) return _G0 * std::log(rho / _rho0);
    const Real eta = 1 - _rho0 / rho;
    return -_b * std::log1p(-eta) + (_G0 - _b) * eta;
  }
  // Isentrope integrand in eta (compression)
  PORTABLE_INLINE_FUNCTION Real IsentropeIntegrand(const Real eta) const {
    Real P_H, E_H;
    Hugoniot(eta, P_H, E_H);
    const Real rhoG = _G0 * _rho0 + _b * (_rho0 / (1 - eta) - _rho0);
    const Real ltheta = -_b * std::log1p(-eta) + (_G0 - _b) * eta;
    return (P_H - rhoG * E_H) * std::exp(-ltheta) / _rho0;
  }
  PORTABLE_INLINE_FUNCTION Real GaussLegendre8(const Real a, const Real b) const;
  PORTABLE_INLINE_FUNCTION Real ColdEnergyCompression(const Real eta) const;
  PORTABLE_INLINE_FUNCTION Real ColdEnergy(const Real rho) const {
    Real e_c, p_c, dp;
    ColdCurveAll(rho, e_c, p_c, dp, false);
    return e_c;
  }
  // e_c, p_c, and (if want_dp and rho < rho0) dp_c/drho at rho <= rho_max
  PORTABLE_INLINE_FUNCTION void ColdCurveAll(const Real rho, Real &e_c, Real &p_c,
                                             Real &dp_c, const bool want_dp) const;
  PORTABLE_INLINE_FUNCTION void VinetExpansion(const Real rho, Real &e_c, Real &p_c,
                                               Real &dp_c) const;
  PORTABLE_INLINE_FUNCTION void InitializeColdCurve(const Real k_min);
  // (dP/drho)_T
  PORTABLE_INLINE_FUNCTION Real dPres_drho_T(const Real rho, const Real temp) const;
  static constexpr const unsigned long _preferred_input =
      thermalqs::density | thermalqs::specific_internal_energy;
  // Scaling factor for density singularity (reference pressure blows up). Consistent with
  // Pagosa implementation
  static constexpr Real RHOMAX_SAFETY = 0.99;
  static constexpr Real K_MIN_DEFAULT_FRACTION = 0.02;
};

namespace gruneisen_utils {
PORTABLE_INLINE_FUNCTION Real find_min_bounded_val(const Real val1, const Real val2,
                                                   const Real min, const Real max) {
  // Try to find the minimum bounded value. If none exists, returns negative infinity
  const Real minval = std::min(val1, val2);
  const Real maxval = std::max(val1, val2);
  if (minval > min && minval < max) {
    // Minimum of two values is bounded so return it
    return minval;
  } else if (maxval > min && maxval < max) {
    // Minimum isn't bounded but maximum is so return that
    return maxval;
  } else {
    // Neither value is bounded so just return negative infinity to clearly show that the
    // value is out of bounds
    return -std::numeric_limits<Real>::infinity();
  }
}
PORTABLE_INLINE_FUNCTION bool is_near_zero(const Real val, const Real tol) {
  return std::abs(val) < tol;
}
} // namespace gruneisen_utils
/*
The Gruneisen EOS diverges at a specific compression. Ensure that the maximum density is
below the smallest singularity in the reference pressure curve
*/
PORTABLE_INLINE_FUNCTION Real Gruneisen::ComputeRhoMax(const Real s1, const Real s2,
                                                       const Real s3, const Real rho0) {
  using namespace gruneisen_utils;
  using namespace math_utils;

  // Polynomial from the denominator of the reference pressure curve:
  auto poly = [=](Real eta) {
    return 1 - s1 * eta - s2 * pow<2>(eta) - s3 * pow<3>(eta);
  };

  // Comparisons to zero are actually made to
  static constexpr Real EPS = robust::EPS();

  // First find the eta root. A negative root indicates that there is no maximum density.
  Real root = -1.; // Non-sensical root means none exists
  Real discriminant, root1, root2;
  if (is_near_zero(s2, EPS) && is_near_zero(s3, EPS) && s1 > 1.) {
    // Linear Us-up analytic root
    root = 1. / s1;
  } else if (is_near_zero(s3, EPS) && !is_near_zero(s2, EPS)) {
    // Quadratic Us-up
    discriminant = pow<2>(s1) + 4. * s2;
    if (discriminant < 0.) {
      root = -1.; // Imaginary roots, so no limit
    } else {
      root1 = (s1 + std::sqrt(discriminant)) / (-2. * s2);
      root2 = (s1 - std::sqrt(discriminant)) / (-2. * s2);
      root = find_min_bounded_val(root1, root2, 0., 1.);
    }
  } else if (!is_near_zero(s3, EPS)) {
    // Cubic Us-up: we'll use an iterative method to search for the minimum bounded root
    // Note: when the discriminant of a cubic is less than zero, only one real root exists
    //       so we don't need anything fancy to find the root.
    discriminant = -18. * s3 * s2 * s1 - 4. * pow<3>(-s2) + pow<2>(s2) * pow<2>(s1) -
                   4. * s3 * pow<3>(s1) - 27. * pow<2>(s3);
    Real minbound = 0.;
    Real maxbound = 1.;
    if (discriminant >= 0) {
      // Three real roots (roots may have multiplicity). We need to use the extrema to
      // ensure we have a proper bracket in which to search for the root (if they exist).
      // Note: If the discriminant is positive, then `pow<2>(s2) - 3 * s3 * s1` must
      //       necessarily be positive (the reverse does not hold). Thus the extrema below
      //       are not imaginary.
      const Real extremum1 = (s2 + std::sqrt(pow<2>(s2) - 3. * s3 * s1)) / (-3. * s3);
      const Real extremum2 = (s2 - std::sqrt(pow<2>(s2) - 3. * s3 * s1)) / (-3. * s3);
      const Real min_extremum = std::min(extremum1, extremum2);
      const Real max_extremum = std::max(extremum1, extremum2);
      // Because poly(eta = 0) = 1, the only possible root will lie in an area of the
      // cubic that has negative slope. That area is determined by the extrema.
      if (s3 < 0.) {
        // Cubic is *increasing*
        // The only place with negative slope is between the extrema so that's where the
        // relevant root will be.
        minbound = std::max(min_extremum, minbound);
        maxbound = std::min(std::max(max_extremum, minbound), maxbound);
      } else {
        // Cubic is *decreasing*
        // The only place with negative slope is outside the extrema. Further, the only
        // possibility for multiple bound roots will occur when the extremum are both
        // positive, this is the only case we need to focus on.
        if (min_extremum > 0) {
          maxbound = std::min(min_extremum, maxbound);
        }
      } // s3 > 0 ; else
    }   // discriminant >= 0
    if (poly(minbound) * poly(maxbound) < 0.) {
      // Root is appropriately bounded
      using RootFinding1D::regula_falsi;
      using RootFinding1D::RootCounts;
      using RootFinding1D::Status;
      static constexpr Real factor = 100;
      static constexpr Real xtol = factor * EPS;
      static constexpr Real ytol = factor / 10 * EPS;
      static constexpr Real eta_guess = 0.001;
      auto status =
          regula_falsi(poly, 0., eta_guess, minbound, maxbound, xtol, ytol, root);
      if (status != Status::SUCCESS) {
        // Root finder failed even though the solution was bracketed... this is an error
        EOS_ERROR("Gruneisen initialization: Cubic root find failed. Maximum "
                  "density cannot be"
                  " automatically determined and must be manually specified\n.");
      }
    } else if (is_near_zero(poly(minbound), EPS)) {
      root = minbound;
    } else if (is_near_zero(poly(maxbound), EPS)) {
      root = maxbound;
    } else {
      // Root doesn't lie within physical bounds for eta so no maximum density exists
      root = -1.;
    }
  } // Linear/Quadratic/Cubic
  // `root` is a value of eta so it should be greater than zero and less than 1
  if (root > 0. && root < 1.) {
    return rho0 / ((1 - root) + EPS); // convert from eta to compression
  } else {
    // No bounded root exists so there is no upper-limit on the compression
    return std::numeric_limits<Real>::infinity();
  }
}


PORTABLE_INLINE_FUNCTION Real Gruneisen::dPres_drho_e_Compression(const Real rho,
                                                                  const Real sie) const {
  using namespace math_utils;
  const Real eta = 1 - _rho0 / rho;
  const Real s = _s1 * eta + _s2 * pow<2>(eta) + _s3 * pow<3>(eta);
  const Real ds = _s1 + 2 * _s2 * eta + 3 * _s3 * pow<2>(eta);
  const Real deta = _rho0 / pow<2>(rho);
  const Real dGam = (_b - _G0) * deta;
  const Real P_H = _P0 + pow<2>(_C0) * _rho0 * eta / pow<2>(1 - s);
  const Real dP_H = math_utils::pow<2>(_C0) * _rho0 / pow<2>(1 - s) * deta *
                    (1 + 2 * eta * ds / (1 - s));
  const Real E_H = (P_H + _P0) * eta / _rho0 / 2.;
  const Real dE_H = deta * (P_H + _P0) / _rho0 / 2. + eta / _rho0 / 2 * dP_H;
  return dP_H + Gamma(rho) * (sie - E_H) + rho * dGam * (sie - E_H) -
         rho * Gamma(rho) * dE_H;
}

PORTABLE_INLINE_FUNCTION Real Gruneisen::dPres_drho_e(const Real rho_in,
                                                      const Real sie) const {
  const Real rho = std::min(rho_in, _rho_max);
  if (rho >= _rho0) return dPres_drho_e_Compression(rho, sie);
  // P = p_c + G0 rho (e - e_c), de_c/drho = p_c / rho^2
  Real e_c, p_c, dp_c;
  ColdCurveAll(rho, e_c, p_c, dp_c, true);
  return dp_c + _G0 * (sie - e_c) - _G0 * p_c / rho;
}

PORTABLE_INLINE_FUNCTION Real Gruneisen::GaussLegendre8(const Real a, const Real b) const {
  constexpr Real x[4] = {0.1834346424956498049394761, 0.5255324099163289858177390,
                         0.7966664774136267395915539, 0.9602898564975362316835609};
  constexpr Real w[4] = {0.3626837833783619829651504, 0.3137066458778872873379622,
                         0.2223810344533744705443560, 0.1012285362903762591525314};
  const Real c = 0.5 * (a + b), h = 0.5 * (b - a);
  Real sum = 0;
  for (int i = 0; i < 4; i++) {
    sum += w[i] * (IsentropeIntegrand(c - h * x[i]) + IsentropeIntegrand(c + h * x[i]));
  }
  return h * sum;
}

PORTABLE_INLINE_FUNCTION Real Gruneisen::ColdEnergyCompression(const Real eta) const {
  // largest node <= eta
  int lo = 0, hi = _n_nodes - 1;
  if (hi < 0) return _e_c0 * std::exp(-_b * std::log1p(-eta) + (_G0 - _b) * eta);
  if (eta >= _eta_nodes[hi]) {
    lo = hi;
  } else {
    while (hi - lo > 1) {
      const int mid = (lo + hi) / 2;
      if (_eta_nodes[mid] <= eta) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
  }
  const Real integral =
      _int_nodes[lo] + (eta > _eta_nodes[lo] ? GaussLegendre8(_eta_nodes[lo], eta) : 0.0);
  const Real ltheta = -_b * std::log1p(-eta) + (_G0 - _b) * eta;
  return std::exp(ltheta) * (_e_c0 + integral);
}

PORTABLE_INLINE_FUNCTION void Gruneisen::VinetExpansion(const Real rho, Real &e_c,
                                                        Real &p_c, Real &dp_c) const {
  const Real x = std::cbrt(_rho0 / rho);
  const Real y = _eta_v * (1 - x);
  const Real ey = std::exp(y);
  p_c = _p_c0 + 3 * _K0 * (1 - x) / (x * x) * ey;
  e_c = _e_c0 - _p_c0 * (1 / rho - 1 / _rho0) +
        9 * _K0 / (_rho0 * _eta_v * _eta_v) * (1 - (1 - y) * ey);
  dp_c = _K0 / rho * ey / (x * x) * (2 - x + _eta_v * x * (1 - x));
}

PORTABLE_INLINE_FUNCTION void Gruneisen::ColdCurveAll(const Real rho, Real &e_c,
                                                      Real &p_c, Real &dp_c,
                                                      const bool want_dp) const {
  if (rho >= _rho0) {
    const Real eta = 1 - _rho0 / rho;
    e_c = ColdEnergyCompression(eta);
    Real P_H, E_H;
    Hugoniot(eta, P_H, E_H);
    const Real rhoG = rho * Gamma(rho);
    p_c = P_H + rhoG * (e_c - E_H);
    dp_c = want_dp ? dPres_drho_e_Compression(rho, e_c) + rhoG * p_c / (rho * rho) : 0.0;
  } else if (rho >= _rho_lim) {
    VinetExpansion(rho, e_c, p_c, dp_c);
  } else {
    // Bounded tail: rho dp_c/drho = k_min rho / rho_lim
    const Real r = std::max(rho, robust::SMALL());
    const Real p_inf = _p_lim - _k_min;
    const Real slope = _k_min / _rho_lim;
    p_c = p_inf + slope * r;
    e_c = _e_lim + p_inf * (1 / _rho_lim - 1 / r) + slope * std::log(r / _rho_lim);
    dp_c = slope;
  }
}

PORTABLE_INLINE_FUNCTION void Gruneisen::InitializeColdCurve(const Real k_min) {
  _e_c0 = -_Cv * _T0;
  // P_H(rho0) = P0, E_H(rho0) = 0, rho0 Gamma(rho0) = G0 rho0
  _p_c0 = _P0 + _G0 * _rho0 * _e_c0;
  // 0 K bulk modulus at rho0 from the compression side: rho (dP/drho)_e + Gamma p_c
  _K0 = _rho0 * dPres_drho_e_Compression(_rho0, _e_c0) + _G0 * _p_c0;
  const Real Kp = std::max(4 * _s1 - 1, 2.0);
  _eta_v = 1.5 * (Kp - 1);
  _k_min = (k_min < 0) ? K_MIN_DEFAULT_FRACTION * _K0 : k_min;
  _rho_lim = 0;
  _p_lim = -std::numeric_limits<Real>::max();
  _e_lim = 0;
  _n_nodes = 0;
  if (_K0 > 0 && _k_min > 0 && _k_min < _K0) {
    // Vinet K(x) = K0 exp(eta_v (1 - x)) x^-2 (2 - x + eta_v x (1 - x)) falls monotonically
    // from K0 at x = 1 to 0 at the spinodal x_s; bisect for K = k_min on [1, x_s].
    auto bulk = [this](const Real x) {
      return _K0 * std::exp(_eta_v * (1 - x)) / (x * x) *
             (2 - x + _eta_v * x * (1 - x));
    };
    const Real ev = _eta_v;
    Real lo = 1, hi = ((ev - 1) + std::sqrt((ev - 1) * (ev - 1) + 8 * ev)) / (2 * ev);
    for (int i = 0; i < 200 && hi - lo > 4 * std::numeric_limits<Real>::epsilon(); i++) {
      const Real mid = 0.5 * (lo + hi);
      if (bulk(mid) > _k_min) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    const Real x = 0.5 * (lo + hi);
    _rho_lim = _rho0 / (x * x * x);
    Real dp;
    VinetExpansion(_rho_lim, _e_lim, _p_lim, dp);
  }
  // Isentrope integral nodes in eta, clustered toward the Hugoniot pole eta_s (or eta = 1)
  if (_rho_max > _rho0 && _Cv >= 0) {
    const Real rho_pole = ComputeRhoMax(_s1, _s2, _s3, _rho0);
    Real eta_s = 1;
    if (rho_pole > _rho0 && rho_pole < std::numeric_limits<Real>::infinity()) {
      eta_s = std::min(1.0, 1 - _rho0 / rho_pole);
    }
    const Real eta_top = std::min(1 - _rho0 / _rho_max, eta_s * (1 - 1e-6));
    const Real h0 = eta_s / 16;
    Real eta = 0, integral = 0;
    _eta_nodes[0] = 0;
    _int_nodes[0] = 0;
    _n_nodes = 1;
    while (eta < eta_top && _n_nodes < NNODE_MAX) {
      const Real step = (eta < 0.5 * eta_s) ? h0 : 0.2 * (eta_s - eta);
      const Real next = std::min(eta + step, eta_top);
      integral += GaussLegendre8(eta, next);
      _eta_nodes[_n_nodes] = next;
      _int_nodes[_n_nodes] = integral;
      _n_nodes++;
      eta = next;
    }
  }
}

template <typename Indexer_t>
PORTABLE_INLINE_FUNCTION Real Gruneisen::PressureFromDensityInternalEnergy(
    const Real rho_in, const Real sie, Indexer_t &&lambda) const {
  using namespace math_utils;
  const Real rho = std::min(rho_in, _rho_max);
  if (rho >= _rho0) {
    // Upstream P(rho, e), unchanged: P_H + rho Gamma (e - E_H) = p_c + rho Gamma (e - e_c)
    Real P_H, E_H;
    Hugoniot(1 - _rho0 / rho, P_H, E_H);
    return P_H + Gamma(rho) * rho * (sie - E_H);
  }
  Real e_c, p_c;
  ColdCurve(rho, e_c, p_c);
  return p_c + _G0 * rho * (sie - e_c);
}
template <typename Indexer_t>
PORTABLE_INLINE_FUNCTION Real
Gruneisen::MinInternalEnergyFromDensity(const Real rho_in, Indexer_t &&lambda) const {
  return ColdEnergy(std::min(rho_in, _rho_max));
}
template <typename Indexer_t>
PORTABLE_INLINE_FUNCTION Real Gruneisen::EntropyFromDensityTemperature(
    const Real rho_in, const Real temp, Indexer_t &&lambda) const {
  // S = Cv ln(T / T0) - Cv ln theta(rho)
  const Real rho = std::min(rho_in, _rho_max);
  if (temp <= 0) return -std::numeric_limits<Real>::infinity();
  const Real Tref = _T0 > 0 ? _T0 : 1.0;
  return _Cv * (std::log(temp / Tref) - LogTheta(rho));
}
template <typename Indexer_t>
PORTABLE_INLINE_FUNCTION Real Gruneisen::EntropyFromDensityInternalEnergy(
    const Real rho_in, const Real sie, Indexer_t &&lambda) const {
  const Real rho = std::min(rho_in, _rho_max);
  return EntropyFromDensityTemperature(rho, TemperatureFromDensityInternalEnergy(rho, sie));
}
template <typename Indexer_t>
PORTABLE_INLINE_FUNCTION Real Gruneisen::BulkModulusFromDensityInternalEnergy(
    const Real rho_in, const Real sie, Indexer_t &&lambda) const {
  const Real rho = std::min(rho_in, _rho_max);
  const Real dPdr_e = dPres_drho_e(rho, sie);
  const Real dPde_r = rho * Gamma(rho);
  // Thermodynamic identity
  return rho * dPdr_e + PressureFromDensityInternalEnergy(rho, sie) / rho * dPde_r;
}
template <typename Indexer_t>
PORTABLE_INLINE_FUNCTION Real Gruneisen::PressureFromDensityTemperature(
    const Real rho_in, const Real temp, Indexer_t &&lambda) const {
  const Real rho = std::min(rho_in, _rho_max);
  return PressureFromDensityInternalEnergy(
      rho, InternalEnergyFromDensityTemperature(rho, temp));
}
template <typename Indexer_t>
PORTABLE_INLINE_FUNCTION Real Gruneisen::BulkModulusFromDensityTemperature(
    const Real rho_in, const Real temp, Indexer_t &&lambda) const {
  const Real rho = std::min(rho_in, _rho_max);
  return BulkModulusFromDensityInternalEnergy(
      rho, InternalEnergyFromDensityTemperature(rho, temp));
}
PORTABLE_INLINE_FUNCTION Real Gruneisen::dPres_drho_T(const Real rho_in,
                                                      const Real temp) const {
  // (dP/drho)_T = (dP/drho)_e + (dP/de)_rho (de/drho)_T, (de/drho)_T = p_c / rho^2
  const Real rho = std::min(rho_in, _rho_max);
  Real e_c, p_c;
  ColdCurve(rho, e_c, p_c);
  return dPres_drho_e(rho, e_c + _Cv * temp) + Gamma(rho) * p_c / rho;
}
PORTABLE_INLINE_FUNCTION Real
Gruneisen::MaxStableDensityAtTemperature(const Real temperature) const {
  // The 0 K curve has rho dp_c/drho > 0 everywhere and the thermal pressure
  // rho Gamma Cv T grows with rho when b >= 0, so the isotherms are monotone up to
  // rho_max. A negative b can still make (dP/drho)_T change sign at high T; then the
  // density of the pressure maximum on the isotherm is returned.
  if (!(_rho_max < std::numeric_limits<Real>::infinity())) return _rho_max;
  if (dPres_drho_T(_rho_max, temperature) >= 0) return _rho_max;
  if (dPres_drho_T(_rho0, temperature) < 0) {
    using PortsOfCall::printf;
    printf("ERROR: The pressure is decreasing as density increases at the reference\n"
           "       density, %.15g, for temperature, %.15g. Check that the reference\n"
           "       temperature is set correctly. This is an unstable state.",
           _rho0, temperature);
    PORTABLE_ALWAYS_THROW_OR_ABORT("Input pressure is off EOS surface");
  }
  // JMM: called inside a device kernel so does not need device annotation
  auto dPdrho_T = [this, temperature](const Real r) {
    return dPres_drho_T(r, temperature);
  };
  const Real rho_lower = _rho0;
  const Real rho_upper = std::min(_rho_max, 1.0e4);
  const Real rho_guess = (rho_lower + rho_upper) / 2.;
  Real rho_at_max_P;
  using RootFinding1D::regula_falsi;
  using RootFinding1D::Status;
  auto status = regula_falsi(dPdrho_T, 0., rho_guess, rho_lower, rho_upper, 1.0e-8,
                             1.0e-8, rho_at_max_P);
  if (status != Status::SUCCESS) {
    // Root finder failed even though the solution should be bracketed
    EOS_ERROR("Gruneisen::MaxStableDensityAtTemperature: "
              "Root find failed to find maximum P at T despite aparent bracket\n");
  }
  return rho_at_max_P;
}
template <typename Indexer_t>
PORTABLE_INLINE_FUNCTION void Gruneisen::DensityEnergyFromPressureTemperature(
    const Real press, const Real temp, Indexer_t &&lambda, Real &rho, Real &sie) const {
  // P(rho, T) increases monotonically with rho on (0, rho_hi]: bracketed Newton.
  Real rho_hi = MaxStableDensityAtTemperature(temp);
  if (!(rho_hi < std::numeric_limits<Real>::infinity())) {
    rho_hi = _n_nodes > 0 ? _rho0 / (1 - _eta_nodes[_n_nodes - 1]) : 1e4 * _rho0;
  }
  const Real rho_lo = 1e-12 * _rho0;
  const Real P_hi = PressureFromDensityTemperature(rho_hi, temp);
  if (press >= P_hi) {
    if (press > P_hi) {
      using PortsOfCall::printf;
      printf("ERROR: Requested pressure, %.15g, exceeds maximum, %.15g, for \n"
             "       temperature, %.15g.\n"
             "       setting pressure to maximum allowed value.\n",
             press, P_hi, temp);
    }
    rho = rho_hi;
  } else if (press <= PressureFromDensityTemperature(rho_lo, temp)) {
    rho = rho_lo;
  } else {
    Real a = rho_lo, b = rho_hi;
    Real x = (_rho0 > a && _rho0 < b) ? _rho0 : std::sqrt(a * b);
    for (int i = 0; i < 200; i++) {
      const Real f = PressureFromDensityTemperature(x, temp) - press;
      if (f == 0) break;
      if (f > 0) {
        b = x;
      } else {
        a = x;
      }
      const Real d = dPres_drho_T(x, temp);
      Real xn = x - f / d;
      if (!(xn > a && xn < b)) xn = (b > 4 * a) ? std::sqrt(a * b) : 0.5 * (a + b);
      const bool done = std::abs(xn - x) <= 4 * std::numeric_limits<Real>::epsilon() * x;
      x = xn;
      if (done || b - a <= 4 * std::numeric_limits<Real>::epsilon() * b) break;
    }
    rho = x;
  }
  sie = InternalEnergyFromDensityTemperature(rho, temp);
}
template <typename Indexer_t>
PORTABLE_INLINE_FUNCTION void Gruneisen::DensityEnergyDerivativesFromPressureTemperature(
    const Real press, const Real temp, Indexer_t &&lambda, Real &rho, Real &sie,
    Real &drho_dP, Real &drho_dT, Real &de_dP, Real &de_dT) const {
  DensityEnergyFromPressureTemperature(press, temp, lambda, rho, sie);
  const Real r = std::min(rho, _rho_max);
  Real e_c, p_c;
  ColdCurve(r, e_c, p_c);
  const Real dPdr_T = dPres_drho_T(r, temp);
  const Real dPdT_r = r * Gamma(r) * _Cv;
  const Real dedr_T = p_c / (r * r);
  drho_dP = 1 / dPdr_T;
  drho_dT = -dPdT_r / dPdr_T;
  de_dP = dedr_T * drho_dP;
  de_dT = _Cv + dedr_T * drho_dT;
}
template <typename Indexer_t>
PORTABLE_INLINE_FUNCTION void
Gruneisen::FillEos(Real &rho_in, Real &temp, Real &sie, Real &press, Real &cv, Real &bmod,
                   const unsigned long output, Indexer_t &&lambda) const {
  // The following could be sped up with work!
  const unsigned long input = ~output;
  if (thermalqs::temperature & input && thermalqs::pressure & input) {
    DensityEnergyFromPressureTemperature(press, temp, lambda, rho_in, sie);
  } else if (thermalqs::density & output ||
             thermalqs::specific_internal_energy & output) {
    // Error out on density or energy output because they're currently required as inputs
    EOS_ERROR("Gruneisen FillEos: Density and energy are currently required inputs "
              "except when pressure and temperature are inputs.\n");
  }
  const Real rho = std::min(rho_in, _rho_max);
  if (output & thermalqs::pressure) press = PressureFromDensityInternalEnergy(rho, sie);
  if (output & thermalqs::temperature)
    temp = TemperatureFromDensityInternalEnergy(rho, sie);
  if (output & thermalqs::bulk_modulus)
    bmod = BulkModulusFromDensityInternalEnergy(rho, sie);
  if (output & thermalqs::specific_heat)
    cv = SpecificHeatFromDensityInternalEnergy(rho, sie);
}

// TODO(JMM): pre-cache these rather than recomputing them each time
template <typename Indexer_t>
PORTABLE_INLINE_FUNCTION void
Gruneisen::ValuesAtReferenceState(Real &rho, Real &temp, Real &sie, Real &press, Real &cv,
                                  Real &bmod, Real &dpde, Real &dvdt,
                                  Indexer_t &&lambda) const {
  rho = _rho0;
  temp = _T0;
  sie = 0;
  press = PressureFromDensityInternalEnergy(rho, sie, lambda);
  cv = SpecificHeatFromDensityInternalEnergy(rho, sie, lambda);
  bmod = BulkModulusFromDensityInternalEnergy(rho, sie, lambda);
  dpde = rho * GruneisenParamFromDensityInternalEnergy(rho, sie, lambda);
  // (dv/dT)_P = Gamma Cv / B_T
  dvdt = robust::ratio(GruneisenParamFromDensityInternalEnergy(rho, sie, lambda) * cv,
                       rho * dPres_drho_T(rho, temp));
}
} // namespace singularity

#endif // _SINGULARITY_EOS_EOS_GRUNEISEN_HPP_
