//------------------------------------------------------------------------------
// A stiffened gas carrying its own (P,T) box, for driving PTESolveDual directly in tests.
//
// It implements exactly the accessors PTESolveDual needs, with closed-form (P,T)
// partials, and reports the box the way a (P,T) table reports its grid (MinimumPressure,
// MaximumPressureAtTemperature, MinimumTemperature).  Being self-contained, the solver's
// trajectory on it does not move when a library EOS's partials change, so recorded
// results stay comparable across versions of the solver alone.
//
//   rho(P,T) = (P + Pinf) / (gm1 Cv T)
//   e(P,T)   = Cv T (P + (gm1+1) Pinf) / (P + Pinf) + q
// Pinf = 0 is the ideal gas.
//------------------------------------------------------------------------------
#ifndef SINGULARITY_TEST_PTE_DUAL_TEST_EOS_HPP_
#define SINGULARITY_TEST_PTE_DUAL_TEST_EOS_HPP_

#include <cstddef>

#include <ports-of-call/portability.hpp>

namespace pte_dual_test {

struct BoxedStiffGas {
  Real gm1, cv, pinf, q;
  Real p_min, p_max, t_min;

  template <typename L>
  PORTABLE_INLINE_FUNCTION void
  DensityEnergyFromPressureTemperature(const Real P, const Real T, L &&, Real &rho,
                                       Real &sie) const {
    rho = (P + pinf) / (gm1 * cv * T);
    sie = cv * T * (P + (gm1 + 1.0) * pinf) / (P + pinf) + q;
  }
  template <typename L>
  PORTABLE_INLINE_FUNCTION void DensityEnergyDerivativesFromPressureTemperature(
      const Real P, const Real T, L &&lambda, Real &rho, Real &sie, Real &drho_dP,
      Real &drho_dT, Real &de_dP, Real &de_dT) const {
    DensityEnergyFromPressureTemperature(P, T, lambda, rho, sie);
    const Real pp = P + pinf;
    drho_dP = 1.0 / (gm1 * cv * T);
    drho_dT = -rho / T;
    de_dP = -gm1 * cv * T * pinf / (pp * pp);
    de_dT = cv * (P + (gm1 + 1.0) * pinf) / pp;
  }
  template <typename L = Real *>
  PORTABLE_INLINE_FUNCTION Real PressureFromDensityTemperature(const Real rho,
                                                               const Real T,
                                                               L && = nullptr) const {
    return gm1 * cv * rho * T - pinf;
  }
  PORTABLE_INLINE_FUNCTION Real MinimumPressure() const { return p_min; }
  PORTABLE_INLINE_FUNCTION Real MaximumPressureAtTemperature(const Real) const {
    return p_max;
  }
  PORTABLE_INLINE_FUNCTION Real MinimumTemperature() const { return t_min; }
};

template <typename T>
struct Idx {
  T *d;
  PORTABLE_INLINE_FUNCTION T &operator[](const std::size_t i) const { return d[i]; }
};
struct NoLambda {
  PORTABLE_INLINE_FUNCTION Real *operator[](const std::size_t) const { return nullptr; }
};

// Clayton et al. (arXiv:2606.27726) Table 6.1 ideal and stiffened gases, cgs, in a box
// given in barye and K.
inline void ClaytonGases(BoxedStiffGas *eos, const Real p_min, const Real p_max,
                         const Real t_min) {
  eos[0] = {0.0014 / 0.001 - 1.0, 0.001e10, 0.0, 0.0, p_min, p_max, t_min};
  eos[1] = {0.0085 / 0.0012 - 1.0, 0.0012e10, 1.0e8, 0.0, p_min, p_max, t_min};
}

} // namespace pte_dual_test

#endif // SINGULARITY_TEST_PTE_DUAL_TEST_EOS_HPP_
