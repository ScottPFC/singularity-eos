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

#ifndef _SINGULARITY_EOS_EOS_VINET_HPP_
#define _SINGULARITY_EOS_EOS_VINET_HPP_

#include <cmath>
#include <cstdio>

#include <limits>

#include <ports-of-call/portable_errors.hpp>
#include <singularity-eos/base/constants.hpp>
#include <singularity-eos/base/math_utils.hpp>
#include <singularity-eos/base/robust_utils.hpp>
#include <singularity-eos/eos/eos_base.hpp>

namespace singularity {

using namespace eos_base;
using namespace robust;

// COMMENT: This is a complete, thermodynamically consistent Vinet
// EOS created at Sandia National Laboratories by AEM.
// J. Appl. Phys. 119, 015904 (2016)
// It is particularly suited for high-pressure applications.
//
// OPTIONAL EXTENSION (solid-pool project). The constructor taking k_min and t_floor
// enables two extensions; zero (the default of the original constructor) disables each,
// recovering the original model exactly.
//  * Tension: for rho < rho_lim, where K_T = rho dp/drho on the T0 isotherm falls to
//    k_min, the isotherm is continued with K_T = k_min:
//      p_ref = p_lim + k_min ln(rho/rho_lim),
//      e_ref = e_lim + (A0 B0 T0 - p_lim)(v - v_lim) + k_min (-v ln(rho/rho_lim) - v + v_lim)
//    (p is C1 and e is C2 at the joint; the thermal part is unchanged). Since
//    rho*Gamma = A0 B0/Cv0 is constant, rho_lim does not depend on T.
//  * Cold floor: below T = t_floor the entropy is continued by its quadratic Taylor
//    expansion in e about e_f(rho) = e(rho, t_floor), so that T > 0 and s stays concave:
//      T = t_floor/(1 - u/(Cv0 t_floor)),  u = e - e_f < 0,   p = p_ref + A0 B0 (T - T0),
//      Cv = Cv0 (t_floor/T)^2.
// Reference: solidpool_oracle.eos.Vinet.
class Vinet : public EosBase<Vinet> {
 public:
  Vinet() = default;
  // TODO(JMM): Should Vinet and MGPower take AZbar before
  // the series coefficients? Would require an overload...
  Vinet(const Real rho0, const Real T0, const Real B0, const Real BP0, const Real A0,
        const Real Cv0, const Real E0, const Real S0, const Real *expconsts,
        const MeanAtomicProperties &AZbar = MeanAtomicProperties())
      : _rho0(rho0), _T0(T0), _B0(B0), _BP0(BP0), _A0(A0), _Cv0(Cv0), _E0(E0), _S0(S0),
        _AZbar(AZbar) {
    CheckParams();
    InitializeVinet(expconsts);
  }
  Vinet(const Real rho0, const Real T0, const Real B0, const Real BP0, const Real A0,
        const Real Cv0, const Real E0, const Real S0, const Real *expconsts,
        const Real k_min, const Real t_floor,
        const MeanAtomicProperties &AZbar = MeanAtomicProperties())
      : _rho0(rho0), _T0(T0), _B0(B0), _BP0(BP0), _A0(A0), _Cv0(Cv0), _E0(E0), _S0(S0),
        _AZbar(AZbar), _k_min(k_min), _t_floor(t_floor) {
    CheckParams();
    InitializeVinet(expconsts);
    InitializeExtension();
  }
  PORTABLE_INLINE_FUNCTION void CheckParams() const;

  // Extension parameters (0 means the corresponding extension is off)
  PORTABLE_FORCEINLINE_FUNCTION Real KMin() const { return _k_min; }
  PORTABLE_FORCEINLINE_FUNCTION Real TFloor() const { return _t_floor; }
  // Spinodal density; 0 if the tension extension is off
  PORTABLE_FORCEINLINE_FUNCTION Real RhoLim() const { return _rho_lim; }

  // Closed form: T = T0 + (P - p_ref(rho))/(A0 B0)
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION void InternalEnergyFromDensityPressure(
      const Real rho, const Real P, Real &sie,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const {
    Real ref[3];
    Vinet_Extended_Ref(rho, ref);
    const Real ab = _A0 * _B0;
    const Real temp = _T0 + robust::ratio(P - ref[1], ab);
    if (_t_floor > 0.0 && temp <= 0.0) {
      // below the T=0 pressure floor p_ref - A0 B0 T0; unreachable
      sie = std::numeric_limits<Real>::quiet_NaN();
      return;
    }
    sie = InternalEnergyFromDensityTemperature(rho, temp, lambda);
  }

  Vinet GetOnDevice() { return *this; }
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real TemperatureFromDensityInternalEnergy(
      const Real rho, const Real sie,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const;
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real InternalEnergyFromDensityTemperature(
      const Real rho, const Real temp,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const;
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real
  PressureFromDensityTemperature(const Real rho, const Real temp,
                                 Indexer_t &&lambda = static_cast<Real *>(nullptr)) const;
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real PressureFromDensityInternalEnergy(
      const Real rho, const Real sie,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const;
  // Entropy added AEM Dec. 2022
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real
  EntropyFromDensityTemperature(const Real rho, const Real temp,
                                Indexer_t &&lambda = static_cast<Real *>(nullptr)) const;
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real EntropyFromDensityInternalEnergy(
      const Real rho, const Real sie,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const;
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real SpecificHeatFromDensityTemperature(
      const Real rho, const Real temp,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const {
    return ColdCv(temp);
  }
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real SpecificHeatFromDensityInternalEnergy(
      const Real rho, const Real sie,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const {
    return ColdCv(TemperatureFromDensityInternalEnergy(rho, sie));
  }
  // Thermal Bulk Modulus added AEM Dec 2022
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real TBulkModulusFromDensityTemperature(
      const Real rho, const Real temp,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const;
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real BulkModulusFromDensityTemperature(
      const Real rho, const Real temp,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const;
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real BulkModulusFromDensityInternalEnergy(
      const Real rho, const Real sie,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const;
  // Thermal expansion coefficient added AEM 2022
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real TExpansionCoeffFromDensityTemperature(
      const Real rho, const Real temp,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const;
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real GruneisenParamFromDensityTemperature(
      const Real rho, const Real temp,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const {
    return robust::ratio(_A0 * _B0, ColdCv(temp) * rho);
  }
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real GruneisenParamFromDensityInternalEnergy(
      const Real rho, const Real sie,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const {
    return robust::ratio(_A0 * _B0,
                         ColdCv(TemperatureFromDensityInternalEnergy(rho, sie)) * rho);
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

  // EOS depends on (rho0/rho)^(1/3) so this is a reasonable bound
  // before the EOS will become ill-behaved.
  PORTABLE_FORCEINLINE_FUNCTION
  Real MinimumDensity() const { return std::cbrt(10 * robust::EPS()) * _rho0; }

  // Essentially unbounded... I think.
  PORTABLE_FORCEINLINE_FUNCTION
  Real MinimumPressure() const { return -1e100; }
  PORTABLE_FORCEINLINE_FUNCTION
  Real MaximumPressureAtTemperature([[maybe_unused]] const Real T) const { return 1e100; }

  // Generic functions provided by the base class. These contain e.g. the vector
  // overloads that use the scalar versions declared here
  SG_ADD_BASE_CLASS_USINGS(Vinet)
  SG_ADD_DEFAULT_MEAN_ATOMIC_FUNCTIONS(_AZbar)

  static constexpr unsigned long PreferredInput() { return _preferred_input; }
  PORTABLE_INLINE_FUNCTION void PrintParams() const {
    static constexpr char st[]{"Vinet Params: "};
    printf("%s rho0:%e T0:%e B0:%e BP0:%e\n  A0:%e Cv0:%e E0:%e S0:%e\n"
           "  k_min:%e t_floor:%e rho_lim:%e\n"
           "non-zero elements in d2tod40 array:\n",
           st, _rho0, _T0, _B0, _BP0, _A0, _Cv0, _E0, _S0, _k_min, _t_floor, _rho_lim);
    for (int i = 0; i < 39; i++) {
      if (_d2tod40[i] > 0.0) printf("d%i:%e\t", i + 2, _d2tod40[i]);
    }
    printf("\n\n");
  }
  inline void Finalize() {}
  static std::string EosType() { return std::string("Vinet"); }
  static std::string EosPyType() { return EosType(); }

 private:
  static constexpr const unsigned long _preferred_input =
      thermalqs::density | thermalqs::temperature;
  Real _rho0, _T0, _B0, _BP0, _A0, _Cv0, _E0, _S0;
  MeanAtomicProperties _AZbar;
  static constexpr const int PressureCoeffsd2tod40Size = 39;
  static constexpr const int VinetInternalParametersSize = PressureCoeffsd2tod40Size + 4;
  Real _VIP[VinetInternalParametersSize], _d2tod40[PressureCoeffsd2tod40Size];
  // Extension state; _k_min = 0 / _t_floor = 0 disable the tension / cold extensions
  Real _k_min = 0.0, _t_floor = 0.0;
  Real _rho_lim = 0.0, _p_lim = 0.0, _e_lim = 0.0;
  void InitializeVinet(const Real *expcoeffs);
  void InitializeExtension();
  // Specific heat at temperature T (reduced in the cold-floor region)
  PORTABLE_FORCEINLINE_FUNCTION Real ColdCv(const Real temp) const {
    if (_t_floor > 0.0 && temp < _t_floor && temp > 0.0) {
      const Real r = _t_floor / temp;
      return _Cv0 * r * r;
    }
    return _Cv0;
  }
  // T0 isotherm of the unextended model: {energy (no E0), pressure, dp/drho}
  PORTABLE_INLINE_FUNCTION void Vinet_Ref(const Real rho, Real *ref) const;
  // Same, including the tension extension
  PORTABLE_INLINE_FUNCTION void Vinet_Extended_Ref(const Real rho, Real *ref) const;
  PORTABLE_INLINE_FUNCTION void Vinet_F_DT_func(const Real rho, const Real T,
                                                Real *output) const;
};

PORTABLE_INLINE_FUNCTION void Vinet::CheckParams() const {

  if (_rho0 < 0.0) {
    PORTABLE_ALWAYS_THROW_OR_ABORT("Required Vinet model parameter rho0 < 0");
  }
  if (_T0 < 0.0) {
    PORTABLE_ALWAYS_THROW_OR_ABORT("Required Vinet model parameter T0 < 0");
  }
  if (_B0 < 0.0) {
    PORTABLE_ALWAYS_THROW_OR_ABORT("Required Vinet model parameter B0 < 0");
  }
  if (_BP0 < 1.0) {
    PORTABLE_ALWAYS_THROW_OR_ABORT("Required Vinet model parameter BP0 < 1");
  }
  if (_A0 < 0.0) {
    PORTABLE_ALWAYS_THROW_OR_ABORT("Required Vinet model parameter A0 < 0");
  }
  if (_k_min < 0.0) {
    PORTABLE_ALWAYS_THROW_OR_ABORT("Required Vinet model parameter k_min < 0");
  }
  if (_t_floor < 0.0) {
    PORTABLE_ALWAYS_THROW_OR_ABORT("Required Vinet model parameter t_floor < 0");
  }
  if (_Cv0 < 0.0) {
    PORTABLE_ALWAYS_THROW_OR_ABORT("Required Vinet model parameter Cv0 < 0");
  }
  _AZbar.CheckParams();
}

inline void Vinet::InitializeVinet(const Real *d2tod40input) {

  // The PressureCoeffsd2tod40Size (=39) allowed d2 to d40 coefficients
  // for the pressure reference curve vs rho
  // are seldom all used so did not want to crowd the argument list with them.
  // Instead I ask the host code to send me a pointer to this array so that I can
  // copy it here. Not used coeffs should be set to 0.0 (of course).
  for (int ind = 0; ind < PressureCoeffsd2tod40Size; ind++) {
    _d2tod40[ind] = d2tod40input[ind];
  }
  // Put a couple of  much used model parameter combinations and
  // the energy coefficients f0 to f40 in an internal parameters array
  // of size 2+41=VinetInternalParametersSize.
  _VIP[0] = robust::ratio(_B0, _rho0) +
            robust::ratio(_A0 * _A0 * _B0 * _B0 * _T0,
                          _rho0 * _rho0 * _Cv0); // sound speed squared
  _VIP[1] = 3.0 / 2.0 * (_BP0 - 1.0);            // exponent eta0
  // initializing with pressure coeffs to get energy coeffs into VIP
  _VIP[2] = 1.0;                                                // prefactor d0
  _VIP[3] = 0.0;                                                // prefactor d1
  for (int ind = 4; ind < VinetInternalParametersSize; ind++) { //_d2tod40[0]=d2
    _VIP[ind] = _d2tod40[ind - 4];                              // dn is in VIP[n+2]
  }
  for (int ind = VinetInternalParametersSize - 2; ind >= 2;
       ind--) { // _VIP[42]=d40=f40 given,first calculated is _VIP[41]=f39
    _VIP[ind] = _VIP[ind] - (ind) / _VIP[1] * _VIP[ind + 1]; // prefactors f40 to f0
  }                                                          // _VIP[n+2]=fn, ind=n+2
}

PORTABLE_INLINE_FUNCTION void Vinet::Vinet_Ref(const Real rho, Real *ref) const {
  constexpr int pref0vp = -2, pref0vip = 2, maxind = VinetInternalParametersSize - 3;
  Real sumP = 0.0, sumB = 0.0, sumE = 0.0;

  Real x = std::cbrt(robust::ratio(_rho0, rho)); /*rho-dependent*/
  Real x2inv = robust::ratio(1.0, x * x);
  Real onemx = 1.0 - x;
  Real etatimes1mx = _VIP[1] * onemx;
  Real expetatimes1mx = exp(etatimes1mx);

#pragma unroll
  for (int ind = maxind; ind >= 2; ind--) {              //_d2tod40[0]=d2
    sumP = _d2tod40[pref0vp + ind] + onemx * sumP;       //_d2tod40[38]=d40
    sumB = _d2tod40[pref0vp + ind] * ind + onemx * sumB; //_d2tod40[-2+40]=d40
    sumE = _VIP[pref0vip + ind] + onemx * sumE;          //_VIP[42]=f40
  }                                                      //_VIP[2]=f0
#pragma unroll
  for (int ind = 1; ind >= 0; ind--) {
    sumE = _VIP[pref0vip + ind] + onemx * sumE;
  }
  sumP = 1.0 + sumP * (onemx * onemx);
  sumB = sumB * onemx;
  sumE = _VIP[1] * onemx * sumE;

  /* T0 isotherm */
  ref[0] = 9.0 * robust::ratio(_B0, _VIP[1] * _VIP[1] * _rho0) *
               (_VIP[pref0vip] - expetatimes1mx * (_VIP[pref0vip] - sumE)) -
           (_A0 * _B0) * (robust::ratio(_T0, _rho0) - robust::ratio(_T0, rho));
  ref[1] = 3.0 * _B0 * x2inv * onemx * expetatimes1mx * sumP;
  Real temp = (1.0 + onemx * (_VIP[1] * x + 1.0)) * sumP + x * onemx * sumB;
  ref[2] = robust::ratio(_B0, _rho0) * x * expetatimes1mx * temp;
}

PORTABLE_INLINE_FUNCTION void Vinet::Vinet_Extended_Ref(const Real rho, Real *ref) const {
  if (_k_min > 0.0 && rho < _rho_lim) {
    const Real v = 1.0 / rho, v_lim = 1.0 / _rho_lim;
    const Real log_r = std::log(rho * v_lim);
    ref[0] = _e_lim + (_A0 * _B0 * _T0 - _p_lim) * (v - v_lim) +
             _k_min * (-v * log_r - v + v_lim);
    ref[1] = _p_lim + _k_min * log_r;
    ref[2] = _k_min / rho;
  } else {
    Vinet_Ref(rho, ref);
  }
}

inline void Vinet::InitializeExtension() {
  _rho_lim = 0.0;
  if (_k_min <= 0.0) return;
  auto excess = [&](const Real rho) {
    Real ref[3];
    Vinet_Ref(rho, ref);
    return rho * ref[2] - _k_min;
  };
  if (excess(_rho0) <= 0.0) {
    PORTABLE_ALWAYS_THROW_OR_ABORT("Vinet model parameter k_min must be below B0");
  }
  // Largest density below rho0 where K_T falls to k_min: scan, then bisect.
  constexpr int nscan = 2000;
  bool found = false;
  Real hi = _rho0, lo = _rho0;
  for (int i = 1; i < nscan; i++) {
    lo = _rho0 * (1.0 - 0.95 * i / (nscan - 1.0));
    if (excess(lo) <= 0.0) {
      found = true;
      break;
    }
    hi = lo;
  }
  if (!found) {
    PORTABLE_ALWAYS_THROW_OR_ABORT("Vinet: no spinodal found above 0.05 rho0");
  }
  for (int i = 0; i < 200 && hi - lo > 1e-16 * hi; i++) {
    const Real mid = 0.5 * (lo + hi);
    if (excess(mid) <= 0.0) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  _rho_lim = 0.5 * (lo + hi);
  Real ref[3];
  Vinet_Ref(_rho_lim, ref);
  _e_lim = ref[0];
  _p_lim = ref[1];
}

PORTABLE_INLINE_FUNCTION void Vinet::Vinet_F_DT_func(const Real rho, const Real T,
                                                     Real *output) const {
  Real ref[3];
  Vinet_Extended_Ref(rho, ref);
  Real energy_ref = ref[0];
  Real pressure = ref[1];
  const Real dpdrho = ref[2];

  /* Go to required temperature */
  Real energy = energy_ref + _Cv0 * (T - _T0) + _E0;
  pressure = pressure + (_A0 * _B0) * (T - _T0);
  Real dpdt = _A0 * _B0;
  Real dedt = _Cv0;
  Real dedrho = robust::ratio(pressure - T * (_A0 * _B0), rho * rho);
  const bool cold = (_t_floor > 0.0 && T > 0.0 && T < _t_floor);
  // cold floor: e = e_f(rho) + u, u = Cv0 t_floor (1 - t_floor/T) < 0
  const Real q = cold ? _t_floor / T : 1.0;
  Real u = 0.0;
  if (cold) {
    u = _Cv0 * _t_floor * (1.0 - q);
    energy = energy_ref + _E0 + _Cv0 * (_t_floor - _T0) + u;
    dedt = _Cv0 * q * q;
  }
  Real entropy;
  if (T < 0.0) {
#ifndef NDEBUG
    PORTABLE_WARN("Negative temperature input");
#endif // NDEBUG
    entropy = 0.0;
  } else if (cold) {
    entropy = (_A0 * _B0) * (robust::ratio(1.0, rho) - robust::ratio(1, _rho0)) + _S0 +
              _Cv0 * std::log(robust::ratio(_t_floor, _T0)) + u / _t_floor -
              u * u / (2.0 * _Cv0 * _t_floor * _t_floor);
  } else {
    entropy = (_A0 * _B0) * (robust::ratio(1.0, rho) - robust::ratio(1, _rho0)) +
              _Cv0 * std::log(robust::ratio(T, _T0)) + _S0;
  }
  Real soundspeed =
      std::max(0.0, dpdrho + T * robust::ratio(dpdt * dpdt, rho * rho * dedt));
  soundspeed = std::sqrt(soundspeed);

  output[0] = energy;
  output[1] = pressure;
  output[2] = dpdrho;
  output[3] = dpdt;
  output[4] = dedt;
  output[5] = dedrho;
  output[6] = entropy;
  output[7] = soundspeed;

  return;
}
template <typename Indexer_t>
PORTABLE_INLINE_FUNCTION Real Vinet::InternalEnergyFromDensityTemperature(
    const Real rho, const Real temp, Indexer_t &&lambda) const {
  Real output[8];
  Vinet_F_DT_func(rho, temp, output);
  return output[0];
}
template <typename Indexer_t>
PORTABLE_INLINE_FUNCTION Real Vinet::PressureFromDensityTemperature(
    const Real rho, const Real temp, Indexer_t &&lambda) const {
  Real output[8];
  Vinet_F_DT_func(rho, temp, output);
  return output[1];
}
template <typename Indexer_t>
PORTABLE_INLINE_FUNCTION Real Vinet::EntropyFromDensityTemperature(
    const Real rho, const Real temp, Indexer_t &&lambda) const {
  Real output[8];
  Vinet_F_DT_func(rho, temp, output);
  return output[6];
}
template <typename Indexer_t>
PORTABLE_INLINE_FUNCTION Real Vinet::TExpansionCoeffFromDensityTemperature(
    const Real rho, const Real temp, Indexer_t &&lambda) const {
  Real output[8];
  Vinet_F_DT_func(rho, temp, output);
  return robust::ratio(output[3], output[2] * rho);
}
template <typename Indexer_t>
PORTABLE_INLINE_FUNCTION Real Vinet::TBulkModulusFromDensityTemperature(
    const Real rho, const Real temp, Indexer_t &&lambda) const {
  Real output[8];
  Vinet_F_DT_func(rho, temp, output);
  return output[2] * rho;
}
template <typename Indexer_t>
PORTABLE_INLINE_FUNCTION Real Vinet::BulkModulusFromDensityTemperature(
    const Real rho, const Real temp, Indexer_t &&lambda) const {
  Real output[8];
  Vinet_F_DT_func(rho, temp, output);
  return output[7] * output[7] * rho;
}
template <typename Indexer_t>
PORTABLE_INLINE_FUNCTION Real Vinet::TemperatureFromDensityInternalEnergy(
    const Real rho, const Real sie, Indexer_t &&lambda) const {
  Real ref[3];
  Vinet_Extended_Ref(rho, ref);
  // Same operation order as the oracle: the energy offset e_ref is as large as 1e12 in
  // the tension extension, so the difference sie - e_ref cancels.
  const Real t_reg = _T0 + robust::ratio(sie - ref[0] - _E0, _Cv0);
  if (_t_floor > 0.0) {
    // cold floor: u = e - e_f(rho), T = t_floor/(1 - u/(Cv0 t_floor)) for u < 0
    const Real u = sie - (ref[0] + _E0 + _Cv0 * (_t_floor - _T0));
    if (u < 0.0) return _t_floor / (1.0 - u / (_Cv0 * _t_floor));
  }
  return t_reg;
}
template <typename Indexer_t>
PORTABLE_INLINE_FUNCTION Real Vinet::PressureFromDensityInternalEnergy(
    const Real rho, const Real sie, Indexer_t &&lambda) const {
  Real temp;
  Real output[8];
  temp = TemperatureFromDensityInternalEnergy(rho, sie);
  Vinet_F_DT_func(rho, temp, output);
  return output[1];
}
template <typename Indexer_t>
PORTABLE_INLINE_FUNCTION Real Vinet::EntropyFromDensityInternalEnergy(
    const Real rho, const Real sie, Indexer_t &&lambda) const {
  Real temp;
  Real output[8];
  temp = TemperatureFromDensityInternalEnergy(rho, sie);
  Vinet_F_DT_func(rho, temp, output);
  return output[6];
}
template <typename Indexer_t>
PORTABLE_INLINE_FUNCTION Real Vinet::BulkModulusFromDensityInternalEnergy(
    const Real rho, const Real sie, Indexer_t &&lambda) const {
  Real temp;
  Real output[8];
  temp = TemperatureFromDensityInternalEnergy(rho, sie);
  Vinet_F_DT_func(rho, temp, output);
  return output[7] * output[7] * rho;
}
// AEM: We should add entropy and Gruneissen parameters here so that it is complete
// If we add also alpha and BT, those should also be in here.
template <typename Indexer_t>
PORTABLE_INLINE_FUNCTION void
Vinet::FillEos(Real &rho, Real &temp, Real &sie, Real &press, Real &cv, Real &bmod,
               const unsigned long output, Indexer_t &&lambda) const {
  const unsigned long input = ~output; // everything that is not output is input
  if (thermalqs::density & output) {
    EOS_ERROR("Vinet FillEos: Density is required input.\n");
  } else if (thermalqs::temperature & output &&
             thermalqs::specific_internal_energy & output) {
    EOS_ERROR("Vinet FillEos: Density and Internal Energy or Density and Temperature "
              "are required input parameters.\n");
  }
  if (thermalqs::specific_internal_energy & input) {
    temp = TemperatureFromDensityInternalEnergy(rho, sie);
  }
  Real Vout[8];
  Vinet_F_DT_func(rho, temp, Vout);
  // if (output & thermalqs::temperature) temp = temp;
  if (output & thermalqs::specific_internal_energy) sie = Vout[0];
  if (output & thermalqs::pressure) press = Vout[1];
  if (output & thermalqs::specific_heat) cv = Vout[4];
  if (output & thermalqs::bulk_modulus) bmod = Vout[7] * Vout[7] * rho;
}

// TODO(JMM): pre-cache these rather than recomputing them each time
template <typename Indexer_t>
PORTABLE_INLINE_FUNCTION void
Vinet::ValuesAtReferenceState(Real &rho, Real &temp, Real &sie, Real &press, Real &cv,
                              Real &bmod, Real &dpde, Real &dvdt,
                              Indexer_t &&lambda) const {
  // AEM: Added all variables I think should be output eventually
  Real tbmod;
  // Real entropy, alpha, Gamma;

  rho = _rho0;
  temp = _T0;
  sie = _E0;
  press = PressureFromDensityTemperature(rho, temp, lambda);
  // entropy = _S0;
  cv = _Cv0;
  tbmod = _B0;
  // alpha = _A0;
  bmod = BulkModulusFromDensityTemperature(rho, temp, lambda);
  // Gamma = robust::ratio(_A0 * _B0, _Cv0 * _rho0);
  // AEM: I suggest taking the two following away.
  dpde = robust::ratio(_A0 * tbmod, _Cv0);
  dvdt = robust::ratio(_A0, _rho0);
}
} // namespace singularity

#endif // _SINGULARITY_EOS_EOS_VINET_HPP_
