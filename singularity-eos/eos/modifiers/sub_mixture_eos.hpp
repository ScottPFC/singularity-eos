//------------------------------------------------------------------------------
// SubMixtureEOS: an ideal-solution group of alike species served as ONE material.
//
// Pacific Fusion addition. See scaled_eos.hpp for the fixed-scale sibling.
//------------------------------------------------------------------------------

#ifndef _SINGULARITY_EOS_EOS_SUB_MIXTURE_EOS_
#define _SINGULARITY_EOS_EOS_SUB_MIXTURE_EOS_

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>

#include <ports-of-call/portability.hpp>
#include <ports-of-call/portable_errors.hpp>
#include <singularity-eos/base/constants.hpp>
#include <singularity-eos/base/eos_error.hpp>
#include <singularity-eos/base/indexable_types.hpp>
#include <singularity-eos/base/robust_utils.hpp>
#include <singularity-eos/eos/eos_base.hpp>

namespace singularity {

using namespace eos_base;

// A set of species whose EOSs are all ScaledEOS of a COMMON base -- isotopes of one element, or
// a filler declared as a rescaling of its host -- closed at common pressure and temperature.
//
// THE POINT. That closure has a closed form. If member m has EOS ScaledEOS(base, s_m), then at
// common pressure s_m * rho_m is the same for every m, and the group collapses to
//
//     P_group(rho,T) = P_base(S_eff * rho, T)
//     e_group(rho,T) = S_eff * e_base(S_eff * rho, T)      S_eff = sum_m Y_m s_m
//
// which is the SAME transform ScaledEOS implements. So an ideal-solution group is not a mixture
// solve at all -- it is one table lookup. Verified against SESAME 5268 D/T on Pacific Fusion's
// V23 tables to machine precision (0 to 3e-16 in BOTH P and e, from 17 K ice to 1e4 K gas), and
// S_eff = 0.802820 there matches the independent molar-mass form M_D/Mbar to seven figures.
//
// WHY NOT JUST USE ScaledEOS. Because S_eff is a mass-fraction weighted mean, so it moves with
// the LOCAL composition: burn consumes D and T at different rates, and a filler mixes into its
// host in varying proportion. ScaledEOS takes its scale at construction, so one handle could
// only ever serve one composition. Here the scale is read from the lambda on every call, so ONE
// handle serves every cell and the caller supplies S_eff per evaluation.
//
// The lambda already reaches every accessor inside the PTE solver
// (mixed_cell_models.hpp evaluates eos[m].PressureFromDensityTemperature(rho, T, lambda[m])),
// so this needs no change to the closure machinery.
//
// FALLBACK. IndexerUtils::SafeGet returns false for a null lambda, in which case `default_scale`
// is used. That keeps single-material paths -- which pass no lambda -- working unchanged, and
// makes a group of one member (default_scale = 1) exactly its base.
template <typename T>
class SubMixtureEOS : public EosBase<SubMixtureEOS<T>> {
 public:
  SG_ADD_BASE_CLASS_USINGS(SubMixtureEOS<T>);
  using BaseType = T;

  static std::string EosType() {
    return std::string("SubMixtureEOS<") + T::EosType() + std::string(">");
  }
  static std::string EosPyType() { return std::string("SubMixture") + T::EosPyType(); }

  SubMixtureEOS() = default;
  //! `scale_lo`/`scale_hi` bracket every S_eff the group can ever present -- see MinimumDensity.
  PORTABLE_FUNCTION
  SubMixtureEOS(T &&t, const Real default_scale = 1.0, const std::size_t scale_idx = 0,
                const Real scale_lo = 1.0, const Real scale_hi = 1.0)
      : t_(std::forward<T>(t)), default_scale_(default_scale), scale_idx_(scale_idx),
        scale_lo_(scale_lo), scale_hi_(scale_hi) {
    CheckParams();
  }

  PORTABLE_INLINE_FUNCTION void CheckParams() const {
    PORTABLE_ALWAYS_REQUIRE(default_scale_ > 0, "Default sub-mixture scale must be positive.");
    PORTABLE_ALWAYS_REQUIRE(!std::isnan(default_scale_),
                            "Default sub-mixture scale must be well defined.");
    PORTABLE_ALWAYS_REQUIRE(scale_lo_ > 0 && scale_hi_ >= scale_lo_,
                            "Sub-mixture scale bracket must be positive and ordered.");
    t_.CheckParams();
  }

  auto GetOnDevice() {
    return SubMixtureEOS<T>(t_.GetOnDevice(), default_scale_, scale_idx_, scale_lo_, scale_hi_);
  }
  inline void Finalize() { t_.Finalize(); }

  // S_eff for this call. Anything outside [scale_lo_, scale_hi_] is REFUSED rather than
  // propagated: it would otherwise reach the tables as a wrong density and come back as a
  // plausible number with no error signal.
  //
  // The bracket, not just positivity, is the test. S_eff is a mass-fraction weighted mean of the
  // members' scales, so it CANNOT leave [min_m s_m, max_m s_m] -- a value outside is proof the
  // lambda slot was not the one FLASH wrote. That is not hypothetical: `SubMixLambda` decays to
  // `double*` for EOSs that declare `Real *lambda` concretely (BilinearRampEOS, which the PTE
  // solvers call), and on that path the TYPED accessor is unavailable and SafeGet falls back to
  // the numeric `scale_idx_`. When that index was off by one it landed on the next material's
  // slot 0 -- SpinerEOSDependsRhoSie's cached log-density -- which is positive often enough to
  // sail through a bare `s > 0` guard and silently probe the base at several times the true
  // density. A bracket check turns that into a correct fallback instead of a wrong answer.
  template <typename Indexer_t>
  PORTABLE_FORCEINLINE_FUNCTION Real Scale(Indexer_t &&lambda) const {
    Real s = default_scale_;
    IndexerUtils::SafeGet<IndexableTypes::SubMixtureScale>(lambda, scale_idx_, s);
    if (std::isnan(s) || s < scale_lo_ || s > scale_hi_) return default_scale_;
    return s;
  }

  template <typename Indexer_t = Real *>
  PORTABLE_FUNCTION Real TemperatureFromDensityInternalEnergy(
      const Real rho, const Real sie, Indexer_t &&lambda = nullptr) const {
    const Real s = Scale(lambda);
    return t_.TemperatureFromDensityInternalEnergy(s * rho, robust::ratio(sie, s), lambda);
  }
  template <typename Indexer_t = Real *>
  PORTABLE_FUNCTION Real InternalEnergyFromDensityTemperature(
      const Real rho, const Real temperature, Indexer_t &&lambda = nullptr) const {
    const Real s = Scale(lambda);
    return s * t_.InternalEnergyFromDensityTemperature(s * rho, temperature, lambda);
  }
  template <typename Indexer_t = Real *>
  PORTABLE_FUNCTION Real PressureFromDensityInternalEnergy(
      const Real rho, const Real sie, Indexer_t &&lambda = nullptr) const {
    const Real s = Scale(lambda);
    return t_.PressureFromDensityInternalEnergy(s * rho, robust::ratio(sie, s), lambda);
  }
  template <typename Indexer_t = Real *>
  PORTABLE_FUNCTION Real PressureFromDensityTemperature(
      const Real rho, const Real temperature, Indexer_t &&lambda = nullptr) const {
    const Real s = Scale(lambda);
    return t_.PressureFromDensityTemperature(s * rho, temperature, lambda);
  }
  template <typename Indexer_t = Real *>
  PORTABLE_FUNCTION Real
  MinInternalEnergyFromDensity(const Real rho, Indexer_t &&lambda = nullptr) const {
    const Real s = Scale(lambda);
    return s * t_.MinInternalEnergyFromDensity(s * rho, lambda);
  }
  template <typename Indexer_t = Real *>
  PORTABLE_FUNCTION Real EntropyFromDensityInternalEnergy(
      const Real rho, const Real sie, Indexer_t &&lambda = nullptr) const {
    const Real s = Scale(lambda);
    return s * t_.EntropyFromDensityInternalEnergy(s * rho, robust::ratio(sie, s), lambda);
  }
  template <typename Indexer_t = Real *>
  PORTABLE_FUNCTION Real EntropyFromDensityTemperature(
      const Real rho, const Real temperature, Indexer_t &&lambda = nullptr) const {
    const Real s = Scale(lambda);
    return s * t_.EntropyFromDensityTemperature(s * rho, temperature, lambda);
  }
  // c_v = de/dT, and e carries S_eff while T does not, so c_v carries it too.
  template <typename Indexer_t = Real *>
  PORTABLE_FUNCTION Real SpecificHeatFromDensityInternalEnergy(
      const Real rho, const Real sie, Indexer_t &&lambda = nullptr) const {
    const Real s = Scale(lambda);
    return s * t_.SpecificHeatFromDensityInternalEnergy(s * rho, robust::ratio(sie, s), lambda);
  }
  template <typename Indexer_t = Real *>
  PORTABLE_FUNCTION Real SpecificHeatFromDensityTemperature(
      const Real rho, const Real temperature, Indexer_t &&lambda = nullptr) const {
    const Real s = Scale(lambda);
    return s * t_.SpecificHeatFromDensityTemperature(s * rho, temperature, lambda);
  }
  // Bulk modulus absorbs the density factor -- B = rho dP/drho -- and Gruneisen cancels it
  // against the energy factor, so BOTH are unscaled on output.
  template <typename Indexer_t = Real *>
  PORTABLE_FUNCTION Real BulkModulusFromDensityInternalEnergy(
      const Real rho, const Real sie, Indexer_t &&lambda = nullptr) const {
    const Real s = Scale(lambda);
    return t_.BulkModulusFromDensityInternalEnergy(s * rho, robust::ratio(sie, s), lambda);
  }
  template <typename Indexer_t = Real *>
  PORTABLE_FUNCTION Real BulkModulusFromDensityTemperature(
      const Real rho, const Real temperature, Indexer_t &&lambda = nullptr) const {
    const Real s = Scale(lambda);
    return t_.BulkModulusFromDensityTemperature(s * rho, temperature, lambda);
  }
  template <typename Indexer_t = Real *>
  PORTABLE_FUNCTION Real GruneisenParamFromDensityInternalEnergy(
      const Real rho, const Real sie, Indexer_t &&lambda = nullptr) const {
    const Real s = Scale(lambda);
    return t_.GruneisenParamFromDensityInternalEnergy(s * rho, robust::ratio(sie, s), lambda);
  }
  template <typename Indexer_t = Real *>
  PORTABLE_FUNCTION Real GruneisenParamFromDensityTemperature(
      const Real rho, const Real temperature, Indexer_t &&lambda = nullptr) const {
    const Real s = Scale(lambda);
    return t_.GruneisenParamFromDensityTemperature(s * rho, temperature, lambda);
  }
  template <typename Indexer_t = Real *>
  PORTABLE_FUNCTION void
  InternalEnergyFromDensityPressure(const Real rho, const Real P, Real &sie,
                                    Indexer_t &&lambda = nullptr) const {
    const Real s = Scale(lambda);
    t_.InternalEnergyFromDensityPressure(s * rho, P, sie, lambda);
    sie *= s;
  }
  template <typename Indexer_t = Real *>
  PORTABLE_FUNCTION void FillEos(Real &rho, Real &temp, Real &energy, Real &press, Real &cv,
                                 Real &bmod, const unsigned long output,
                                 Indexer_t &&lambda = nullptr) const {
    const Real s = Scale(lambda);
    Real srho, senergy;
    switch (t_.PreferredInput()) {
    case thermalqs::density | thermalqs::temperature:
      srho = s * rho;
      t_.FillEos(srho, temp, energy, press, cv, bmod, output, lambda);
      energy = s * energy;
      cv = s * cv;
      break;
    case thermalqs::density | thermalqs::specific_internal_energy:
      srho = s * rho;
      senergy = robust::ratio(energy, s);
      t_.FillEos(srho, temp, senergy, press, cv, bmod, output, lambda);
      cv = s * cv;
      break;
    default:
      EOS_ERROR("Didn't find a valid input for SubMixtureEOS::FillEOS\n");
    }
  }
  template <typename Indexer_t = Real *>
  PORTABLE_FUNCTION void ValuesAtReferenceState(Real &rho, Real &temp, Real &sie, Real &press,
                                                Real &cv, Real &bmod, Real &dpde, Real &dvdt,
                                                Indexer_t &&lambda = nullptr) const {
    const Real s = Scale(lambda);
    t_.ValuesAtReferenceState(rho, temp, sie, press, cv, bmod, dpde, dvdt, lambda);
    rho = robust::ratio(rho, s);
    sie *= s;
    cv *= s;
  }
  template <typename Indexer_t = Real *>
  PORTABLE_FUNCTION void DensityEnergyFromPressureTemperature(
      const Real press, const Real temp, Indexer_t &&lambda, Real &rho, Real &sie) const {
    const Real s = Scale(lambda);
    t_.DensityEnergyFromPressureTemperature(press, temp, lambda, rho, sie);
    rho = robust::ratio(rho, s);
    sie *= s;
  }

  // Density bounds take NO lambda, so they cannot know the cell's S_eff -- but they do not need
  // to. S_eff = sum_m Y_m s_m is a weighted MEAN of the member scales, so for ANY composition it
  // lies in [min_m s_m, max_m s_m]. The group at density rho is evaluated at S_eff*rho, so its
  // reachable density range is bounded exactly by
  //
  //     [ base_min / scale_hi ,  base_max / scale_lo ]
  //
  // which is correct for every composition rather than for one nominal one.
  //
  // The direction matters. `PTESolverBase` clamps with `min(rho[m], MaximumDensity())`, so a
  // bound that is too SMALL forces a legitimate state down and pushes the solve off-manifold;
  // one that is merely loose lets the base EOS clamp internally, exactly as an ungrouped
  // material would. Reporting the DEFAULT scale here made the D+T group's maximum 25% too low
  // and produced 26,566 "PTE result OFF-MANIFOLD (accepted despite converged flag)" with a
  // timestep collapse from 5.6e-12 to 2.6e-14 s; the identity group, where the bracket is
  // degenerate, logged none.
  PORTABLE_INLINE_FUNCTION Real MinimumDensity() const {
    return robust::ratio(t_.MinimumDensity(), scale_hi_);
  }
  PORTABLE_INLINE_FUNCTION Real MaximumDensity() const {
    return robust::ratio(t_.MaximumDensity(), scale_lo_);
  }
  PORTABLE_INLINE_FUNCTION Real MinimumTemperature() const { return t_.MinimumTemperature(); }
  PORTABLE_INLINE_FUNCTION Real MinimumPressure() const { return t_.MinimumPressure(); }
  PORTABLE_INLINE_FUNCTION Real MaximumPressureAtTemperature(const Real temp) const {
    return t_.MaximumPressureAtTemperature(temp);
  }

  // Mean atomic mass is a per-particle quantity, so it carries 1/S_eff -- for a D+T group that
  // is exactly Mbar = M_D/S_eff, the composition-weighted mean the group represents. Atomic
  // NUMBER is unchanged: the members are isotopes or rescalings, not different elements.
  PORTABLE_INLINE_FUNCTION
  Real MeanAtomicMass() const { return robust::ratio(t_.MeanAtomicMass(), default_scale_); }
  PORTABLE_INLINE_FUNCTION
  Real MeanAtomicNumber() const { return t_.MeanAtomicNumber(); }

  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real MeanAtomicMassFromDensityTemperature(
      const Real rho, const Real temperature,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const {
    const Real s = Scale(lambda);
    return robust::ratio(
        t_.MeanAtomicMassFromDensityTemperature(s * rho, temperature, lambda), s);
  }
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real MeanAtomicNumberFromDensityTemperature(
      const Real rho, const Real temperature,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const {
    const Real s = Scale(lambda);
    return t_.MeanAtomicNumberFromDensityTemperature(s * rho, temperature, lambda);
  }

  PORTABLE_INLINE_FUNCTION int nlambda() const noexcept { return t_.nlambda(); }
  template <typename Indexable>
  static inline constexpr bool NeedsLambda() {
    return T::template NeedsLambda<Indexable>();
  }
  static constexpr unsigned long PreferredInput() { return T::PreferredInput(); }
  PORTABLE_INLINE_FUNCTION void PrintParams() const {
    printf("Sub-mixture group, default S_eff = %g, lambda slot %ld of\n", default_scale_,
           static_cast<long>(scale_idx_));
    t_.PrintParams();
  }
  // IsModified / UnmodifyOnce / GetUnmodifiedObject come from the macro; defining them by hand
  // as well is a redefinition, not an override.
  SG_ADD_MODIFIER_METHODS(T, t_);

 private:
  T t_;
  Real default_scale_;
  std::size_t scale_idx_;
  //! Smallest and largest member scale; brackets every S_eff the group can present.
  Real scale_lo_;
  Real scale_hi_;
};

} // namespace singularity

#endif // _SINGULARITY_EOS_EOS_SUB_MIXTURE_EOS_
