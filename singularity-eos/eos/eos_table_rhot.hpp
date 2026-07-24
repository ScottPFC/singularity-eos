//------------------------------------------------------------------------------
// © 2021-2026. Triad National Security, LLC. All rights reserved.  This
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

#ifndef _SINGULARITY_EOS_EOS_EOS_TABLE_RHOT_HPP_
#define _SINGULARITY_EOS_EOS_EOS_TABLE_RHOT_HPP_

// A (rho,T)-tabulated EOS that answers the cyclic PTE closure's (P,T) primitives by a LIVE
// dense-branch density root on the full (rho,T) surface -- the C++ port of the validated
// offline adapter pfc/sim/matdata/_pte_rhot.py::RhoTPteEos.
//
// Motivation.  The sibling TableDependsPT pre-inverts the (rho,T) surface onto a (P,T) grid
// once, at build time.  That inversion can only span the log-representable, monotone,
// POSITIVE-pressure region (a log-P grid needs P>0, and rho(P;T) must be single valued), so a
// cold condensed material -- whose stable branch extends through P=0 into tension -- has its
// low-positive-P shoulder truncated (e.g. copper's fluid (P,T) table floors at P>=1.78 GPa,
// T>=634 K).  When the mixed cyclic PTE then needs a cold, sub-GPa liner state it clamps the
// material to that corner and the mixture residual can never close (the RUN013 stall).
//
// This model instead keeps the FULL (rho,T) table and roots at query time, exactly as the
// offline solver does.  It NEVER truncates: any (P,T) the solver proposes is answered by the
// densest INCREASING crossing of P(rho;T)=P (dP/drho>0), i.e. the condensed/stable branch.
// That single convention gives the three things the Clayton cyclic method needs
// (arXiv:2606.27726, Thm 2.26): (i) a UNIQUE root, (ii) on the MONOTONE dP/drho>0 region, so
// (iii) the chain-ruled (P,T) partials dtau/dP = -1/(rho^2 P_rho) are well conditioned
// (P_rho>0).  MinimumPressure() is reported as 0 so the solver's box clamp cannot drive the
// state into tension (the PTE equilibrium of a condensed material with any gas present is
// positive-P by construction).
//
// The stored surface is the SAME per-phase, Maxwell-consistent (rho,T) data the single-material
// HEM path already uses (the `_rhot` companion): P(rho,T), e(rho,T) and the four (rho,T) first
// partials dP/drho, dP/dT, de/drho, de/dT, so the chain-rule Jacobian uses tabulated analytic
// partials (no finite differencing).  Interpolation is bilinear on the native (non-uniform,
// log) (rho,T) node grid, matching RhoTPteEos.evaluate_at.
//
// Units: table-native (GPa, Mg/m^3 == g/cm^3, MJ/kg), as written by _sp5.py; the
// FLASH<->singularity boundary reconciles unit systems (all mixture materials share units, so
// the closure is unit-system-agnostic internally).  See eos_table_pt.hpp for the sibling.

#ifdef SINGULARITY_USE_PT_TABLES
#ifndef SINGULARITY_USE_SPINER_WITH_HDF5
#error "SINGULARITY_USE_PT_TABLES requires SINGULARITY_USE_SPINER_WITH_HDF5"
#endif

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <hdf5.h>
#include <hdf5_hl.h>

// ports-of-call
#include <ports-of-call/portability.hpp>
#include <ports-of-call/portable_errors.hpp>

// base
#include <singularity-eos/base/constants.hpp>
#include <singularity-eos/base/robust_utils.hpp>
#include <singularity-eos/base/sp5/singularity_eos_sp5.hpp>
#include <singularity-eos/base/spiner_table_utils.hpp>
#include <singularity-eos/eos/eos_base.hpp>
#include <singularity-eos/eos/eos_spiner_common.hpp>

// spiner
#include <spiner/databox.hpp>

namespace singularity {

using namespace eos_base;

class TableDependsRhoT : public EosBase<TableDependsRhoT> {
  friend class table_utils::SpinerTricks<TableDependsRhoT>;
  using SpinerTricks = table_utils::SpinerTricks<TableDependsRhoT>;

 public:
  using DataBox = spiner_common::DataBox;

  // Lambda holds a warm-start (P,T) guess (mirrors TableDependsPT::Lambda).
  struct Lambda {
    enum Index { P = 0, T = 1 };
  };

  SG_ADD_DEFAULT_MEAN_ATOMIC_FUNCTIONS(AZbar_)
  SG_ADD_BASE_CLASS_USINGS(TableDependsRhoT);

  PORTABLE_INLINE_FUNCTION
  TableDependsRhoT() : memoryStatus_(DataStatus::Deallocated) {}

  inline TableDependsRhoT(const std::string &filename, int matid);
  inline TableDependsRhoT(const std::string &filename, const std::string &materialName);

  inline TableDependsRhoT GetOnDevice() { return SpinerTricks::GetOnDevice(this); }
  inline void Finalize() { SpinerTricks::Finalize(this); }
  std::size_t DynamicMemorySizeInBytes() const {
    return SpinerTricks::DynamicMemorySizeInBytes(this);
  }
  std::size_t DumpDynamicMemory(char *dst) {
    return SpinerTricks::DumpDynamicMemory(dst, this);
  }
  std::size_t SetDynamicMemory(char *src,
                               const SharedMemSettings &stngs = DEFAULT_SHMEM_STNGS) {
    if (stngs.data != nullptr) sharedMemory_ = stngs.data;
    return SpinerTricks::SetDynamicMemory(src, this);
  }

  PORTABLE_INLINE_FUNCTION void CheckParams() const {
    PORTABLE_ALWAYS_REQUIRE(numRho_ > 1, "At least two density points");
    PORTABLE_ALWAYS_REQUIRE(numT_ > 1, "At least two temperature points");
    PORTABLE_ALWAYS_REQUIRE(rhoMax_ > rhoMin_, "Density bounds ordered");
    PORTABLE_ALWAYS_REQUIRE(Tmax_ > Tmin_, "Temperature bounds ordered");
  }

  // ---- the primitive PTESolverPTCyclic drives: (P,T) -> (rho, e) ----
  // Dense-branch root on the full (rho,T) surface (port of RhoTPteEos._density_at + evaluate).
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION void
  DensityEnergyFromPressureTemperature(const Real press, const Real temp,
                                       Indexer_t &&lambda, Real &rho, Real &sie) const;

  // (P,T) -> (rho, e) plus the four (P,T) first partials, chain-ruled from the tabulated
  // (rho,T) partials at the dense-branch root (port of RhoTPteEos.evaluate):
  //   dtau/dP = -1/(rho^2 P_rho),  dtau/dT = (P_T/P_rho)/rho^2,
  //   de/dP   = e_rho/P_rho,       de/dT   = e_T - e_rho P_T / P_rho.
  // Returned as (drho/dP, drho/dT, de/dP, de/dT) to match the TableDependsPT signature the
  // cyclic mixture_() consumes; drho = -dtau/tau^2 * ... i.e. drho/dX = -rho^2 * dtau/dX.
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION void DensityEnergyDerivativesFromPressureTemperature(
      const Real press, const Real temp, Indexer_t &&lambda, Real &rho, Real &sie,
      Real &drho_dP, Real &drho_dT, Real &de_dP, Real &de_dT) const;

  // ---- native (rho,T) / (rho,e) entry points: direct bilinear on the stored surface ----
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real InternalEnergyFromDensityTemperature(
      const Real rho, const Real temperature,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const;
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real PressureFromDensityTemperature(
      const Real rho, const Real temperature,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const;
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real SpecificHeatFromDensityTemperature(
      const Real rho, const Real temperature,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const;
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real BulkModulusFromDensityTemperature(
      const Real rho, const Real temperature,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const;
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real TemperatureFromDensityInternalEnergy(
      const Real rho, const Real sie,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const;

  // Aux methods not needed by the cyclic PT closure; fail loudly if reached (mirror sibling).
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real
  EntropyFromDensityTemperature(const Real, const Real,
                                Indexer_t && = static_cast<Real *>(nullptr)) const {
    PORTABLE_ALWAYS_THROW_OR_ABORT("Entropy not implemented for TableDependsRhoT");
    return 0.0;
  }
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real
  GruneisenParamFromDensityTemperature(const Real, const Real,
                                       Indexer_t && = static_cast<Real *>(nullptr)) const {
    PORTABLE_ALWAYS_THROW_OR_ABORT("Gruneisen not implemented for TableDependsRhoT");
    return 0.0;
  }

  // (rho, sie) aux variants (mirror TableDependsPT): pressure/heat go through the T inversion;
  // entropy/gruneisen fail loudly.
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real PressureFromDensityInternalEnergy(
      const Real rho, const Real sie,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const {
    return PressureFromDensityTemperature(
        rho, TemperatureFromDensityInternalEnergy(rho, sie, lambda), lambda);
  }
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real SpecificHeatFromDensityInternalEnergy(
      const Real rho, const Real sie,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const {
    return SpecificHeatFromDensityTemperature(
        rho, TemperatureFromDensityInternalEnergy(rho, sie, lambda), lambda);
  }
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real
  EntropyFromDensityInternalEnergy(const Real, const Real,
                                   Indexer_t && = static_cast<Real *>(nullptr)) const {
    PORTABLE_ALWAYS_THROW_OR_ABORT("Entropy not implemented for TableDependsRhoT");
    return 0.0;
  }
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real BulkModulusFromDensityInternalEnergy(
      const Real rho, const Real sie,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const {
    return BulkModulusFromDensityTemperature(
        rho, TemperatureFromDensityInternalEnergy(rho, sie, lambda), lambda);
  }
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real
  GruneisenParamFromDensityInternalEnergy(const Real, const Real,
                                          Indexer_t && = static_cast<Real *>(nullptr)) const {
    PORTABLE_ALWAYS_THROW_OR_ABORT("Gruneisen not implemented for TableDependsRhoT");
    return 0.0;
  }

  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION void
  FillEos(Real &rho, Real &temp, Real &energy, Real &press, Real &cv, Real &bmod,
          const unsigned long output,
          Indexer_t &&lambda = static_cast<Real *>(nullptr)) const;

  // Reference state (~room T at normal density) for FLASH normalization / initial guesses.
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION void
  ValuesAtReferenceState(Real &rho, Real &temp, Real &sie, Real &press, Real &cv, Real &bmod,
                         Real &dpde, Real &dvdt,
                         Indexer_t &&lambda = static_cast<Real *>(nullptr)) const {
    const Real T_ref = clampT_(298.15);
    const Real rho_ref =
        (normalDensity_ > 0.0) ? normalDensity_ : std::sqrt(rhoMin_ * rhoMax_);
    const Real P_ref = PressureFromDensityTemperature(rho_ref, T_ref, lambda);
    Real r, e, drho_dP, drho_dT, de_dP, de_dT;
    DensityEnergyDerivativesFromPressureTemperature(P_ref, T_ref, lambda, r, e, drho_dP,
                                                    drho_dT, de_dP, de_dT);
    rho = r;
    temp = T_ref;
    sie = e;
    press = P_ref;
    cv = de_dT;
    bmod = r / robust::make_positive(drho_dP);
    dpde = robust::ratio(-robust::ratio(drho_dT, drho_dP), de_dT);
    dvdt = -robust::ratio(drho_dT, r * r);
  }

  static constexpr unsigned long PreferredInput() { return _preferred_input; }
  int matid() const { return matid_; }

  PORTABLE_FORCEINLINE_FUNCTION Real MinimumDensity() const { return rhoMin_; }
  PORTABLE_FORCEINLINE_FUNCTION Real MaximumDensity() const { return rhoMax_; }
  PORTABLE_FORCEINLINE_FUNCTION Real MinimumTemperature() const { return Tmin_; }
  PORTABLE_FORCEINLINE_FUNCTION Real MaximumTemperature() const { return Tmax_; }
  // Report a NON-NEGATIVE minimum pressure so the cyclic solver's box clamp
  // (PTESolverPT::ScaleDx / PTESolverPTCyclic::ClampToBox_) keeps the state on the stable,
  // positive-P branch and out of tension -- the physical PTE equilibrium of a condensed
  // material mixed with any gas is P>0.  (The surface itself still carries the tension branch;
  // we simply never let the SOLVER go there, matching the offline pressure_bounds floor.)
  PORTABLE_FORCEINLINE_FUNCTION Real MinimumPressure() const { return 0.0; }
  // Highest achievable pressure at T: the dense-grid-end pressure (condensed branch top).
  PORTABLE_FORCEINLINE_FUNCTION Real MaximumPressureAtTemperature(const Real temp) const {
    return pressureAtRhoIndexT_(numRho_ - 1, clampT_(temp));
  }

  PORTABLE_INLINE_FUNCTION void PrintParams() const {
    printf("TableDependsRhoT (live (rho,T) root for cyclic PTE):\n\tmatid = %i\n\tnumRho = "
           "%i\n\tnumT = %i\n",
           matid_, numRho_, numT_);
  }

  static std::string EosType() { return std::string("TableDependsRhoT"); }
  static std::string EosPyType() { return EosType(); }
  constexpr static inline int nlambda() noexcept { return _n_lambda; }
  template <typename T>
  static inline constexpr bool NeedsLambda() {
    return false;
  }

 private:
  inline herr_t loadTable_(const std::string &matid_str, hid_t file);

  PORTABLE_FORCEINLINE_FUNCTION Real clampT_(const Real t) const {
    return std::min(std::max(t, Tmin_), Tmax_);
  }

  // Cell index k with nodes[k] <= value <= nodes[k+1], clamped to [0, n-2] (like the sibling).
  PORTABLE_INLINE_FUNCTION int cell_(const DataBox &nodes, int n, Real value) const {
    int lo = 0, hi = n - 1;
    while (hi - lo > 1) {
      const int mid = (lo + hi) / 2;
      if (nodes(mid) <= value)
        lo = mid;
      else
        hi = mid;
    }
    return std::min(std::max(lo, 0), n - 2);
  }

  // Linear-in-T weight and column index for temperature t (already clamped).
  PORTABLE_FORCEINLINE_FUNCTION void tWeight_(const Real t, int &j, Real &w) const {
    j = cell_(T_, numT_, t);
    const Real t_j = T_(j), t_jp = T_(j + 1);
    w = (t - t_j) / (t_jp - t_j);
  }

  // Interpolate (P, dP/drho, e, de/drho) along rho within cell i at fixed T-column j:
  // P is LINEAR in rho (so it matches the density root's P interpolation), e is interpolated
  // THROUGH tau=1/rho (Clayton arXiv:2606.27726 Remark 3.2 -- interpolating e independently of
  // tau makes (rho)->(P,e) inconsistent), and the returned partials are the EXACT analytic
  // derivatives of THAT interpolant. Mutually consistent value+partials are what the cyclic
  // Newton needs; separately-tabulated partials (the old bilin_ path) drift off the value
  // interpolant and stall the solve near a small residual. (Port of TableDependsPT::column_.)
  PORTABLE_INLINE_FUNCTION void rhoColumn_(int i, int j, Real rho, Real &P, Real &dP_drho,
                                           Real &e, Real &de_drho) const {
    const Real r_a = rhoGrid_(i), r_b = rhoGrid_(i + 1);
    const Real P_a = P_(i, j), P_b = P_(i + 1, j);
    const Real e_a = sie_(i, j), e_b = sie_(i + 1, j);
    dP_drho = (P_b - P_a) / (r_b - r_a);
    P = P_a + dP_drho * (rho - r_a);
    const Real tau = 1.0 / rho, tau_a = 1.0 / r_a, tau_b = 1.0 / r_b;
    if (std::abs(tau_b - tau_a) <= INCOMPRESSIBLE_REL_TOL_ * std::abs(tau_a)) {
      de_drho = (e_b - e_a) / (r_b - r_a);
      e = e_a + de_drho * (rho - r_a);
    } else {
      const Real de_dtau = (e_b - e_a) / (tau_b - tau_a);
      e = e_a + de_dtau * (tau - tau_a);
      de_drho = de_dtau * (-1.0 / (rho * rho)); // de/drho = de/dtau * dtau/drho
    }
  }

  // Full (rho,T) evaluation: rho-column at the two bracketing T-nodes blended linearly in T for
  // values + d/drho, with d/dT the T-secant of the rho-interpolated value -- i.e. the analytic
  // gradient of the (P linear-in-rho, e through-tau) x (linear-in-T) interpolant. All six
  // outputs come from ONE interpolant, so they are mutually consistent.
  PORTABLE_INLINE_FUNCTION void evalRhoT_(Real rho, Real t, Real &P, Real &e, Real &dP_drho,
                                          Real &dP_dT, Real &de_drho, Real &de_dT) const {
    const int i = cell_(rhoGrid_, numRho_, std::min(std::max(rho, rhoMin_), rhoMax_));
    int j;
    Real w;
    tWeight_(clampT_(t), j, w);
    Real P_j, dPr_j, e_j, der_j, P_jp, dPr_jp, e_jp, der_jp;
    rhoColumn_(i, j, rho, P_j, dPr_j, e_j, der_j);
    rhoColumn_(i, j + 1, rho, P_jp, dPr_jp, e_jp, der_jp);
    P = (1.0 - w) * P_j + w * P_jp;
    e = (1.0 - w) * e_j + w * e_jp;
    dP_drho = (1.0 - w) * dPr_j + w * dPr_jp;
    de_drho = (1.0 - w) * der_j + w * der_jp;
    const Real dT = T_(j + 1) - T_(j);
    dP_dT = (P_jp - P_j) / dT;
    de_dT = (e_jp - e_j) / dT;
  }

  // P(rho_node_i, T) with T linearly blended between columns (used by MaximumPressure).
  PORTABLE_FORCEINLINE_FUNCTION Real pressureAtRhoIndexT_(int i, Real t) const {
    int j;
    Real w;
    tWeight_(clampT_(t), j, w);
    return (1.0 - w) * P_(i, j) + w * P_(i, j + 1);
  }

  // Dense-branch root: densest rho with P(rho;T) = press on an INCREASING crossing (P_rho>0).
  // Direct port of RhoTPteEos._density_at (pfc/sim/matdata/_pte_rhot.py).
  PORTABLE_INLINE_FUNCTION Real densityOfPT_(Real press, Real temp) const;

  static constexpr Real RHO_ROOT_TOL_ = 1.0e-12; // matches _pte_rhot.py::_RHO_XTOL
  static constexpr Real INCOMPRESSIBLE_REL_TOL_ = 1.0e-12; // tau-a~tau-b guard (e-through-tau)
  static constexpr unsigned long _preferred_input =
      thermalqs::pressure | thermalqs::temperature;
  static constexpr int _n_lambda = 2;

  // 1-D node arrays + 2-D fields [numRho_(slow), numT_(fast)].
  DataBox rhoGrid_, T_;
  DataBox P_, sie_, dPdRho_, dPdT_, dEdRho_, dEdT_;
#define DBLIST &rhoGrid_, &T_, &P_, &sie_, &dPdRho_, &dPdT_, &dEdRho_, &dEdT_
  std::vector<const DataBox *> GetDataBoxPointers_() const {
    return std::vector<const DataBox *>{DBLIST};
  }
  std::vector<DataBox *> GetDataBoxPointers_() { return std::vector<DataBox *>{DBLIST}; }
#undef DBLIST

  int numRho_ = 0, numT_ = 0;
  Real Tmin_, Tmax_, rhoMin_, rhoMax_;
  Real normalDensity_ = 0.0;
  MeanAtomicProperties AZbar_;
  int matid_ = -1;
  DataStatus memoryStatus_ = DataStatus::Deallocated;
  char *sharedMemory_ = nullptr;
};

// ============================ constructors ==================================
//
// Loads the SAME per-phase (rho,T) surface the single-material HEM path uses, but in the
// rectangular "dependsRhoT" layout this model reads: 1-D ascending node arrays `density`,
// `temperature` and 2-D fields [numRho(slow), numT(fast)] `pressure`, `specific internal
// energy`, `dPdRho`, `dPdT`, `dEdRho`, `dEdT`.  A matdata exporter
// (pfc.sim.matdata: `write_rhot_depends_sp5`, the (rho,T) analogue of `write_inverted_pt_sp5`)
// writes these from the SAME ConsistentEos the (P,T) table is built from, so the two agree
// on the surface and only differ in index direction.  Units table-native (GPa, g/cm^3, MJ/kg).

inline TableDependsRhoT::TableDependsRhoT(const std::string &filename, int matid)
    : matid_(matid), memoryStatus_(DataStatus::OnHost) {
  const std::string matid_str = std::to_string(matid);
  H5Eset_auto(H5E_DEFAULT, spiner_common::aborting_error_handler, NULL);
  hid_t file =
      spiner_common::h5_safe_fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  loadTable_(matid_str, file);
  spiner_common::h5_safe_fclose(file);
  CheckParams();
}

inline TableDependsRhoT::TableDependsRhoT(const std::string &filename,
                                          const std::string &materialName)
    : memoryStatus_(DataStatus::OnHost) {
  H5Eset_auto(H5E_DEFAULT, spiner_common::aborting_error_handler, NULL);
  hid_t file =
      spiner_common::h5_safe_fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  hid_t matGroup =
      spiner_common::h5_safe_gopen(file, materialName.c_str(), H5P_DEFAULT);
  spiner_common::h5_safe_get_attribute<int>(matGroup, ".", "matid", &matid_, true);
  spiner_common::h5_safe_gclose(matGroup);
  loadTable_(std::to_string(matid_), file);
  spiner_common::h5_safe_fclose(file);
  CheckParams();
}

inline herr_t TableDependsRhoT::loadTable_(const std::string &matid_str, hid_t file) {
  hid_t matGroup = spiner_common::h5_safe_gopen(file, matid_str.c_str(), H5P_DEFAULT);
  hid_t grp = spiner_common::h5_safe_gopen(matGroup, "dependsRhoT", H5P_DEFAULT);

  spiner_common::h5_safe_get_attribute<int>(grp, ".", "nRho", &numRho_, true);
  spiner_common::h5_safe_get_attribute<int>(grp, ".", "nT", &numT_, true);
  spiner_common::h5_safe_get_attribute<double>(matGroup, ".", SP5::Material::normalDensity,
                                               &normalDensity_, false);
  spiner_common::h5_safe_get_attribute<double>(matGroup, ".", SP5::Material::meanAtomicMass,
                                               &(AZbar_.Abar), false);
  spiner_common::h5_safe_get_attribute<double>(matGroup, ".", SP5::Material::meanAtomicNumber,
                                               &(AZbar_.Zbar), false);

  rhoGrid_.resize(numRho_);
  T_.resize(numT_);
  P_.resize(numRho_, numT_);
  sie_.resize(numRho_, numT_);
  dPdRho_.resize(numRho_, numT_);
  dPdT_.resize(numRho_, numT_);
  dEdRho_.resize(numRho_, numT_);
  dEdT_.resize(numRho_, numT_);

  H5LTread_dataset_double(grp, "density", rhoGrid_.data());
  H5LTread_dataset_double(grp, "temperature", T_.data());
  H5LTread_dataset_double(grp, "pressure", P_.data());
  H5LTread_dataset_double(grp, "specific internal energy", sie_.data());
  H5LTread_dataset_double(grp, "dPdRho", dPdRho_.data());
  H5LTread_dataset_double(grp, "dPdT", dPdT_.data());
  H5LTread_dataset_double(grp, "dEdRho", dEdRho_.data());
  H5LTread_dataset_double(grp, "dEdT", dEdT_.data());

  rhoMin_ = rhoGrid_(0);
  rhoMax_ = rhoGrid_(numRho_ - 1);
  Tmin_ = T_(0);
  Tmax_ = T_(numT_ - 1);

  spiner_common::h5_safe_gclose(grp);
  spiner_common::h5_safe_gclose(matGroup);
  return 0;
}

// ============================ evaluation ====================================

// Dense-branch density root: scan the density grid from the dense end and take the densest
// interval where P(rho;T) crosses `press` while INCREASING (P_rho>0) -- the condensed branch.
// Off-branch queries clamp to the nearer grid end.  (Port of RhoTPteEos._density_at.)
PORTABLE_INLINE_FUNCTION Real TableDependsRhoT::densityOfPT_(Real press, Real temp) const {
  const Real t = clampT_(temp);
  int j;
  Real w;
  tWeight_(t, j, w);
  auto Pcol = [&](int i) -> Real { return (1.0 - w) * P_(i, j) + w * P_(i, j + 1); };
  // Scan from the dense end (high i) toward low density for the densest increasing crossing.
  for (int k = numRho_ - 2; k >= 0; --k) {
    const Real d0 = Pcol(k) - press;
    const Real d1 = Pcol(k + 1) - press;
    if (d1 == 0.0) return rhoGrid_(k + 1);
    if (d0 == 0.0) return rhoGrid_(k);
    if (d0 < 0.0 && 0.0 < d1) { // increasing crossing => condensed branch
      // Bisection for the root within [rhoGrid_(k), rhoGrid_(k+1)] (linear-in-rho within cell,
      // so this converges immediately; kept general for robustness).
      Real rlo = rhoGrid_(k), rhi = rhoGrid_(k + 1);
      for (int it = 0; it < 100; ++it) {
        const Real rm = 0.5 * (rlo + rhi);
        const Real wr = (rm - rhoGrid_(k)) / (rhoGrid_(k + 1) - rhoGrid_(k));
        const Real pm = (1.0 - wr) * Pcol(k) + wr * Pcol(k + 1) - press; // linear-in-rho
        if (pm < 0.0)
          rlo = rm;
        else
          rhi = rm;
        if ((rhi - rlo) <= RHO_ROOT_TOL_ * (std::abs(rm) + RHO_ROOT_TOL_)) break;
      }
      return 0.5 * (rlo + rhi);
    }
  }
  // No interior condensed crossing: clamp to the nearer grid end.
  return (press >= Pcol(numRho_ - 1)) ? rhoGrid_(numRho_ - 1) : rhoGrid_(0);
}

template <typename Indexer_t>
PORTABLE_INLINE_FUNCTION void TableDependsRhoT::DensityEnergyFromPressureTemperature(
    const Real press, const Real temp, Indexer_t &&, Real &rho, Real &sie) const {
  const Real t = clampT_(temp);
  rho = densityOfPT_(press, t);
  sie = InternalEnergyFromDensityTemperature(rho, t, static_cast<Real *>(nullptr));
}

template <typename Indexer_t>
PORTABLE_INLINE_FUNCTION void
TableDependsRhoT::DensityEnergyDerivativesFromPressureTemperature(
    const Real press, const Real temp, Indexer_t &&, Real &rho, Real &sie, Real &drho_dP,
    Real &drho_dT, Real &de_dP, Real &de_dT) const {
  const Real t = clampT_(temp);
  rho = densityOfPT_(press, t);
  // Value + (rho,T) partials from ONE interpolant (evalRhoT_) -- mutually consistent, so the
  // chain-ruled (P,T) partials match the residual the cyclic solver drives (the old separately-
  // tabulated bilin_ partials drifted off the value interpolant and stalled the solve).
  Real P, p_rho, p_t, e_rho, e_t;
  evalRhoT_(rho, t, P, sie, p_rho, p_t, e_rho, e_t);
  // Chain-rule to (P,T) partials (RhoTPteEos.evaluate).  drho/dX = -rho^2 * dtau/dX.
  const Real inv_prho = robust::ratio(1.0, p_rho);
  const Real dtau_dP = -inv_prho / (rho * rho);
  const Real dtau_dT = (p_t * inv_prho) / (rho * rho);
  drho_dP = -rho * rho * dtau_dP; // = 1 / p_rho
  drho_dT = -rho * rho * dtau_dT; // = -p_t / p_rho
  de_dP = e_rho * inv_prho;
  de_dT = e_t - e_rho * p_t * inv_prho;
}

template <typename Indexer_t>
PORTABLE_INLINE_FUNCTION Real TableDependsRhoT::PressureFromDensityTemperature(
    const Real rho, const Real temperature, Indexer_t &&) const {
  const Real r = std::min(std::max(rho, rhoMin_), rhoMax_);
  Real P, e, p_rho, p_t, e_rho, e_t;
  evalRhoT_(r, clampT_(temperature), P, e, p_rho, p_t, e_rho, e_t);
  return P;
}

template <typename Indexer_t>
PORTABLE_INLINE_FUNCTION Real TableDependsRhoT::InternalEnergyFromDensityTemperature(
    const Real rho, const Real temperature, Indexer_t &&) const {
  const Real r = std::min(std::max(rho, rhoMin_), rhoMax_);
  Real P, e, p_rho, p_t, e_rho, e_t;
  evalRhoT_(r, clampT_(temperature), P, e, p_rho, p_t, e_rho, e_t);
  return e;
}

template <typename Indexer_t>
PORTABLE_INLINE_FUNCTION Real TableDependsRhoT::SpecificHeatFromDensityTemperature(
    const Real rho, const Real temperature, Indexer_t &&) const {
  const Real r = std::min(std::max(rho, rhoMin_), rhoMax_);
  Real P, e, p_rho, p_t, e_rho, e_t;
  evalRhoT_(r, clampT_(temperature), P, e, p_rho, p_t, e_rho, e_t);
  return e_t; // (de/dT)_rho = c_v
}

template <typename Indexer_t>
PORTABLE_INLINE_FUNCTION Real TableDependsRhoT::BulkModulusFromDensityTemperature(
    const Real rho, const Real temperature, Indexer_t &&) const {
  const Real r = std::min(std::max(rho, rhoMin_), rhoMax_);
  Real P, e, p_rho, p_t, e_rho, e_t;
  evalRhoT_(r, clampT_(temperature), P, e, p_rho, p_t, e_rho, e_t);
  return r * p_rho; // K_T = rho (dP/drho)_T
}

// Invert e(rho,T) for T at fixed rho by bisection (rho fixed, e monotone increasing in T).
template <typename Indexer_t>
PORTABLE_INLINE_FUNCTION Real TableDependsRhoT::TemperatureFromDensityInternalEnergy(
    const Real rho, const Real sie, Indexer_t &&lambda) const {
  Real tlo = Tmin_, thi = Tmax_;
  for (int it = 0; it < 100; ++it) {
    const Real tm = 0.5 * (tlo + thi);
    const Real em = InternalEnergyFromDensityTemperature(rho, tm, lambda);
    if (em < sie)
      tlo = tm;
    else
      thi = tm;
    if ((thi - tlo) <= 1.0e-10 * (tm + 1.0e-10)) break;
  }
  return 0.5 * (tlo + thi);
}

template <typename Indexer_t>
PORTABLE_INLINE_FUNCTION void
TableDependsRhoT::FillEos(Real &rho, Real &temp, Real &energy, Real &press, Real &cv,
                          Real &bmod, const unsigned long output, Indexer_t &&lambda) const {
  if (output & thermalqs::density && output & thermalqs::specific_internal_energy) {
    DensityEnergyFromPressureTemperature(press, temp, lambda, rho, energy);
  } else {
    if (output & thermalqs::pressure) press = PressureFromDensityTemperature(rho, temp, lambda);
    if (output & thermalqs::specific_internal_energy)
      energy = InternalEnergyFromDensityTemperature(rho, temp, lambda);
  }
  if (output & thermalqs::specific_heat)
    cv = SpecificHeatFromDensityTemperature(rho, temp, lambda);
  if (output & thermalqs::bulk_modulus)
    bmod = BulkModulusFromDensityTemperature(rho, temp, lambda);
}

} // namespace singularity

#endif // SINGULARITY_USE_PT_TABLES
#endif // _SINGULARITY_EOS_EOS_EOS_TABLE_RHOT_HPP_
