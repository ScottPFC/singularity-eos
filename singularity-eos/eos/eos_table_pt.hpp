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

#ifndef _SINGULARITY_EOS_EOS_EOS_TABLE_PT_HPP_
#define _SINGULARITY_EOS_EOS_EOS_TABLE_PT_HPP_

// A (P,T)-tabulated, Maxwell-consistent inverted EOS for the cyclic
// pressure-temperature-equilibrium (PTE) mixture closure (PTESolverPT).
//
// Unlike SpinerEOSDependsRhoT (a (rho,T) table extracted from raw SESAME), this model
// reads an *inverted* (P,T) table built by the pfc.sim.matdata pipeline from a single
// consistent free energy F: rho(P,T), e(P,T) and their four first partials, validated
// against Menikoff-Plohr / Theorem 2.26 (Clayton-McConnell-Solomon, arXiv:2606.27726).
// The primitive PTESolverPT drives is DensityEnergyFromPressureTemperature(P,T)->(rho,e),
// which here is a *direct* O(1) table interpolation (no per-cell density root-find): the
// inversion is pre-solved, so the mixture hot loop is cheap.
//
// Interpolation (port of pfc/sim/matdata/_pte_table.py::_column, paper sec 3 / Remark 3.2):
// on the native (non-uniform) (P,T) node grid, rho is linear in P within a cell so
// tau = 1/rho is rational and monotone (invertible -> a unique P root), e is interpolated
// *through* the tau interpolant (interpolating e independently would make (P,T)->(tau,e)
// inconsistent), and both are linear in T between adjacent columns. The raw node values are
// used directly (NOT spiner interpToReal, which is bilinear-in-log and breaks the through-tau
// scheme); the stored analytic partials feed the analytic Jacobian (PTESolverPTAnalytic).
//
// Units: the table is written by _sp5.py in (GPa, Mg/m^3 == g/cm^3, MJ/kg). The model reads
// and returns those table-native units; the FLASH<->singularity interface reconciles unit
// systems at the boundary (all materials in a mixture share the same units, so the closure
// is unit-system-agnostic internally). TODO(pte): plumb an explicit unit-system conversion.

#ifdef SINGULARITY_USE_PT_TABLES
#ifndef SINGULARITY_USE_SPINER_WITH_HDF5
#error "SINGULARITY_USE_PT_TABLES requires SINGULARITY_USE_SPINER_WITH_HDF5"
#endif

#include <algorithm>
#include <cstdlib>
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

class TableDependsPT : public EosBase<TableDependsPT> {
  friend class table_utils::SpinerTricks<TableDependsPT>;
  using SpinerTricks = table_utils::SpinerTricks<TableDependsPT>;

 public:
  using DataBox = spiner_common::DataBox;

  // Lambda holds a warm-start (P,T) guess (the paper's sec 7.3 warm start).
  struct Lambda {
    enum Index { P = 0, T = 1 };
  };

  SG_ADD_DEFAULT_MEAN_ATOMIC_FUNCTIONS(AZbar_)
  SG_ADD_BASE_CLASS_USINGS(TableDependsPT);

  PORTABLE_INLINE_FUNCTION
  TableDependsPT() : memoryStatus_(DataStatus::Deallocated) {}

  inline TableDependsPT(const std::string &filename, int matid);
  inline TableDependsPT(const std::string &filename, const std::string &materialName);

  inline TableDependsPT GetOnDevice() { return SpinerTricks::GetOnDevice(this); }
  inline void Finalize() { SpinerTricks::Finalize(this); }
  std::size_t DynamicMemorySizeInBytes() const {
    return SpinerTricks::DynamicMemorySizeInBytes(this);
  }
  std::size_t DumpDynamicMemory(char *dst) {
    return SpinerTricks::DumpDynamicMemory(dst, this);
  }
  // See the twin comment in eos_table_rhot.hpp: the databoxes must be pointed at the SHARED
  // allocation when one is supplied, or every rank keeps its own copy and dangles when the
  // caller frees its packed buffer.
  std::size_t SetDynamicMemory(char *src,
                               const SharedMemSettings &stngs = DEFAULT_SHMEM_STNGS) {
    char *base = (stngs.data == nullptr) ? src : stngs.data;
    sharedMemory_ = stngs.data;
    return SpinerTricks::SetDynamicMemory(base, this);
  }

  PORTABLE_INLINE_FUNCTION void CheckParams() const {
    PORTABLE_ALWAYS_REQUIRE(numP_ > 1, "At least two pressure points");
    PORTABLE_ALWAYS_REQUIRE(numT_ > 1, "At least two temperature points");
    PORTABLE_ALWAYS_REQUIRE(Pmax_ > Pmin_, "Pressure bounds ordered");
    PORTABLE_ALWAYS_REQUIRE(Tmax_ > Tmin_, "Temperature bounds ordered");
  }

  // ---- the primitive PTESolverPT drives: (P,T) -> (rho, e), direct interpolation ----
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION void
  DensityEnergyFromPressureTemperature(const Real press, const Real temp,
                                       Indexer_t &&lambda, Real &rho, Real &sie) const;

  // (P,T) -> (rho, e) plus the four first partials of *this interpolant* (drho/dP)_T,
  // (drho/dT)_P, (de/dP)_T, (de/dT)_P. These are the exact analytic derivatives of
  // DensityEnergyFromPressureTemperature above, so PTESolverPTAnalytic's Jacobian is
  // consistent with the residual it drives (vs. the base PTESolverPT finite-difference).
  // (This is the partial set the validated pfc solver used: _pte_table.py::evaluate.)
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION void DensityEnergyDerivativesFromPressureTemperature(
      const Real press, const Real temp, Indexer_t &&lambda, Real &rho, Real &sie,
      Real &drho_dP, Real &drho_dT, Real &de_dP, Real &de_dT) const;

  // ---- (rho,T) / (rho,e) entry points (invert the monotone rho(P) at fixed T) ----
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
  PORTABLE_INLINE_FUNCTION Real PressureFromDensityInternalEnergy(
      const Real rho, const Real sie,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const;
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real SpecificHeatFromDensityInternalEnergy(
      const Real rho, const Real sie,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const;
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real TemperatureFromDensityInternalEnergy(
      const Real rho, const Real sie,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const;

  // Aux methods not needed by the PT closure; fail loudly if reached.
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real
  EntropyFromDensityTemperature(const Real, const Real,
                                Indexer_t && = static_cast<Real *>(nullptr)) const {
    PORTABLE_ALWAYS_THROW_OR_ABORT("Entropy not implemented for TableDependsPT");
    return 0.0;
  }
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real
  BulkModulusFromDensityTemperature(const Real rho, const Real temperature,
                                    Indexer_t &&lambda = static_cast<Real *>(nullptr))
      const;
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real
  GruneisenParamFromDensityTemperature(const Real, const Real,
                                       Indexer_t && = static_cast<Real *>(nullptr)) const {
    PORTABLE_ALWAYS_THROW_OR_ABORT("Gruneisen not implemented for TableDependsPT");
    return 0.0;
  }
  // (rho, sie) aux variants (each model must provide the scalar; EosBase only supplies the
  // vector overloads that delegate here). Entropy/Gruneisen fail loudly; bulk modulus goes
  // through the T inversion.
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real
  EntropyFromDensityInternalEnergy(const Real, const Real,
                                   Indexer_t && = static_cast<Real *>(nullptr)) const {
    PORTABLE_ALWAYS_THROW_OR_ABORT("Entropy not implemented for TableDependsPT");
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
    PORTABLE_ALWAYS_THROW_OR_ABORT("Gruneisen not implemented for TableDependsPT");
    return 0.0;
  }

  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION void
  FillEos(Real &rho, Real &temp, Real &energy, Real &press, Real &cv, Real &bmod,
          const unsigned long output,
          Indexer_t &&lambda = static_cast<Real *>(nullptr)) const;

  // Reference state (~room T at the material's normal density) used by the C API / FLASH
  // for normalization + initial guesses. Derived from the table at (P_ref, T_ref).
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION void
  ValuesAtReferenceState(Real &rho, Real &temp, Real &sie, Real &press, Real &cv, Real &bmod,
                         Real &dpde, Real &dvdt,
                         Indexer_t &&lambda = static_cast<Real *>(nullptr)) const {
    const Real T_ref = std::min(std::max(298.15, Tmin_), Tmax_);
    const Real rho_ref =
        (normalDensity_ > 0.0) ? normalDensity_ : std::sqrt(rhoMin_ * rhoMax_);
    const Real P_ref = pressureOfRhoT_(rho_ref, T_ref);
    Real r, e, drho_dP, drho_dT, de_dP, de_dT;
    DensityEnergyDerivativesFromPressureTemperature(P_ref, T_ref, lambda, r, e, drho_dP,
                                                    drho_dT, de_dP, de_dT);
    rho = r;
    temp = T_ref;
    sie = e;
    press = P_ref;
    cv = de_dT;                                // heat-capacity scale (>0 in single phase)
    bmod = r / robust::make_positive(drho_dP); // isothermal bulk modulus K_T
    // (dP/de)_rho ~ (dP/dT)_rho / (de/dT)_rho, with (dP/dT)_rho = -(drho/dT)/(drho/dP).
    dpde = robust::ratio(-robust::ratio(drho_dT, drho_dP), de_dT);
    dvdt = -robust::ratio(drho_dT, r * r); // (d(1/rho)/dT)_P
  }

  static constexpr unsigned long PreferredInput() { return _preferred_input; }
  int matid() const { return matid_; }

  PORTABLE_FORCEINLINE_FUNCTION Real MinimumDensity() const { return rhoMin_; }
  PORTABLE_FORCEINLINE_FUNCTION Real MaximumDensity() const { return rhoMax_; }
  PORTABLE_FORCEINLINE_FUNCTION Real MinimumTemperature() const { return Tmin_; }
  PORTABLE_FORCEINLINE_FUNCTION Real MaximumTemperature() const { return Tmax_; }
  PORTABLE_FORCEINLINE_FUNCTION Real MinimumPressure() const { return Pmin_; }
  PORTABLE_FORCEINLINE_FUNCTION Real MaximumPressureAtTemperature(const Real) const {
    return Pmax_;
  }

  PORTABLE_INLINE_FUNCTION void PrintParams() const {
    printf("TableDependsPT (inverted (P,T) table for PTE):\n\tmatid = %i\n\tnumP = "
           "%i\n\tnumT = %i\n",
           matid_, numP_, numT_);
  }

  static std::string EosType() { return std::string("TableDependsPT"); }
  static std::string EosPyType() { return EosType(); }
  constexpr static inline int nlambda() noexcept { return _n_lambda; }
  template <typename T>
  static inline constexpr bool NeedsLambda() {
    return false;
  }

 private:
  inline herr_t loadTable_(const std::string &matid_str, hid_t file);

  // Cell index k such that nodes[k] <= value <= nodes[k+1], clamped so the stencil exists.
  // (Port of _pte_table.py::_cell: bisect_right - 1, clamped to [0, n-2].)
  PORTABLE_INLINE_FUNCTION int cell_(const DataBox &nodes, int n, Real value) const {
    // Binary search for the last node <= value.
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

  // Interpolate (tau, dtau/dP, e, de/dP) along P in column j, cell i (paper eq. 3.3/3.6).
  PORTABLE_INLINE_FUNCTION void column_(int i, int j, Real p_i, Real p_ip, Real pressure,
                                        Real &tau, Real &dtau_dp, Real &energy,
                                        Real &de_dp) const {
    const Real rho_a = rho_(i, j), rho_b = rho_(i + 1, j);
    const Real e_a = sie_(i, j), e_b = sie_(i + 1, j);
    const Real d_rho_dp = (rho_b - rho_a) / (p_ip - p_i);
    const Real rho = rho_a + d_rho_dp * (pressure - p_i);
    tau = 1.0 / rho;
    dtau_dp = -d_rho_dp / (rho * rho);
    const Real tau_a = 1.0 / rho_a, tau_b = 1.0 / rho_b;
    if (std::abs(tau_b - tau_a) <= INCOMPRESSIBLE_REL_TOL_ * std::abs(tau_a)) {
      de_dp = (e_b - e_a) / (p_ip - p_i);
      energy = e_a + de_dp * (pressure - p_i);
    } else {
      const Real de_dtau = (e_b - e_a) / (tau_b - tau_a);
      energy = e_a + de_dtau * (tau - tau_a);
      de_dp = de_dtau * dtau_dp;
    }
  }

  // rho(P) at fixed T, and its dP-derivative (for the P inversion below).
  PORTABLE_INLINE_FUNCTION Real rhoOfPT_(Real press, Real temp) const {
    Real rho, sie;
    DensityEnergyFromPressureTemperature(press, temp, static_cast<Real *>(nullptr), rho,
                                         sie);
    return rho;
  }

  // Invert the monotone rho(P;T) for P at fixed T by bisection on [Pmin_, Pmax_].
  PORTABLE_INLINE_FUNCTION Real pressureOfRhoT_(Real rho_target, Real temp) const {
    Real plo = Pmin_, phi = Pmax_;
    const Real rlo = rhoOfPT_(plo, temp), rhi = rhoOfPT_(phi, temp);
    if (rho_target <= rlo) return plo;
    if (rho_target >= rhi) return phi;
    for (int it = 0; it < 100; ++it) {
      const Real pm = 0.5 * (plo + phi);
      const Real rm = rhoOfPT_(pm, temp);
      if (rm < rho_target)
        plo = pm;
      else
        phi = pm;
      if ((phi - plo) <= PRESSURE_ROOT_TOL_ * (std::abs(pm) + PRESSURE_ROOT_TOL_)) break;
    }
    return 0.5 * (plo + phi);
  }

  static constexpr Real INCOMPRESSIBLE_REL_TOL_ = 1.0e-12; // matches _pte_table.py
  static constexpr Real PRESSURE_ROOT_TOL_ = 1.0e-12;
  static constexpr unsigned long _preferred_input =
      thermalqs::pressure | thermalqs::temperature;
  static constexpr int _n_lambda = 2;

  // 2-D fields [numP_(slow), numT_(fast)] and 1-D node arrays.
  DataBox rho_, sie_, dRhodP_, dRhodT_, dEdP_, dEdT_P_;
  DataBox P_, T_;
#define DBLIST &rho_, &sie_, &dRhodP_, &dRhodT_, &dEdP_, &dEdT_P_, &P_, &T_
  std::vector<const DataBox *> GetDataBoxPointers_() const {
    return std::vector<const DataBox *>{DBLIST};
  }
  std::vector<DataBox *> GetDataBoxPointers_() { return std::vector<DataBox *>{DBLIST}; }
#undef DBLIST

  int numP_ = 0, numT_ = 0;
  Real Pmin_, Pmax_, Tmin_, Tmax_, rhoMin_, rhoMax_;
  Real normalDensity_ = 0.0;
  MeanAtomicProperties AZbar_;
  int matid_ = -1;
  DataStatus memoryStatus_ = DataStatus::Deallocated;
  char *sharedMemory_ = nullptr;
};

// ============================ constructors ==================================

inline TableDependsPT::TableDependsPT(const std::string &filename, int matid)
    : matid_(matid), memoryStatus_(DataStatus::OnHost) {
  const std::string matid_str = std::to_string(matid);
  H5Eset_auto(H5E_DEFAULT, spiner_common::aborting_error_handler, NULL);
  hid_t file =
      spiner_common::h5_safe_fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  loadTable_(matid_str, file);
  spiner_common::h5_safe_fclose(file);
  CheckParams();
}

inline TableDependsPT::TableDependsPT(const std::string &filename,
                                      const std::string &materialName)
    : memoryStatus_(DataStatus::OnHost) {
  H5Eset_auto(H5E_DEFAULT, spiner_common::aborting_error_handler, NULL);
  hid_t file =
      spiner_common::h5_safe_fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  // The named group aliases the matid group; read the matid attr back out.
  hid_t matGroup =
      spiner_common::h5_safe_gopen(file, materialName.c_str(), H5P_DEFAULT);
  spiner_common::h5_safe_get_attribute<int>(matGroup, ".", "matid", &matid_, true);
  spiner_common::h5_safe_gclose(matGroup);
  loadTable_(std::to_string(matid_), file);
  spiner_common::h5_safe_fclose(file);
  CheckParams();
}

inline herr_t TableDependsPT::loadTable_(const std::string &matid_str, hid_t file) {
  hid_t matGroup = spiner_common::h5_safe_gopen(file, matid_str.c_str(), H5P_DEFAULT);
  hid_t grp = spiner_common::h5_safe_gopen(matGroup, SP5::Depends::logPLogT, H5P_DEFAULT);

  spiner_common::h5_safe_get_attribute<int>(grp, ".", "nP", &numP_, true);
  spiner_common::h5_safe_get_attribute<int>(grp, ".", "nT", &numT_, true);
  spiner_common::h5_safe_get_attribute<double>(matGroup, ".", SP5::Material::normalDensity,
                                               &normalDensity_, false);
  // Mean atomic mass/number (SESAME 201), used by FLASH to set the species A/Z.
  spiner_common::h5_safe_get_attribute<double>(matGroup, ".", SP5::Material::meanAtomicMass,
                                               &(AZbar_.Abar), false);
  spiner_common::h5_safe_get_attribute<double>(matGroup, ".", SP5::Material::meanAtomicNumber,
                                               &(AZbar_.Zbar), false);

  P_.resize(numP_);
  T_.resize(numT_);
  rho_.resize(numP_, numT_);
  sie_.resize(numP_, numT_);
  dRhodP_.resize(numP_, numT_);
  dRhodT_.resize(numP_, numT_);
  dEdP_.resize(numP_, numT_);
  dEdT_P_.resize(numP_, numT_);

  H5LTread_dataset_double(grp, SP5::Fields::P, P_.data());
  H5LTread_dataset_double(grp, SP5::Fields::T, T_.data());
  H5LTread_dataset_double(grp, SP5::Fields::rho, rho_.data());
  H5LTread_dataset_double(grp, SP5::Fields::sie, sie_.data());
  H5LTread_dataset_double(grp, SP5::Fields::dRhodP, dRhodP_.data());
  H5LTread_dataset_double(grp, SP5::Fields::dRhodT, dRhodT_.data());
  H5LTread_dataset_double(grp, SP5::Fields::dEdP, dEdP_.data());
  H5LTread_dataset_double(grp, SP5::Fields::dEdT_P, dEdT_P_.data());

  Pmin_ = P_(0);
  Pmax_ = P_(numP_ - 1);
  Tmin_ = T_(0);
  Tmax_ = T_(numT_ - 1);
  // rho increases with P at fixed T (Thm 2.26), so the global density extent lives on the
  // low-P / high-P columns; scan the boundary columns for the min/max.
  rhoMin_ = rho_(0, 0);
  rhoMax_ = rho_(numP_ - 1, 0);
  for (int j = 0; j < numT_; ++j) {
    rhoMin_ = std::min(rhoMin_, rho_(0, j));
    rhoMax_ = std::max(rhoMax_, rho_(numP_ - 1, j));
  }

  spiner_common::h5_safe_gclose(grp);
  spiner_common::h5_safe_gclose(matGroup);
  return 0;
}

// ============================ evaluation ====================================

template <typename Indexer_t>
PORTABLE_INLINE_FUNCTION void TableDependsPT::DensityEnergyFromPressureTemperature(
    const Real press, const Real temp, Indexer_t &&, Real &rho, Real &sie) const {
  const Real p = std::min(std::max(press, Pmin_), Pmax_);
  const Real t = std::min(std::max(temp, Tmin_), Tmax_);
  const int i = cell_(P_, numP_, p);
  const int j = cell_(T_, numT_, t);
  const Real p_i = P_(i), p_ip = P_(i + 1);
  const Real t_j = T_(j), t_jp = T_(j + 1);

  Real tau_j, dtau_dp_j, e_j, de_dp_j;
  Real tau_jp, dtau_dp_jp, e_jp, de_dp_jp;
  column_(i, j, p_i, p_ip, p, tau_j, dtau_dp_j, e_j, de_dp_j);
  column_(i, j + 1, p_i, p_ip, p, tau_jp, dtau_dp_jp, e_jp, de_dp_jp);

  const Real w = (t - t_j) / (t_jp - t_j);
  const Real tau = (1.0 - w) * tau_j + w * tau_jp;
  rho = 1.0 / tau;
  sie = (1.0 - w) * e_j + w * e_jp;
}

template <typename Indexer_t>
PORTABLE_INLINE_FUNCTION void
TableDependsPT::DensityEnergyDerivativesFromPressureTemperature(
    const Real press, const Real temp, Indexer_t &&, Real &rho, Real &sie, Real &drho_dP,
    Real &drho_dT, Real &de_dP, Real &de_dT) const {
  const Real p = std::min(std::max(press, Pmin_), Pmax_);
  const Real t = std::min(std::max(temp, Tmin_), Tmax_);
  const int i = cell_(P_, numP_, p);
  const int j = cell_(T_, numT_, t);
  const Real p_i = P_(i), p_ip = P_(i + 1);
  const Real t_j = T_(j), t_jp = T_(j + 1);

  Real tau_j, dtau_dp_j, e_j, de_dp_j;
  Real tau_jp, dtau_dp_jp, e_jp, de_dp_jp;
  column_(i, j, p_i, p_ip, p, tau_j, dtau_dp_j, e_j, de_dp_j);
  column_(i, j + 1, p_i, p_ip, p, tau_jp, dtau_dp_jp, e_jp, de_dp_jp);

  const Real dt = t_jp - t_j;
  const Real w = (t - t_j) / dt;
  const Real tau = (1.0 - w) * tau_j + w * tau_jp;
  rho = 1.0 / tau;
  sie = (1.0 - w) * e_j + w * e_jp;
  // Blend the P-line partials in T; T-partials are the secant slopes (linear in T).
  const Real dtau_dP = (1.0 - w) * dtau_dp_j + w * dtau_dp_jp;
  const Real dtau_dT = (tau_jp - tau_j) / dt;
  drho_dP = -dtau_dP / (tau * tau); // rho = 1/tau => drho = -dtau/tau^2
  drho_dT = -dtau_dT / (tau * tau);
  de_dP = (1.0 - w) * de_dp_j + w * de_dp_jp;
  de_dT = (e_jp - e_j) / dt;
}

template <typename Indexer_t>
PORTABLE_INLINE_FUNCTION Real TableDependsPT::PressureFromDensityTemperature(
    const Real rho, const Real temperature, Indexer_t &&) const {
  return pressureOfRhoT_(rho, std::min(std::max(temperature, Tmin_), Tmax_));
}

template <typename Indexer_t>
PORTABLE_INLINE_FUNCTION Real TableDependsPT::InternalEnergyFromDensityTemperature(
    const Real rho, const Real temperature, Indexer_t &&lambda) const {
  const Real t = std::min(std::max(temperature, Tmin_), Tmax_);
  const Real p = pressureOfRhoT_(rho, t);
  Real r, sie;
  DensityEnergyFromPressureTemperature(p, t, lambda, r, sie);
  return sie;
}

template <typename Indexer_t>
PORTABLE_INLINE_FUNCTION Real TableDependsPT::SpecificHeatFromDensityTemperature(
    const Real rho, const Real temperature, Indexer_t &&) const {
  // Cv ~ (de/dT)_P near the (P,T) node (the table's stored analytic (de/dT)_P). This is
  // c_p, not c_v, strictly; the PT closure uses it only as a warm-start heat scale.
  const Real t = std::min(std::max(temperature, Tmin_), Tmax_);
  const Real p = pressureOfRhoT_(rho, t);
  const int i = cell_(P_, numP_, p);
  const int j = cell_(T_, numT_, t);
  return dEdT_P_(i, j);
}

template <typename Indexer_t>
PORTABLE_INLINE_FUNCTION Real TableDependsPT::TemperatureFromDensityInternalEnergy(
    const Real rho, const Real sie, Indexer_t &&) const {
  // Invert e(rho, T) for T by bisection: at fixed T, P is set by rho, then e follows.
  Real tlo = Tmin_, thi = Tmax_;
  auto e_of_t = [&](Real tt) {
    const Real pp = pressureOfRhoT_(rho, tt);
    Real r, ee;
    DensityEnergyFromPressureTemperature(pp, tt, static_cast<Real *>(nullptr), r, ee);
    return ee;
  };
  const Real elo = e_of_t(tlo), ehi = e_of_t(thi);
  if (sie <= elo) return tlo;
  if (sie >= ehi) return thi;
  for (int it = 0; it < 100; ++it) {
    const Real tm = 0.5 * (tlo + thi);
    if (e_of_t(tm) < sie)
      tlo = tm;
    else
      thi = tm;
    if ((thi - tlo) <= PRESSURE_ROOT_TOL_ * (std::abs(tm) + PRESSURE_ROOT_TOL_)) break;
  }
  return 0.5 * (tlo + thi);
}

template <typename Indexer_t>
PORTABLE_INLINE_FUNCTION Real TableDependsPT::PressureFromDensityInternalEnergy(
    const Real rho, const Real sie, Indexer_t &&lambda) const {
  const Real t = TemperatureFromDensityInternalEnergy(rho, sie, lambda);
  return pressureOfRhoT_(rho, t);
}

template <typename Indexer_t>
PORTABLE_INLINE_FUNCTION Real TableDependsPT::SpecificHeatFromDensityInternalEnergy(
    const Real rho, const Real sie, Indexer_t &&lambda) const {
  const Real t = TemperatureFromDensityInternalEnergy(rho, sie, lambda);
  return SpecificHeatFromDensityTemperature(rho, t, lambda);
}

template <typename Indexer_t>
PORTABLE_INLINE_FUNCTION Real TableDependsPT::BulkModulusFromDensityTemperature(
    const Real rho, const Real temperature, Indexer_t &&) const {
  // K_T = rho (dP/drho)_T = rho / (drho/dP)_T, from the stored analytic partial.
  const Real t = std::min(std::max(temperature, Tmin_), Tmax_);
  const Real p = pressureOfRhoT_(rho, t);
  const int i = cell_(P_, numP_, p);
  const int j = cell_(T_, numT_, t);
  return rho / robust::make_positive(dRhodP_(i, j));
}

template <typename Indexer_t>
PORTABLE_INLINE_FUNCTION void
TableDependsPT::FillEos(Real &rho, Real &temp, Real &energy, Real &press, Real &cv,
                        Real &bmod, const unsigned long output, Indexer_t &&lambda) const {
  // Minimal FillEos: (P,T) is the natural input; produce whatever outputs are requested.
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
#endif // _SINGULARITY_EOS_EOS_EOS_TABLE_PT_HPP_
