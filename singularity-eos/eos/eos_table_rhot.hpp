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
#include <new>
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

// Diagnostic call counters -- used by the offline micro-bench AND, via a bind(C) getter, by
// FLASH's eos_pteDiag to report real per-cell op counts.
//
// OFF BY DEFAULT.  These are plain non-atomic global stores on the hottest path in the file
// (g_evalRhoT increments on EVERY interpolation), which keeps evalRhoT_ from being a pure
// function and costs a store per call for a number nobody reads in production.  Build with
// -DSINGULARITY_RHOT_COUNTERS to get them back; the getter reports zeros when they are off,
// which reads as "not measured" rather than "measured zero" because SG_RHOT_COUNTERS_ENABLED
// is exposed alongside them.
#ifdef SINGULARITY_RHOT_COUNTERS
#define SG_RHOT_COUNT(which) (++::singularity::sg_rhot_acct::which)
#define SG_RHOT_COUNTERS_ENABLED 1
#else
#define SG_RHOT_COUNT(which) ((void)0)
#define SG_RHOT_COUNTERS_ENABLED 0
#endif

// Node-spacing classification of a loaded axis.  evalRhoT_ indexes both axes on every call, and
// it can only do that in O(1) arithmetic when the nodes are evenly spaced -- in the value or in
// its log.  Anything else needs a search.  The envelope tables shipped today are written on the
// native SESAME nodes, which are evenly spaced in NEITHER, so this reports NONUNIFORM and the
// search stands; regenerating the envelope on a uniform-log grid would unlock the O(1) path.
//
// Recorded per table at load and drained once by FLASH (eos_initSingularity) so the verdict is
// printed on rank 0 next to the other EOS init lines, rather than by every rank from inside C++.
namespace sg_rhot_grid {
enum AxisKind { NONUNIFORM = 0, UNIFORM_LINEAR = 1, UNIFORM_LOG = 2 };
struct Record {
  int matid;
  int rho_kind;
  int t_kind;
  int num_rho;
  int num_t;
};
inline std::vector<Record> g_records;

// Even spacing to a relative tolerance, judged against the mean step so a single ragged node is
// not lost in an absolute threshold.  Requires all-positive values before taking logs.
inline AxisKind classify(const double *v, int n, double rtol = 1.0e-6) {
  if (n < 3) return UNIFORM_LINEAR; // two nodes are trivially evenly spaced
  auto even = [&](bool in_log) {
    if (in_log) {
      for (int i = 0; i < n; ++i)
        if (!(v[i] > 0.0)) return false;
    }
    const double first = in_log ? (std::log(v[1]) - std::log(v[0])) : (v[1] - v[0]);
    if (!(std::abs(first) > 0.0)) return false;
    for (int i = 1; i < n - 1; ++i) {
      const double d = in_log ? (std::log(v[i + 1]) - std::log(v[i])) : (v[i + 1] - v[i]);
      if (std::abs(d - first) > rtol * std::abs(first)) return false;
    }
    return true;
  };
  if (even(false)) return UNIFORM_LINEAR;
  if (even(true)) return UNIFORM_LOG;
  return NONUNIFORM;
}
} // namespace sg_rhot_grid

namespace sg_rhot_acct {
inline long g_evalRhoT = 0;      // fundamental (rho,T) interpolation
inline long g_densityOfPT = 0;   // (P,T)->rho root (analytic or scan)
inline long g_rootBisect = 0;    // flat-cell bisection fallbacks inside rootInCell
inline long g_TfromE = 0;        // (rho,e)->T inversions (grid-search; fallback bisection)
inline void reset() { g_evalRhoT = g_densityOfPT = g_rootBisect = g_TfromE = 0; }
inline constexpr bool enabled() { return SG_RHOT_COUNTERS_ENABLED != 0; }
}

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
  // Point the databoxes at the SHARED allocation whenever one is supplied, exactly as
  // SpinerEOSDependsRhoT::SetDynamicMemory does. Recording `stngs.data` and then handing
  // SpinerTricks `src` anyway -- the previous behaviour -- left every rank's databoxes aimed at
  // its OWN packed buffer, so nothing was shared and the pointers dangled the moment the caller
  // freed that buffer, which the documented MPI recipe does immediately after DeSerialize.
  // `sharedMemory_` was written and never read, so the bug was silent.
  std::size_t SetDynamicMemory(char *src,
                               const SharedMemSettings &stngs = DEFAULT_SHMEM_STNGS) {
    char *base = (stngs.data == nullptr) ? src : stngs.data;
    sharedMemory_ = stngs.data;
    const std::size_t n = SpinerTricks::SetDynamicMemory(base, this);
    // EosBase::DeSerialize reaches here having done `memcpy(this, src, sizeof(CRTP))` -- a RAW
    // byte copy of the whole object, std::vector members included. So these three headers still
    // describe the SERIALIZING rank's heap: a pointer this rank never allocated, with size() and
    // capacity() intact. Re-initialize them IN PLACE with placement-new; do NOT assign or
    // resize, because either would first free a pointer we do not own.
    //
    // (memcpy-ing a non-trivially-copyable member is UB in the first place, but the wire format
    // is upstream's and this is the containable fix.)
    new (&mMono_) std::vector<int>();
    new (&satT_) std::vector<double>();
    new (&satPsat_) std::vector<double>();
    // mMono_ and the axis kinds are DERIVED from the databoxes, which are valid again, so they
    // are rebuilt exactly. satT_/satPsat_ are FILE data and cannot be -- they stay empty, which
    // degrades gracefully to the dense-only root (the documented `empty()` behaviour). No table
    // shipped today carries a /saturation group (checked across the whole V12_rhoT set), so
    // nothing is lost in practice; a table that did would need them added to the wire format.
    rebuildDerived_();
    return n;
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
  // Gamma = (1/rho) (dP/de)_rho = (1/rho) (dP/dT)_rho / (de/dT)_rho -- exact and LOCAL in the
  // fields this table already stores, so no new data and no integration.  Implemented because
  // the lever closure reaches it once it pins a material to a tie-line: aborting here took down
  // RUN160/RUN161 at the step the liner first went two-phase.
  //
  // On a flat tie-line (de/dT)_rho is the two-phase c_v, which is large but finite, so the ratio
  // is well behaved there.  Where c_v underflows to zero the parameter is genuinely undefined;
  // return 0 (Gamma -> 0 means pressure independent of energy) rather than a signed infinity that
  // would poison a caller's linear algebra.
  template <typename Indexer_t = Real *>
  PORTABLE_INLINE_FUNCTION Real GruneisenParamFromDensityTemperature(
      const Real rho, const Real temperature,
      Indexer_t && = static_cast<Real *>(nullptr)) const {
    const Real r = std::min(std::max(rho, rhoMin_), rhoMax_);
    Real P, e, p_rho, p_t, e_rho, e_t;
    evalRhoT_(r, clampT_(temperature), P, e, p_rho, p_t, e_rho, e_t);
    return (e_t > 0.0) ? p_t / (r * e_t) : 0.0;
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
  PORTABLE_INLINE_FUNCTION Real GruneisenParamFromDensityInternalEnergy(
      const Real rho, const Real sie,
      Indexer_t &&lambda = static_cast<Real *>(nullptr)) const {
    return GruneisenParamFromDensityTemperature(
        rho, TemperatureFromDensityInternalEnergy(rho, sie, lambda), lambda);
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
  //
  // `kind`/`inv_step`/`log_min` come from the load-time axis classification (sg_rhot_grid): on an
  // evenly spaced axis the index is arithmetic, otherwise this falls back to the bisection that
  // was always here.  The arithmetic result is CORRECTED by at most a step in either direction so
  // that round-off at a node boundary can never return a cell that does not bracket the value --
  // the bracket, not the formula, is what rhoColumn_ relies on.
  PORTABLE_INLINE_FUNCTION int cell_(const DataBox &nodes, int n, Real value, int kind = 0,
                                     Real inv_step = 0.0, Real log_min = 0.0) const {
    int lo;
    if (kind == sg_rhot_grid::UNIFORM_LINEAR) {
      lo = static_cast<int>(std::floor((value - nodes(0)) * inv_step));
    } else if (kind == sg_rhot_grid::UNIFORM_LOG && value > 0.0) {
      lo = static_cast<int>(std::floor((std::log(value) - log_min) * inv_step));
    } else {
      int a = 0, b = n - 1;
      while (b - a > 1) {
        const int mid = (a + b) / 2;
        if (nodes(mid) <= value)
          a = mid;
        else
          b = mid;
      }
      return std::min(std::max(a, 0), n - 2);
    }
    lo = std::min(std::max(lo, 0), n - 2);
    while (lo > 0 && nodes(lo) > value) --lo;
    while (lo < n - 2 && nodes(lo + 1) < value) ++lo;
    return lo;
  }

  // Self-check for the O(1) index: compare it against the bisection at every node and midpoint,
  // and demote `kind` to NONUNIFORM (so cell_ searches) if they ever disagree.  Called at load.
  inline void rebuildDerived_();

  inline void demoteAxisIfIndexDisagrees_(const DataBox &nodes, int n, int &kind, Real inv_step,
                                          Real log_min) const {
    if (kind == sg_rhot_grid::NONUNIFORM || n < 3) return;
    for (int k = 0; k < n; ++k) {
      Real probes[2] = {nodes(k), (k + 1 < n) ? 0.5 * (nodes(k) + nodes(k + 1)) : nodes(k)};
      const int nprobe = (k + 1 < n) ? 2 : 1;
      for (int q = 0; q < nprobe; ++q) {
        const Real v = probes[q];
        if (cell_(nodes, n, v, kind, inv_step, log_min) !=
            cell_(nodes, n, v, sg_rhot_grid::NONUNIFORM)) {
          kind = sg_rhot_grid::NONUNIFORM;
          return;
        }
      }
    }
  }

  // Linear-in-T weight and column index for temperature t (already clamped).
  PORTABLE_FORCEINLINE_FUNCTION void tWeight_(const Real t, int &j, Real &w) const {
    j = cell_(T_, numT_, t, tAxisKind_, tInvStep_, logTMin_);
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
    SG_RHOT_COUNT(g_evalRhoT);
    const int i = cell_(rhoGrid_, numRho_, std::min(std::max(rho, rhoMin_), rhoMax_),
                        rhoAxisKind_, rhoInvStep_, logRhoMin_);
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

  // Saturation pressure P_sat(T) [cgs] on the shipped L-V dome by linear interp; returns <0 if
  // there is no dome or T is outside the dome range (=> no vapor branch, dense root used).
  PORTABLE_INLINE_FUNCTION Real pSatAt_(Real t) const {
    const int n = static_cast<int>(satT_.size());
    if (n < 2 || t < satT_[0] || t > satT_[n - 1]) return -1.0;
    int lo = 0, hi = n - 1;
    while (hi - lo > 1) {
      const int mid = (lo + hi) / 2;
      if (satT_[mid] <= t)
        lo = mid;
      else
        hi = mid;
    }
    const Real w = (t - satT_[lo]) / (satT_[hi] - satT_[lo]);
    return (1.0 - w) * satPsat_[lo] + w * satPsat_[hi];
  }

  // Dense-branch root: densest rho with P(rho;T) = press on an INCREASING crossing (P_rho>0).
  // Direct port of RhoTPteEos._density_at (pfc/sim/matdata/_pte_rhot.py).  Binary-searches the
  // column's top monotone run when press lands in it (the compressed-branch common case), else
  // linear-scans from the dense end; both return the analytic in-cell root (P linear in rho).
  // Branch-aware: below P_sat(T) inside the L-V dome it first takes the VAPOR (least-dense)
  // crossing (port of BranchAwareRhoTPteEos), gated on the shipped dome.
  PORTABLE_INLINE_FUNCTION Real densityOfPT_(Real press, Real temp) const;

  static constexpr Real RHO_ROOT_TOL_ = 1.0e-12; // matches _pte_rhot.py::_RHO_XTOL
  static constexpr Real INCOMPRESSIBLE_REL_TOL_ = 1.0e-12; // tau-a~tau-b guard (e-through-tau)
  static constexpr unsigned long _preferred_input =
      thermalqs::pressure | thermalqs::temperature;
  static constexpr int _n_lambda = 2;

  // 1-D node arrays + 2-D fields [numRho_(slow), numT_(fast)].
  DataBox rhoGrid_, T_;
  DataBox P_, sie_, dPdRho_, dPdT_, dEdRho_, dEdT_;
  // Per-T-column start index of the top monotone-increasing run of P(:,j) (host-computed at
  // load; empty => warm-start disabled, always full scan).  Guards the localized density scan
  // so it can never skip a denser crossing.  Not serialized (derived from P_; host-only).
  std::vector<int> mMono_;
  // Optional L-V saturation dome (root /saturation/vapor_liquid), for branch-aware (P,T)->rho:
  // below P_sat(T) inside the dome the physical single-phase root is the VAPOR (least-dense)
  // branch, not the dense one. cgs (satPsat_ converted GPa->dyne/cm^2 at load). Empty => the
  // surface has no shipped dome => dense-only root (backward-compatible). Host-only (like mMono_).
  std::vector<double> satT_, satPsat_;
#define DBLIST &rhoGrid_, &T_, &P_, &sie_, &dPdRho_, &dPdT_, &dEdRho_, &dEdT_
  std::vector<const DataBox *> GetDataBoxPointers_() const {
    return std::vector<const DataBox *>{DBLIST};
  }
  std::vector<DataBox *> GetDataBoxPointers_() { return std::vector<DataBox *>{DBLIST}; }
#undef DBLIST

  int numRho_ = 0, numT_ = 0;
  // Load-time axis classification (sg_rhot_grid::AxisKind) + the constants the O(1) index needs.
  // Derived from the node arrays, so like mMono_ they are host-only and not serialized.
  int rhoAxisKind_ = 0, tAxisKind_ = 0;
  Real rhoInvStep_ = 0.0, logRhoMin_ = 0.0;
  Real tInvStep_ = 0.0, logTMin_ = 0.0;
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
// (`pfc.sim.matdata._rhot_companion.write_depends_rhot_group`, called by default from
// `export_rhot_companion`) writes these from the SAME free energy F the (P,T) table is built
// from, so the two agree on the surface and only differ in index direction.
//
// Units are CGS throughout (g/cm^3, K, barye, erg/g) -- NOT the table-native GPa/MJ-per-kg an
// earlier version of this comment claimed; verified against every shipped table, which also
// carry the `unit_system = "cgs"` group attribute.
//
// `dPdRho` and `dEdRho` here are the CONSTANT-TEMPERATURE partials, the natural pairing for a
// (rho,T) surface.  Do not assume the same for the identically named field under
// `dependsLogRhoLogT`: that one is (dP/drho)_E, at constant ENERGY.

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

// Recompute everything derived from the node arrays / P_ rather than read from the file.
//
// MUST be called both at load AND after any deserialization that repoints the databoxes.
// `mMono_` is a std::vector<int>, deliberately NOT in DBLIST (it is derived, not table data), so
// the serialize -> DeSerialize round trip the MPI shared-memory path performs reconstructs an
// object whose mMono_ header still refers to the SERIALIZING rank's heap. It is non-empty, so
// the `!mMono_.empty()` guard in densityOfPT_ passes and `mMono_[j]` then reads freed memory --
// a segfault on the first (P,T) root, on every rank, independent of the (P,T) values. That is
// exactly the RUN023/RUN026 crash: only densityOfPT_ touches mMono_, so the DALTON closure
// (evalRhoT_ only) survived the same tables, and single-process offline probes never round trip
// and so never reproduced it.
//
// The axis-kind scalars are rebuilt here too. They would not segfault, but after a round trip
// they are not guaranteed to correspond to the deserialized grids, and a wrong cell index
// silently interpolates in the wrong interval -- worse than a crash.
inline void TableDependsRhoT::rebuildDerived_() {
  // Top monotone-increasing run start per T-column: smallest m with P(k+1,j) > P(k,j) for all
  // k in [m, numRho_-2].  A localized density scan starting at k+1 >= m (on the high-P side)
  // cannot skip a denser increasing crossing, so it matches the full dense-end scan exactly.
  mMono_.resize(numT_);
  for (int j = 0; j < numT_; ++j) {
    int m = numRho_ - 1;
    while (m - 1 >= 0 && P_(m, j) > P_(m - 1, j)) --m;
    mMono_[j] = m;
  }

  // Axis spacing -> whether cell_() can index in O(1) or has to search.  See sg_rhot_grid.
  rhoAxisKind_ = sg_rhot_grid::classify(rhoGrid_.data(), numRho_);
  tAxisKind_ = sg_rhot_grid::classify(T_.data(), numT_);
  if (rhoAxisKind_ == sg_rhot_grid::UNIFORM_LINEAR) {
    rhoInvStep_ = (numRho_ - 1) / (rhoGrid_(numRho_ - 1) - rhoGrid_(0));
  } else if (rhoAxisKind_ == sg_rhot_grid::UNIFORM_LOG) {
    logRhoMin_ = std::log(rhoGrid_(0));
    rhoInvStep_ = (numRho_ - 1) / (std::log(rhoGrid_(numRho_ - 1)) - logRhoMin_);
  }
  if (tAxisKind_ == sg_rhot_grid::UNIFORM_LINEAR) {
    tInvStep_ = (numT_ - 1) / (T_(numT_ - 1) - T_(0));
  } else if (tAxisKind_ == sg_rhot_grid::UNIFORM_LOG) {
    logTMin_ = std::log(T_(0));
    tInvStep_ = (numT_ - 1) / (std::log(T_(numT_ - 1)) - logTMin_);
  }

  // Prove the arithmetic index before trusting it.  No table shipped today is evenly spaced, so
  // the O(1) branch would otherwise go into production having never executed on real data; and a
  // wrong cell index is not a small error, it silently interpolates in the wrong interval.  Walk
  // every node and every midpoint, demand the fast index reproduce the bisection exactly, and
  // demote the axis to NONUNIFORM on any disagreement.  Load-time only.
  demoteAxisIfIndexDisagrees_(rhoGrid_, numRho_, rhoAxisKind_, rhoInvStep_, logRhoMin_);
  demoteAxisIfIndexDisagrees_(T_, numT_, tAxisKind_, tInvStep_, logTMin_);

  demoteAxisIfIndexDisagrees_(rhoGrid_, numRho_, rhoAxisKind_, rhoInvStep_, logRhoMin_);
  demoteAxisIfIndexDisagrees_(T_, numT_, tAxisKind_, tInvStep_, logTMin_);
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

  rebuildDerived_();

  sg_rhot_grid::g_records.push_back({matid_, rhoAxisKind_, tAxisKind_, numRho_, numT_});

  // Optional L-V dome for branch-aware (P,T)->rho: root /saturation/vapor_liquid/{temperature,
  // p_sat}. Absent => satT_/satPsat_ stay empty => dense-only root (backward-compatible). p_sat
  // is stored in GPa (matdata-canonical) -> convert to cgs (dyne/cm^2) to match the surface P_.
  if (H5Lexists(file, "saturation", H5P_DEFAULT) > 0) {
    hid_t satRoot = H5Gopen2(file, "saturation", H5P_DEFAULT);
    if (satRoot >= 0) {
      if (H5Lexists(satRoot, "vapor_liquid", H5P_DEFAULT) > 0) {
        hid_t vl = H5Gopen2(satRoot, "vapor_liquid", H5P_DEFAULT);
        if (vl >= 0) {
          hid_t tds = H5Dopen2(vl, "temperature", H5P_DEFAULT);
          if (tds >= 0) {
            hid_t sp = H5Dget_space(tds);
            const hssize_t nsat = H5Sget_simple_extent_npoints(sp);
            if (nsat > 1) {
              satT_.resize(static_cast<std::size_t>(nsat));
              satPsat_.resize(static_cast<std::size_t>(nsat));
              H5LTread_dataset_double(vl, "temperature", satT_.data());
              H5LTread_dataset_double(vl, "p_sat", satPsat_.data());
              for (auto &x : satPsat_) x *= 1.0e10; // GPa -> dyne/cm^2 (cgs)
            }
            H5Sclose(sp);
            H5Dclose(tds);
          }
          H5Gclose(vl);
        }
      }
      H5Gclose(satRoot);
    }
  }

  spiner_common::h5_safe_gclose(grp);
  spiner_common::h5_safe_gclose(matGroup);
  return 0;
}

// ============================ evaluation ====================================

// Dense-branch density root: scan the density grid from the dense end and take the densest
// interval where P(rho;T) crosses `press` while INCREASING (P_rho>0) -- the condensed branch.
// Off-branch queries clamp to the nearer grid end.  (Port of RhoTPteEos._density_at.)
PORTABLE_INLINE_FUNCTION Real TableDependsRhoT::densityOfPT_(Real press, Real temp) const {
  SG_RHOT_COUNT(g_densityOfPT);
  const Real t = clampT_(temp);
  int j;
  Real w;
  tWeight_(t, j, w);
  auto Pcol = [&](int i) -> Real { return (1.0 - w) * P_(i, j) + w * P_(i, j + 1); };
  // Root within [rhoGrid_(k), rhoGrid_(k+1)].  Pcol is LINEAR in rho across the cell, so the root
  // is the analytic linear interpolation -- ONE division, no iteration (the former 1e-12 bisection
  // ran ~40 iterations x several DataBox reads and dominated the whole accessor).  In a nearly
  // rho-INDEPENDENT cell (hot-plasma flat region) the ratio (press-pa)/(pb-pa) suffers
  // catastrophic cancellation (pa~pb~1e16 differing in the last few digits), so fall back to
  // sign-based bisection there (rare; matches the old root).  Steep condensed-branch cells (the
  // common mixed-cell case) take the cheap analytic path.
  auto rootInCell = [&](int k) -> Real {
    const Real pa = Pcol(k), pb = Pcol(k + 1);
    const Real denom = pb - pa;
    if (std::abs(denom) > 1.0e-6 * (std::abs(pa) + std::abs(pb))) { // well-conditioned
      const Real wr = (press - pa) / denom;
      return rhoGrid_(k) + wr * (rhoGrid_(k + 1) - rhoGrid_(k));
    }
    SG_RHOT_COUNT(g_rootBisect);
    Real rlo = rhoGrid_(k), rhi = rhoGrid_(k + 1); // flat cell: robust sign-based bisection
    for (int it = 0; it < 100; ++it) {
      const Real rm = 0.5 * (rlo + rhi);
      const Real wr = (rm - rhoGrid_(k)) / (rhoGrid_(k + 1) - rhoGrid_(k));
      const Real pm = (1.0 - wr) * pa + wr * pb - press;
      if (pm < 0.0)
        rlo = rm;
      else
        rhi = rm;
      if ((rhi - rlo) <= RHO_ROOT_TOL_ * (std::abs(rm) + RHO_ROOT_TOL_)) break;
    }
    return 0.5 * (rlo + rhi);
  };

  // Branch-aware: below P_sat(T) inside the L-V dome the physical single-phase root is the VAPOR
  // (least-dense increasing) crossing, not the dense one -- a raw loop-bearing surface exposes
  // both branches, so the dense-end scan below would wrongly return liquid for a vapor cell.
  // Scan from the LIGHT end for the first increasing crossing; if none is here (or no dome), fall
  // through to the dense scan. (Port of BranchAwareRhoTPteEos._vapor_density_at.)
  const Real psat = pSatAt_(t);
  if (psat > 0.0 && press < psat) {
    for (int k = 0; k <= numRho_ - 2; ++k) {
      const Real d0 = Pcol(k) - press;
      const Real d1 = Pcol(k + 1) - press;
      if (d0 == 0.0) return rhoGrid_(k);
      if (d1 == 0.0) return rhoGrid_(k + 1);
      if (d0 < 0.0 && 0.0 < d1) return rootInCell(k); // least-dense increasing => vapor branch
    }
  }

  // Fast path: the DENSEST increasing crossing lives in the column's top monotone-increasing run
  // [mMonoQ, numRho_-1] (strictly increasing, so any crossing there is denser than anything below
  // it).  When press is within that run's pressure range, BINARY-SEARCH it -- O(log numRho) vs the
  // O(numRho) dense-end linear scan -- and the result is identical to the full scan (same densest
  // crossing).  If press is at/below the run's floor (=> the crossing, if any, is in the
  // low-density non-monotone region) or the run degenerates, fall through to the full scan.
  if (!mMono_.empty()) {
    const int mMonoQ = (j + 1 < numT_) ? std::max(mMono_[j], mMono_[j + 1]) : mMono_[j];
    if (mMonoQ <= numRho_ - 2) {
      const Real p_floor = Pcol(mMonoQ), p_top = Pcol(numRho_ - 1);
      if (press > p_floor && press <= p_top) {
        int lo = mMonoQ, hi = numRho_ - 1; // Pcol(lo) < press <= Pcol(hi), run increasing
        while (hi - lo > 1) {
          const int mid = (lo + hi) / 2;
          if (Pcol(mid) < press)
            lo = mid;
          else
            hi = mid;
        }
        if (Pcol(hi) == press) return rhoGrid_(hi); // exact node (matches full-scan d==0 branch)
        return rootInCell(lo);
      }
    }
  }

  // Full scan from the dense end (high i) toward low density for the densest increasing crossing.
  for (int k = numRho_ - 2; k >= 0; --k) {
    const Real d0 = Pcol(k) - press;
    const Real d1 = Pcol(k + 1) - press;
    if (d1 == 0.0) return rhoGrid_(k + 1);
    if (d0 == 0.0) return rhoGrid_(k);
    if (d0 < 0.0 && 0.0 < d1) return rootInCell(k); // increasing crossing => condensed branch
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
// NB: this ~40-iteration full-range bisection (each iter an evalRhoT_) drives a large share of
// the cyclic solve's interpolation work (bench_pte_solve.cpp); a T-grid binary-search + analytic
// in-cell solve is ~5x cheaper but needs a robust non-monotone (cv fit-artifact) fallback -- see
// pte_tools/prove_tfrome_fastpath.py.  Kept as bisection until that is proven identical.
template <typename Indexer_t>
PORTABLE_INLINE_FUNCTION Real TableDependsRhoT::TemperatureFromDensityInternalEnergy(
    const Real rho, const Real sie, Indexer_t &&lambda) const {
  SG_RHOT_COUNT(g_TfromE);
  const Real r = std::min(std::max(rho, rhoMin_), rhoMax_);
  Real tlo = Tmin_, thi = Tmax_;
  for (int it = 0; it < 100; ++it) {
    const Real tm = 0.5 * (tlo + thi);
    if (InternalEnergyFromDensityTemperature(r, tm, lambda) < sie)
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
  // A (P,T)->(rho,e) request resolves rho first; everything else is a (rho,T) evaluation.
  const bool from_pt =
      (output & thermalqs::density) && (output & thermalqs::specific_internal_energy);
  if (from_pt) {
    DensityEnergyFromPressureTemperature(press, temp, lambda, rho, energy);
  }
  // FUSE: fill any remaining requested (rho,T) quantities (P, e, cv, bmod) from ONE evalRhoT_
  // instead of a separate accessor per quantity each re-interpolating the same point.
  // Bitwise-identical (same clamps, same interpolant; cv = de/dT, bmod = rho dP/drho).
  const unsigned long rhoT_out =
      (from_pt ? 0UL : (thermalqs::pressure | thermalqs::specific_internal_energy)) |
      thermalqs::specific_heat | thermalqs::bulk_modulus;
  if (output & rhoT_out) {
    const Real r = std::min(std::max(rho, rhoMin_), rhoMax_);
    Real P, e, p_rho, p_t, e_rho, e_t;
    evalRhoT_(r, clampT_(temp), P, e, p_rho, p_t, e_rho, e_t);
    if (!from_pt && (output & thermalqs::pressure)) press = P;
    if (!from_pt && (output & thermalqs::specific_internal_energy)) energy = e;
    if (output & thermalqs::specific_heat) cv = e_t;
    if (output & thermalqs::bulk_modulus) bmod = r * p_rho;
  }
}

} // namespace singularity

#endif // SINGULARITY_USE_PT_TABLES
#endif // _SINGULARITY_EOS_EOS_EOS_TABLE_RHOT_HPP_
