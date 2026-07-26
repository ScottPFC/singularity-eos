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

#ifndef _SINGULARITY_EOS_CLOSURE_MIXED_CELL_MODELS_
#define _SINGULARITY_EOS_CLOSURE_MIXED_CELL_MODELS_

#include <ports-of-call/portability.hpp>
#include <ports-of-call/portable_errors.hpp>
#include <singularity-eos/base/error_utils.hpp>
#include <singularity-eos/base/fast-math/logs.hpp>
#include <singularity-eos/base/robust_utils.hpp>
#include <singularity-eos/base/variadic_utils.hpp>
#include <singularity-eos/eos/eos.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

#ifdef SINGULARITY_USE_KOKKOSKERNELS
#include <KokkosBatched_ApplyQ_Decl.hpp>
#include <KokkosBatched_QR_Decl.hpp>
#include <KokkosBatched_Trsm_Decl.hpp>
#else
#include <Eigen/Dense>
#endif // SINGULARITY_USE_KOKKOSKERNELS

namespace singularity {

// ======================================================================
// Implementation details below
// ======================================================================

struct MixParams {
  bool verbose = false;
  bool iterate_t_guess = true;
  Real derivative_eps = 3.0e-6;
  Real pte_rel_tolerance_p = 1.0e-6;
  Real pte_rel_tolerance_t = 1.0e-4;
  Real pte_rel_tolerance_e = 1.0e-6;
  Real pte_rel_tolerance_v = 1.0e-6;
  Real pte_abs_tolerance_p = 1.0e-6;
  Real pte_abs_tolerance_t = 1.0e-4;

  Real pte_slow_convergence_thresh = 0.99;
  Real pte_rel_tolerance_p_sufficient = 1.e2 * pte_rel_tolerance_p;
  Real pte_rel_tolerance_t_sufficient = 10 * pte_rel_tolerance_t;
  Real pte_rel_tolerance_e_sufficient = 1.e2 * pte_rel_tolerance_e;
  Real pte_rel_tolerance_v_sufficient = 1.e2 * pte_rel_tolerance_v;
  Real pte_abs_tolerance_p_sufficient = 1.e2 * pte_abs_tolerance_p;
  Real pte_abs_tolerance_t_sufficient = 10 * pte_abs_tolerance_t;

  std::size_t pte_max_iter_per_mat = 128;
  Real line_search_alpha = 1.e-2;
  std::size_t line_search_max_iter = 6;
  Real line_search_fac = 0.5;
  Real vfrac_safety_fac = 0.95;
  Real temperature_limit = 1.0e15;
  Real default_tguess = 300.;
  Real min_dtde = 1.0e-16;
  std::size_t pte_small_step_tries = 2;
  Real pte_small_step_thresh = 1e-16;
  Real pte_max_dpdv = -1e-8;

  // Pressure-sign-mismatch jump: when some materials have deeply negative
  // pressure while others are positive, bisect to find the P=0 crossing and
  // jump the negative-P material to escape the cold-curve binding well.
  bool pte_pressure_jump_enabled = true;
  std::size_t pte_pressure_jump_bisect_iters = 20;
  std::size_t pte_pressure_jump_max_attempts = 3;
  Real pte_pressure_jump_min_vfrac = 1.0e-8;

  // Stagnation-gated pressure jump: when the error ratio exceeds this
  // threshold for pte_stagnation_tries consecutive iterations and the
  // solution is not close_enough, attempt a pressure jump.  This catches
  // the failure mode where ScaleDx() never returns a tiny value (so
  // small_step_iters is never incremented) but the solver is stuck in a
  // linear drift with one material deep in a negative-pressure binding well.
  Real pte_stagnation_thresh = 0.999;
  std::size_t pte_stagnation_tries = 10;

  // JMM: note the deliberate integer here. Negative value means do
  // the default. Assuming here that we never have a problem with 2^63
  // materials...
  std::int64_t pte_reduced_system_exclude_idx = -1;
};

struct SolverStatus {
  bool converged = false;
  bool slow_convergence_detected = false;
  std::size_t max_niter = 0;
  std::size_t max_line_niter = 0;
  std::size_t small_step_iters = 0;
  std::size_t stagnation_iters = 0;
  std::size_t pressure_jump_attempts = 0;
  Real residual;
};

namespace mix_impl {
template <typename T,
          typename = typename std::enable_if<std::is_floating_point<T>::value>::type>
constexpr bool isfinite(const T &a) {
  return (a == a) && ((a == 0) || (a != 2 * a));
}

constexpr Real square(const Real x) { return x * x; }

PORTABLE_INLINE_FUNCTION
bool check_nans(Real const *const a, const std::size_t n, const bool verbose = false) {
  bool retval = true;
  for (std::size_t i = 0; i < n; ++i)
    if (!isfinite(a[i])) {
      retval = false;
#ifndef KOKKOS_ENABLE_CUDA
      if (verbose) {
        printf("bad val in element %ld/%ld\n", i, n);
      }
#endif // KOKKOS_ENABLE_CUDA
    }
  return retval;
}

PORTABLE_INLINE_FUNCTION
bool solve_Ax_b_wscr(const std::size_t n, Real *a, Real *b, Real *scr) {
  // Simple Jacobi preconditioner
  //
  // JMM: This DOES seem to reliably reduce the condition number of
  // our matrices, suggesting they can be reliably transformed into a
  // diagonally dominant form. However, it's unclear to me if it
  // actually helps us solve difficult mixed cells. It seems to me
  // that when the matrix is ill-conditioned there is some other
  // feature of the problem making the solve difficult, perhaps the
  // shape of the residual space, or some source of catastrophic
  // cancellation we haven't been able to eliminate. That said, it's
  // basically free, so I decided to keep it.
  for (std::size_t row = 0; row < n; ++row) {
    Real maxabs = 0;
    for (std::size_t column = 0; column < n; ++column) {
      maxabs = std::max(std::abs(a[row * n + column]), maxabs);
    }
    if (maxabs == 0) {
      // row is all zeros. Matrix is not invertible.
      return false;
    }
    for (std::size_t column = 0; column < n; ++column) {
      a[row * n + column] /= maxabs;
    }
    b[row] /= maxabs;
  }

  if (n == 1) {                       // JMM: Just in case
    b[0] = robust::ratio(b[0], a[0]); // re-use b as x
  } else if (n == 2) {
    // JMM: Special case Cramer's rule for 2x2 matrices, where it is
    // both more efficient and more accurate. Note that at 3x3 and
    // greater, it becomes worse in both efficiency and
    // accuracy.
    Real *x = scr;
    const Real det = a[0] * a[3] - a[1] * a[2];
    x[0] = robust::ratio(a[3] * b[0] - a[1] * b[1], det);
    x[1] = robust::ratio(a[0] * b[1] - a[2] * b[0], det);
    b[0] = x[0];
    b[1] = x[1];
  } else {
    // TODO(JMM): Should we switch to an SVD-based psuedo-inverse for
    // larger matrices?
#ifdef SINGULARITY_USE_KOKKOSKERNELS
#ifndef PORTABILITY_STRATEGY_KOKKOS
#error "Kokkos Kernels requires Kokkos."
#endif
    // aliases for kokkos views
    using Unmgd = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    using Lrgt = Kokkos::LayoutRight;
    using vec_t = Kokkos::View<Real *, Unmgd>;
    // aliases for QR solve template params
    using QR_alg = KokkosBatched::Algo::QR::Unblocked;
    using Lft = KokkosBatched::Side::Left;
    using Trs = KokkosBatched::Trans::Transpose;
    using nTrs = KokkosBatched::Trans::NoTranspose;
    using ApQ_alg = KokkosBatched::Algo::ApplyQ::Unblocked;
    using UP = KokkosBatched::Uplo::Upper;
    using NonU = KokkosBatched::Diag::NonUnit;
    using Tr_alg = KokkosBatched::Algo::Trsm::Unblocked;
    // aliases for solver structs ('invoke' member of the struct is the
    // actual function call)
    using QR_factor = KokkosBatched::SerialQR<QR_alg>;
    using ApplyQ_transpose = KokkosBatched::SerialApplyQ<Lft, Trs, ApQ_alg>;
    using InvertR = KokkosBatched::SerialTrsm<Lft, UP, nTrs, NonU, Tr_alg>;
    // view of matrix
    Kokkos::View<Real **, Lrgt, Unmgd> A(a, n, n);
    // view of RHS
    Kokkos::View<Real **, Lrgt, Unmgd> B(b, n, 1);
    // view of reflectors
    vec_t t(scr, n);
    // view of workspace
    vec_t w(scr + n, n);
    // QR factor A, A x = B -> Q R x = B
    // store result in A and t
    QR_factor::invoke(A, t, w);
    // Apply Q^T from the left to both sides
    // Q^T Q R x = Q^T B -> R x = Q^T B
    // store result of Q^T B in B
    ApplyQ_transpose::invoke(A, t, B, w);
    // Apply R^-1 from the left to both sides
    // R^-1 R x = R^-1 Q^T B -> x = R^-1 Q^T B
    // store solution vector x in B
    InvertR::invoke(1.0, A, B);
#else
#ifdef PORTABILITY_STRATEGY_KOKKOS
#warning "Eigen should not be used with Kokkos."
#endif
    // Eigen VERSION
    using Matrix_t = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<Matrix_t> A(a, n, n);

    Eigen::Map<Eigen::VectorXd> B(b, n);
    Eigen::Map<Eigen::VectorXd> X(scr, n);
    X = A.colPivHouseholderQr().solve(B);

    B = X;
#endif // SINGULARITY_USE_KOKKOSKERNELS
  }
  bool retval = check_nans(b, n);
  return retval;
}

/*
 * Kahan summation, with the Neumaier correction
 * https://onlinelibrary.wiley.com/doi/10.1002/zamm.19740540106
 */
struct IdentityOperator {
  PORTABLE_FORCEINLINE_FUNCTION Real operator()(const Real x) const { return x; }
};
template <typename Data_t, typename Operator_t = IdentityOperator>
PORTABLE_FORCEINLINE_FUNCTION Real
sum_neumaier(Data_t &&data, std::size_t n, std::size_t offset = 0, int iskip = -1,
             const Operator_t &op = IdentityOperator()) {
  Real sum = 0;
  Real c = 0; // correction
  for (std::size_t i = 0; i < n; ++i) {
    if (iskip >= 0 && i == static_cast<std::size_t>(iskip)) continue;
    Real x = op(data[i + offset]);
    Real t = sum + x;
    if (std::abs(sum) >= std::abs(x)) {
      c += (sum - t) + x;
    } else {
      c += (x - t) + sum;
    }
    sum = t;
  }
  return sum + c;
}

class CacheAccessor {
 public:
  CacheAccessor() = default;
  PORTABLE_INLINE_FUNCTION
  explicit CacheAccessor(Real *scr) : cache_(scr) {}
  PORTABLE_FORCEINLINE_FUNCTION
  Real *operator[](const std::size_t m) const { return cache_ + m * MAX_NUM_LAMBDAS; }

  PORTABLE_FORCEINLINE_FUNCTION
  static std::size_t Size(const std::size_t nmat) { return MAX_NUM_LAMBDAS * nmat; }
  PORTABLE_FORCEINLINE_FUNCTION
  static std::size_t SizeInBytes(const std::size_t nmat) {
    return Size(nmat) * sizeof(Real);
  }

 private:
  Real *cache_;
};

// ======================================================================
// Base class
// ======================================================================
template <typename EOSIndexer, typename RealIndexer, typename LambdaIndexer>
class PTESolverBase {
 public:
  PTESolverBase() = delete;
  PORTABLE_INLINE_FUNCTION std::size_t Nmat() const { return nmat; }
  PORTABLE_INLINE_FUNCTION std::size_t &Niter() { return niter; }

  // A note on the virtual functions below. We are NOT using runtime
  // polymorphism. The point of these virtual functions is to provide
  // default implementations for functions that are shared between
  // solvers. But a given solver should be treated by value, not with
  // pointers, and avoid relocatable device code.

  // Fixup is meant to be a hook for derived classes to provide arbitrary manipulations
  // after each iteration of the Newton solver.  This version just renormalizes the
  // volume fractions, which is useful to deal with roundoff error.
  // See comment above about virtual keyword
  PORTABLE_INLINE_FUNCTION
  virtual void Fixup() { NormalizeVfrac(); }
  // Finalize restores the temperatures, energies, and pressures to unscaled values from
  // the internally scaled quantities used by the solvers
  // See comment above about virtual keyword
  PORTABLE_INLINE_FUNCTION
  virtual void Finalize() {
    for (std::size_t m = 0; m < nmat; m++) {
      temp[m] *= Tnorm;
      u[m] *= uscale;
      press[m] *= uscale;
    }
  }
  // Solve the linear system for the update dx
  PORTABLE_INLINE_FUNCTION
  bool Solve() const {
    for (std::size_t m = 0; m < neq; m++)
      dx[m] = residual[m];
    return solve_Ax_b_wscr(neq, jacobian, dx, sol_scratch);
  }
  // Parameters for the solver
  PORTABLE_INLINE_FUNCTION
  const MixParams &GetParams() const { return params_; }

  // Attempt to escape a cold-curve binding well by jumping materials
  // across the spinodal to a stable branch.  This fires when the Newton
  // solver is stuck due to pressure sign mismatches or ill-conditioned
  // Jacobians near the spinodal boundary.
  // T_physical is the unscaled equilibrium temperature.
  //
  // Three phases (implemented in separate helpers below):
  //   1. TryVaporJump — jump materials to the vapor-stable branch
  //   2. TryDenseSideJump — jump materials to the dense-stable branch
  //   3. NormalizeAndRecomputeAfterJump — fix up vfracs and recompute state
  PORTABLE_INLINE_FUNCTION
  bool TryPressureJump(const Real T_physical, const MixParams &params) {
    const Real min_vfrac = params.pte_pressure_jump_min_vfrac;
    const Real abs_tol_p = params.pte_abs_tolerance_p;
    const std::size_t max_bisect = params.pte_pressure_jump_bisect_iters;
    // Safety factor to keep jump targets off the spinodal boundary.
    // Multiplying a spinodal density by this factor pushes the result
    // slightly into the adjacent stable region, avoiding the zone where
    // dP/drho ≈ 0 and the Jacobian is ill-conditioned.  Used on both
    // sides: vapor-side (target = spinodal_safety * RhoSpinodalVapor)
    // and dense-side (bisection upper bound at rho > RhoPmin).
    constexpr Real spinodal_safety = 0.9;

#ifdef PTE_DEBUG_TRACE
    std::printf("  [TryPressureJump] T_physical=%.6e  abs_tol_p=%.6e  max_bisect=%zu\n",
                T_physical, abs_tol_p, max_bisect);
    for (std::size_t m = 0; m < nmat; ++m) {
      const Real phys_press = press[m] * uscale;
      std::printf("    mat[%zu]: press_scaled=%.6e  uscale=%.6e  phys_press=%.6e  "
                  "rho=%.6e  vfrac=%.6e  rhobar=%.6e\n",
                  m, press[m], uscale, phys_press, rho[m], vfrac[m], rhobar[m]);
    }
#endif

    // Phase 1: Try vapor-side jumps
    bool any_jumped = TryVaporJump(T_physical, abs_tol_p, max_bisect,
                                   min_vfrac, spinodal_safety);

    // Phase 2: Dense-side fallback (only if no vapor jumps were made)
    if (!any_jumped) {
      any_jumped = TryDenseSideJump(T_physical, max_bisect,
                                      min_vfrac, spinodal_safety);
    }

    if (!any_jumped) {
#ifdef PTE_DEBUG_TRACE
      std::printf("    => No materials jumped, returning false\n");
      std::fflush(stdout);
#endif
      return false;
    }

    // Phase 3: Normalize volume fractions and recompute thermodynamic state
    NormalizeAndRecomputeAfterJump(T_physical);
    return true;
  }

 protected:
  PORTABLE_INLINE_FUNCTION
  PTESolverBase(std::size_t nmats, std::size_t neqs, const EOSIndexer &eos_,
                const Real vfrac_tot, const Real sie_tot, const RealIndexer &rho_,
                const RealIndexer &vfrac_, const RealIndexer &sie_,
                const RealIndexer &temp_, const RealIndexer &press_,
                LambdaIndexer &lambda_, Real *&scratch, Real Tnorm,
                const MixParams &params = MixParams())
      : params_(params), nmat(nmats), neq(neqs), niter(0), vfrac_total(vfrac_tot),
        sie_total(sie_tot), eos(eos_), rho(rho_), vfrac(vfrac_), sie(sie_), temp(temp_),
        press(press_), lambda(lambda_), Tnorm(Tnorm) {
    jacobian = AssignIncrement(scratch, neq * neq);
    dx = AssignIncrement(scratch, neq);
    sol_scratch = AssignIncrement(scratch, 2 * neq);
    residual = AssignIncrement(scratch, neq);
    u = AssignIncrement(scratch, nmat);
    rhobar = AssignIncrement(scratch, nmat);
  }

  PORTABLE_FORCEINLINE_FUNCTION
  void InitRhoBarandRho() {
    // rhobar is a fixed quantity: the average density of
    // material m averaged over the full PTE volume
    for (std::size_t m = 0; m < nmat; ++m) {
      PORTABLE_REQUIRE(vfrac[m] > 0.,
                       "Non-positive volume fraction provided to PTE solver");
      PORTABLE_REQUIRE(rho[m] > 0., "Non-positive density provided to PTE solver");
      rhobar[m] = rho[m] * vfrac[m];
    }
    rho_total = sum_neumaier(rhobar, nmat);
  }

  PORTABLE_FORCEINLINE_FUNCTION
  void NormalizeVfrac() const {
    Real vfrac_sum = sum_neumaier(vfrac, nmat);
    for (std::size_t m = 0; m < nmat; ++m) {
      vfrac[m] *= robust::ratio(vfrac_total, vfrac_sum);
    }
  }

  // -----------------------------------------------------------------
  // Compute a reference pressure (P_ref) from stable materials for
  // spinodal jump targeting.  Returns the vfrac-weighted average
  // pressure of materials that are thermodynamically stable:
  //   - No spinodal (RhoPmin <= 0): EOS has no vapor dome
  //   - Dense-stable: rho > RhoPmin with positive pressure
  //   - Vapor-stable: rho <= RhoSpinodalVapor (on the stable vapor branch)
  // Falls back to the largest-vfrac material's pressure if no
  // material qualifies as stable.
  // -----------------------------------------------------------------
  PORTABLE_INLINE_FUNCTION
  Real ComputePRef(const Real T_physical) const {
    Real P_ref_num = 0;
    Real P_ref_den = 0;
    for (std::size_t m = 0; m < nmat; ++m) {
      const Real rpm = eos[m].RhoPmin(T_physical);
      const Real rho_vapor = eos[m].RhoSpinodalVapor(T_physical);
      bool m_stable = (rpm <= 0 ||
                       (rho[m] > rpm && press[m] * uscale > 0) ||
                       (rho_vapor > 0 && rho[m] <= rho_vapor));
      if (m_stable) {
        P_ref_num += vfrac[m] * press[m] * uscale;
        P_ref_den += vfrac[m];
      }
    }
    if (P_ref_den > 0) {
      return P_ref_num / P_ref_den;
    }
    // Fallback: no material qualifies as stable.  Use the
    // largest-vfrac material's pressure as a best-effort reference.
    std::size_t dom = 0;
    for (std::size_t m = 1; m < nmat; ++m) {
      if (vfrac[m] > vfrac[dom]) dom = m;
    }
    return press[dom] * uscale;
  }

  // -----------------------------------------------------------------
  // TryPressureJump helper: vapor-side spinodal jump.
  //
  // Handle materials that need to cross the spinodal to the stable
  // vapor branch.  Three trigger conditions:
  //   (a) Negative pressure — material is in tension, needs to expand
  //   (b) Pinned at dense-side spinodal — rho ≈ RhoPmin, solver
  //       stagnates because dP/drho ≈ 0 at the spinodal boundary
  //       (ill-conditioned Jacobian).  RhoPmin is a local P minimum,
  //       so the equilibrium can be at lower rho (in the unstable
  //       region where P is higher).
  //   (c) Drifted past spinodal into unstable region — rho < RhoPmin
  //       but rho > RhoSpinodalVapor.  The ScaleDx escape hatch
  //       allowed rho to drop below RhoPmin during Newton iteration.
  //
  // When the full spinodal crossing is feasible, the target density is
  // spinodal_safety * RhoSpinodalVapor(T).  When vfrac_hi_max caps
  // the jump (can't expand enough to cross the spinodal), a P=0
  // bisection finds the vfrac where the jumped material's pressure
  // reaches zero.  This is a natural, moderate target — it only
  // expands the material enough to get out of tension, minimizing
  // compression of other materials and the energy mismatch that
  // causes Newton to crash the temperature.
  //
  // All materials are checked uniformly; the jump conditions
  // themselves correctly identify which materials need intervention.
  //
  // Jumped materials have their vfrac set negative (sign convention)
  // so that NormalizeAndRecomputeAfterJump can give them priority.
  // -----------------------------------------------------------------
  PORTABLE_INLINE_FUNCTION
  bool TryVaporJump(const Real T_physical, const Real abs_tol_p,
                    const std::size_t max_bisect, const Real min_vfrac,
                    const Real spinodal_safety) {
    // Margin around RhoPmin for detecting materials pinned at the
    // dense-side spinodal where dP/drho ~ 0 causes solver stagnation.
    // A tight margin (5%) ensures we only jump materials that are
    // genuinely stuck right at the pressure minimum.
    constexpr Real pinned_margin = 1.05;
    // Defensive density floor for the P=0 bisection upper bound.
    // Prevents the bisection from exploring densities where the EOS may
    // not be well-defined.  In practice vfrac_hi_max almost always binds
    // first; this only matters when rhobar is extremely small.
    constexpr Real min_bisection_density = 1.0e-12;
    // Maximum factor by which non-jumped materials' densities may
    // increase as a result of the jump.  Limits the jumped material's
    // vfrac so that other materials keep vfrac >= vfrac_current / cap,
    // i.e. rho_post <= cap * rho_current.  Prevents the priority
    // normalization from crushing minority materials into extreme
    // densities that the solver cannot recover from.
    constexpr Real density_change_cap = 10.0;

    bool any_jumped = false;

    for (std::size_t m = 0; m < nmat; ++m) {
      const Real phys_press = press[m] * uscale;
      const Real rho_pmin = eos[m].RhoPmin(T_physical);

      bool need_vapor_jump = false;

      // Condition (a): negative pressure
      if (phys_press < -abs_tol_p) {
        need_vapor_jump = true;
#ifdef PTE_DEBUG_TRACE
        std::printf("    mat[%zu]: NEGATIVE P (%.6e), need vapor jump\n", m, phys_press);
#endif
      }
      // Condition (b): pinned at dense-side spinodal (local P minimum).
      // dP/drho ≈ 0 here, so Newton stagnates.  The equilibrium may be
      // at lower rho in the unstable region where P rises toward the
      // vapor-side peak.
      else if (rho_pmin > 0 && rho[m] >= rho_pmin &&
               rho[m] <= pinned_margin * rho_pmin) {
        need_vapor_jump = true;
#ifdef PTE_DEBUG_TRACE
        std::printf("    mat[%zu]: PINNED at spinodal (rho=%.6e, RhoPmin=%.6e, "
                    "P=%.6e), need vapor jump\n",
                    m, rho[m], rho_pmin, phys_press);
#endif
      }
      // Condition (c): drifted past dense-side spinodal into unstable region.
      // The ScaleDx escape hatch allowed rho to drop below RhoPmin.
      // If the material is between the two spinodal edges, jump to the
      // vapor-side stable branch rather than letting the dense-side
      // fallback push it back to the dense side.
      else if (rho_pmin > 0 && rho[m] < rho_pmin) {
        const Real rho_vapor = eos[m].RhoSpinodalVapor(T_physical);
        if (rho_vapor > 0 && rho[m] > rho_vapor) {
          need_vapor_jump = true;
#ifdef PTE_DEBUG_TRACE
          std::printf("    mat[%zu]: IN UNSTABLE REGION (rho=%.6e, RhoPmin=%.6e, "
                      "RhoSpinodalVapor=%.6e, P=%.6e), need vapor jump\n",
                      m, rho[m], rho_pmin, rho_vapor, phys_press);
#endif
        }
      }

      if (!need_vapor_jump) {
#ifdef PTE_DEBUG_TRACE
        std::printf(
            "    mat[%zu]: rho=%.6e  RhoPmin=%.6e  P=%.6e  no vapor jump needed\n", m,
            rho[m], rho_pmin, phys_press);
#endif
        continue;
      }

      // Maximum vfrac this material can take without increasing other
      // materials' densities by more than density_change_cap.
      Real vfrac_hi_max = vfrac_total;
      for (std::size_t k = 0; k < nmat; ++k) {
        if (k == m) continue;
        vfrac_hi_max -= std::max(min_vfrac, vfrac[k] / density_change_cap);
      }
      vfrac_hi_max = std::max(vfrac_hi_max, min_vfrac);
#ifdef PTE_DEBUG_TRACE
      std::printf("    mat[%zu]: vfrac_hi_max=%.6e\n", m, vfrac_hi_max);
#endif

      const Real rho_vapor = eos[m].RhoSpinodalVapor(T_physical);
      if (rho_vapor > 0) {
        const Real ideal_vfrac =
            robust::ratio(rhobar[m], spinodal_safety * rho_vapor);
        if (ideal_vfrac <= vfrac_hi_max) {
          // Can cross the spinodal — jump to vapor-stable region
          vfrac[m] = ideal_vfrac;
#ifdef PTE_DEBUG_TRACE
          std::printf("    mat[%zu]: VAPOR JUMP (full) to RhoSpinodalVapor=%.6e "
                      "(safety*=%.6e), vfrac=%.6e, rho=%.6e\n",
                      m, rho_vapor, spinodal_safety * rho_vapor, vfrac[m],
                      robust::ratio(rhobar[m], vfrac[m]));
#endif
        } else {
          // Cannot cross spinodal — vfrac_hi_max caps the jump.
          // Bisect within [current vfrac, vfrac_hi_max] for P=0: expand
          // just enough to get out of tension.  This minimizes
          // compression of non-jumped materials and the energy mismatch.
          Real lo = vfrac[m];     // current vfrac (high density, P < 0)
          Real hi = vfrac_hi_max; // max expansion (low density end)

          const Real rho_hi = robust::ratio(rhobar[m], hi);
          const Real P_hi =
              eos[m].PressureFromDensityTemperature(rho_hi, T_physical, lambda[m]);

          if (P_hi < 0.0) {
            // No P=0 crossing in feasible range.  Jump to max only if
            // it reduces the magnitude of negative pressure.
            if (std::abs(P_hi) < std::abs(phys_press)) {
              vfrac[m] = hi;
#ifdef PTE_DEBUG_TRACE
              std::printf("    mat[%zu]: VAPOR JUMP (capped, P_hi<0 but improving "
                          "%.6e -> %.6e), vfrac=%.6e, rho=%.6e\n",
                          m, phys_press, P_hi, vfrac[m],
                          robust::ratio(rhobar[m], vfrac[m]));
#endif
            } else {
#ifdef PTE_DEBUG_TRACE
              std::printf("    mat[%zu]: VAPOR JUMP (capped, P_hi<0 not improving "
                          "%.6e vs %.6e), skipping\n",
                          m, phys_press, P_hi);
#endif
              continue;
            }
          } else {
            // P_hi >= 0: bisect for P = 0 crossing
            for (std::size_t iter = 0; iter < max_bisect; ++iter) {
              const Real mid = 0.5 * (lo + hi);
              const Real rho_mid = robust::ratio(rhobar[m], mid);
              const Real P_mid =
                  eos[m].PressureFromDensityTemperature(rho_mid, T_physical, lambda[m]);
              if (P_mid < 0.0) {
                lo = mid;
              } else {
                hi = mid;
              }
            }
            vfrac[m] = hi;
#ifdef PTE_DEBUG_TRACE
            std::printf("    mat[%zu]: VAPOR JUMP (capped, P=0 bisection) "
                        "vfrac=%.6e, rho=%.6e\n",
                        m, vfrac[m], robust::ratio(rhobar[m], vfrac[m]));
#endif
          }
        }
        any_jumped = true;
        vfrac[m] = -vfrac[m]; // mark as vapor-jumped for priority normalization
      } else if (phys_press < -abs_tol_p) {
        // Fallback: P=0 bisection for materials without pmin_vapor_dome
        // (no precomputed spinodal curve).
        const Real lo_init = vfrac[m];
        Real hi = std::min(vfrac_hi_max, robust::ratio(rhobar[m], min_bisection_density));
#ifdef PTE_DEBUG_TRACE
        std::printf("    mat[%zu]: no spinodal curve, fallback P=0 bisection "
                    "lo=%.6e  hi=%.6e\n",
                    m, lo_init, hi);
#endif
        if (hi <= lo_init) continue;

        const Real rho_hi = robust::ratio(rhobar[m], hi);
        const Real P_hi =
            eos[m].PressureFromDensityTemperature(rho_hi, T_physical, lambda[m]);
        if (P_hi < 0.0) {
          // No P=0 crossing exists in the feasible range.  Jump anyway
          // if expanding to maximum feasible density reduces the magnitude
          // of negative pressure, giving Newton a better starting point.
          if (std::abs(P_hi) < std::abs(phys_press)) {
            vfrac[m] = hi;
            any_jumped = true;
            vfrac[m] = -vfrac[m]; // mark as vapor-jumped for priority normalization
#ifdef PTE_DEBUG_TRACE
            std::printf("    mat[%zu]: P_hi<0 but improving (%.6e -> %.6e), "
                        "jump to hi=%.6e\n",
                        m, phys_press, P_hi, hi);
#endif
          }
          continue;
        }

        // Bisect for P = 0 crossing
        Real lo = lo_init;
        for (std::size_t iter = 0; iter < max_bisect; ++iter) {
          const Real mid = 0.5 * (lo + hi);
          const Real rho_mid = robust::ratio(rhobar[m], mid);
          const Real P_mid =
              eos[m].PressureFromDensityTemperature(rho_mid, T_physical, lambda[m]);
          if (P_mid < 0.0) {
            lo = mid;
          } else {
            hi = mid;
          }
        }
        vfrac[m] = hi;
        any_jumped = true;
        vfrac[m] = -vfrac[m]; // mark as vapor-jumped for priority normalization
#ifdef PTE_DEBUG_TRACE
        std::printf("    mat[%zu]: P=0 bisection done: vfrac=%.6e  rho=%.6e\n", m,
                    -vfrac[m], robust::ratio(rhobar[m], -vfrac[m]));
#endif
      }
      // else: condition (b)/(c) but no spinodal curve — can't jump, skip
    }

    return any_jumped;
  }

  // -----------------------------------------------------------------
  // TryPressureJump helper: dense-side spinodal equilibration.
  //
  // No vapor-side jumps were made.  Check for materials on the
  // unstable side of the spinodal (rho < RhoPmin) and jump them to
  // the stable (dense) side, targeting P_ref — the vfrac-weighted
  // average pressure of all stable materials.
  //
  // Valid regardless of pressure signs: P_ref is computed from stable
  // materials only, and the bisection bounds handle out-of-range
  // targets by clamping to the nearest endpoint on the dense-stable
  // branch.  Even when P_ref is negative, the jump moves the material
  // onto a branch where dP/drho > 0 and the Jacobian is well-
  // conditioned, allowing Newton to make progress.
  //
  // Jumped materials keep positive vfrac (no sign convention needed)
  // since they are being pushed to the dense side, not the vapor side.
  // -----------------------------------------------------------------
  PORTABLE_INLINE_FUNCTION
  bool TryDenseSideJump(const Real T_physical,
                        const std::size_t max_bisect, const Real min_vfrac,
                        const Real spinodal_safety) {
#ifdef PTE_DEBUG_TRACE
    std::printf("    => Dense-side fallback: spinodal equilibration\n");
#endif

    const Real P_ref = ComputePRef(T_physical);
#ifdef PTE_DEBUG_TRACE
    std::printf("    P_ref=%.6e\n", P_ref);
#endif

    bool any_jumped = false;

    for (std::size_t m = 0; m < nmat; ++m) {
      const Real rho_pmin = eos[m].RhoPmin(T_physical);
#ifdef PTE_DEBUG_TRACE
      std::printf("    mat[%zu]: rho=%.6e  RhoPmin=%.6e", m, rho[m], rho_pmin);
#endif
      if (rho[m] >= rho_pmin) {
        // On the dense-stable side of the spinodal — Newton can handle this
#ifdef PTE_DEBUG_TRACE
        std::printf("  DENSE-STABLE, skipping\n");
#endif
        continue;
      }
      // Check if material is already on the vapor-stable branch
      // (past the vapor-side spinodal).  Only push materials back to
      // the dense side if they are genuinely in the unstable region.
      // For materials without a vapor spinodal curve (RhoSpinodalVapor
      // returns 0), we can't determine vapor stability, so we
      // conservatively proceed with the dense-side jump.
      {
        const Real rho_vapor = eos[m].RhoSpinodalVapor(T_physical);
        if (rho_vapor > 0 && rho[m] <= rho_vapor) {
#ifdef PTE_DEBUG_TRACE
          std::printf("  VAPOR-STABLE (rho=%.6e <= RhoSpinodalVapor=%.6e), skipping\n",
                      rho[m], rho_vapor);
#endif
          continue;
        }
      }
#ifdef PTE_DEBUG_TRACE
      std::printf("  UNSTABLE — need dense-side jump\n");
#endif

      // Material m is in the unstable region (RhoSpinodalVapor < rho < RhoPmin).
      // Bisect vfrac on the stable (dense) side to find P = P_ref.
      const Real rho_max = eos[m].MaximumDensity();
      Real lo = std::max(min_vfrac, robust::ratio(rhobar[m], rho_max));
      Real hi = spinodal_safety * robust::ratio(rhobar[m], rho_pmin);

      if (hi <= lo) {
        vfrac[m] = lo;
        any_jumped = true;
#ifdef PTE_DEBUG_TRACE
        std::printf("    mat[%zu]: degenerate range (hi=%.6e <= lo=%.6e), "
                    "jumping to vfrac=%.6e (rho_max)\n",
                    m, hi, lo, lo);
#endif
        continue;
      }

      const Real P_lo = eos[m].PressureFromDensityTemperature(
          robust::ratio(rhobar[m], lo), T_physical, lambda[m]);
      const Real P_hi = eos[m].PressureFromDensityTemperature(
          robust::ratio(rhobar[m], hi), T_physical, lambda[m]);

#ifdef PTE_DEBUG_TRACE
      std::printf("    mat[%zu]: bisect bounds: lo_vfrac=%.6e (rho=%.6e, P=%.6e)  "
                  "hi_vfrac=%.6e (rho=%.6e, P=%.6e)  P_ref=%.6e\n",
                  m, lo, robust::ratio(rhobar[m], lo), P_lo, hi,
                  robust::ratio(rhobar[m], hi), P_hi, P_ref);
#endif

      if (P_ref >= P_lo) {
        vfrac[m] = lo;
        any_jumped = true;
#ifdef PTE_DEBUG_TRACE
        std::printf("    mat[%zu]: P_ref >= P_lo, jumping to max density vfrac=%.6e\n",
                    m, lo);
#endif
        continue;
      }
      if (P_ref <= P_hi) {
        vfrac[m] = hi;
        any_jumped = true;
#ifdef PTE_DEBUG_TRACE
        std::printf("    mat[%zu]: P_ref <= P_hi, jumping to Pmin vfrac=%.6e\n", m, hi);
#endif
        continue;
      }

      // Normal bisection: P(lo) > P_ref > P(hi)
      for (std::size_t iter = 0; iter < max_bisect; ++iter) {
        const Real mid = 0.5 * (lo + hi);
        const Real rho_mid = robust::ratio(rhobar[m], mid);
        const Real P_mid =
            eos[m].PressureFromDensityTemperature(rho_mid, T_physical, lambda[m]);
        if (P_mid > P_ref) {
          lo = mid;
        } else {
          hi = mid;
        }
      }
      vfrac[m] = 0.5 * (lo + hi);
      any_jumped = true;

#ifdef PTE_DEBUG_TRACE
      {
        const Real rho_final = robust::ratio(rhobar[m], vfrac[m]);
        const Real P_final =
            eos[m].PressureFromDensityTemperature(rho_final, T_physical, lambda[m]);
        std::printf("    mat[%zu]: bisection done: vfrac=%.6e  rho=%.6e  P=%.6e\n", m,
                    vfrac[m], rho_final, P_final);
      }
#endif
    }

    return any_jumped;
  }

  // -----------------------------------------------------------------
  // TryPressureJump helper: normalize volume fractions and recompute
  // thermodynamic state after a pressure jump.
  //
  // Priority normalization: vapor-jumped materials (marked with
  // negative vfrac by TryVaporJump) keep their target vfrac;
  // non-jumped materials share the remaining volume proportionally.
  // If only dense-side jumps were made (all vfracs positive), falls
  // back to simple proportional normalization.
  //
  // After normalization, recomputes rho, sie, u, and press for all
  // materials at the new volume fractions.
  // -----------------------------------------------------------------
  PORTABLE_INLINE_FUNCTION
  void NormalizeAndRecomputeAfterJump(const Real T_physical) {
    // Renormalize volume fractions so they sum to vfrac_total.
    bool has_vapor_jumped = false;
    for (std::size_t m = 0; m < nmat; ++m) {
      if (vfrac[m] < 0) {
        has_vapor_jumped = true;
        break;
      }
    }
    if (has_vapor_jumped) {
      Real vfrac_jumped_sum = 0;
      Real vfrac_nonjumped_sum = 0;
      for (std::size_t m = 0; m < nmat; ++m) {
        if (vfrac[m] < 0) {
          vfrac_jumped_sum += (-vfrac[m]);
        } else {
          vfrac_nonjumped_sum += vfrac[m];
        }
      }
      Real vfrac_remaining = vfrac_total - vfrac_jumped_sum;
      if (vfrac_remaining > 0 && vfrac_nonjumped_sum > 0) {
        Real scale = vfrac_remaining / vfrac_nonjumped_sum;
#ifdef PTE_DEBUG_TRACE
        std::printf("    => Priority normalization: jumped_sum=%.6e "
                    "remaining=%.6e  non-jumped scale=%.6e\n",
                    vfrac_jumped_sum, vfrac_remaining, scale);
#endif
        for (std::size_t m = 0; m < nmat; ++m) {
          if (vfrac[m] < 0) {
            vfrac[m] = -vfrac[m]; // restore sign, keep target
          } else {
            vfrac[m] *= scale; // fill remaining volume
          }
        }
      } else {
        // Volume exhaustion fallback: restore signs and use proportional
#ifdef PTE_DEBUG_TRACE
        std::printf("    => Priority normalization fallback: jumped_sum=%.6e "
                    "exceeds vfrac_total=%.6e, using proportional\n",
                    vfrac_jumped_sum, vfrac_total);
#endif
        for (std::size_t m = 0; m < nmat; ++m) {
          if (vfrac[m] < 0) vfrac[m] = -vfrac[m];
        }
        NormalizeVfrac();
      }
    } else {
      NormalizeVfrac();
    }

    // Recompute state for all materials at the new volume fractions
    for (std::size_t m = 0; m < nmat; ++m) {
      rho[m] = robust::ratio(rhobar[m], vfrac[m]);
      const Real e_m =
          eos[m].InternalEnergyFromDensityTemperature(rho[m], T_physical, lambda[m]);
      sie[m] = e_m;
      u[m] = robust::ratio(rhobar[m] * e_m, uscale);
      press[m] = robust::ratio(
          eos[m].PressureFromDensityTemperature(rho[m], T_physical, lambda[m]), uscale);
    }

#ifdef PTE_DEBUG_TRACE
    std::printf("    => Post-jump state:\n");
    for (std::size_t m = 0; m < nmat; ++m) {
      std::printf("      mat[%zu]: rho=%.6e  vfrac=%.6e  press_phys=%.6e  sie=%.6e\n", m,
                  rho[m], vfrac[m], press[m] * uscale, sie[m]);
    }
    std::fflush(stdout);
#endif
  }

  /* JMM: Find the volume fractions that:
   * 1) Best respect initial guesses
   * 2) Sum to 1
   * 3) Are bounded from above such that no material has density less
   * than the density at minimum pressure, which avoids scenarios
   * where dP/drho < 0, where the solver can stuck.
   *
   * If these are not all possible to respect, error out and expect
   * the GetTguess function to save us by picking a higher
   * temperature.
   *
   * If this is possible to achieve, we can do so by using the "water
   * filling" algorithm. See, e.g., Gallager, 1968.
   */
  PORTABLE_FORCEINLINE_FUNCTION
  bool SetVfracFromT(const Real T) {
    constexpr bool FAIL = true;
    constexpr bool SUCCESS = false;
    constexpr Real min_dv = 1e-12;
    // Safety factor in maximum volume fraction keeps us on the dense
    // side of rho at P min, so that roundoff can't send us in to the
    // dP/drho < 0 region.
    constexpr Real safety = 0.9;

    // TODO(JMM): Stash this somewhere rather than recomputing it?
    auto get_vfrac_max = [=](const std::size_t m) {
      const Real rho_min = eos[m].RhoPmin(T);
      const Real vfrac_max = std::min(safety * robust::ratio(rhobar[m], rho_min), 1.0);
      return vfrac_max;
    };
    auto get_dv = [=](const std::size_t m) {
      const Real vfrac_max = get_vfrac_max(m);
      const Real vnew = std::min(vfrac_max, vfrac[m]);
      const Real dv = vfrac[m] - vnew;
      return dv;
    };

    // Do an initial sweep of volume fractions to account for missing
    // initial guesses
    // TODO(JMM): I'm not sure it's possible to pass in a missing
    // initial guess, since then rhobar would be undefined, which
    // would make the system underconstrained.
    Real vfrac_max_sum = 0;
    for (std::size_t m = 0; m < nmat; ++m) {
      const Real vfrac_max = get_vfrac_max(m);
      // This chooses the smallest allowed density, since vfrac is
      // inversely proportional to rho.
      if (vfrac[m] <= 0) vfrac[m] = vfrac_max;
      // Tally this to check if a valid state is possible
      vfrac_max_sum += vfrac_max;
    }
    if (vfrac_max_sum < vfrac_total) {
      // No way to have all volume fractions less than their allowed
      // maxima and sum to the correct total.
      return FAIL;
    }

    // start normalized
    NormalizeVfrac();

    // JMM: Idea here is keep a running tally of how much volume we've
    // had to subtract from a given material to keep vfrac <
    // vfrac_max. We will distribute this excess volume accross all
    // materials that haven't hit their max yet, and then
    // repeat. Formal maximum number of iterations required is nmat
    for (std::size_t niter = 0; niter < nmat; ++niter) {
      Real remaining_vfrac = 0;
      std::size_t n_uncapped_materials = nmat;
      for (std::size_t m = 0; m < nmat; ++m) {
        const Real dv = get_dv(m);
        vfrac[m] -= dv;
        remaining_vfrac += dv;
        n_uncapped_materials -= (dv > min_dv);
      }
      if (std::abs(remaining_vfrac) < min_dv) {
        break;
      }
      for (std::size_t m = 0; m < nmat; ++m) {
        const bool uncapped = ((get_vfrac_max(m) - vfrac[m]) > min_dv);
        if (uncapped) {
          vfrac[m] += robust::ratio(remaining_vfrac, n_uncapped_materials);
        }
      }
    }

    // One more normalization, to be safe
    NormalizeVfrac();

    return SUCCESS;
  }

  PORTABLE_FORCEINLINE_FUNCTION
  Real GetTguess() {
    // guess some non-zero temperature to start
    // If a guess was passed in, it's saved in Tnorm
    Real Tguess{};
    if (Tnorm > 0.0) {
      Tguess = Tnorm;
    } else {
      Tguess = params_.default_tguess;
      for (std::size_t m = 0; m < nmat; ++m)
        Tguess = std::max(Tguess, temp[m]);
    }
    PORTABLE_REQUIRE(Tguess > 0., "Non-positive temperature guess for PTE");
    // check for sanity.  basically checks that the input temperatures weren't garbage
    PORTABLE_REQUIRE(Tguess < params_.temperature_limit,
                     "Very large input temperature or temperature guess");

    // JMM: To get a better guess for temperature such that the
    // energies add up, we sometimes take one Newton step, and accept
    // if it makes the temperature larger. Empirically, we find this
    // avoids local saddle points that the solver can have trouble
    // navigating.
    auto newton_step = [=](Real T) {
      Real dudt = 0;
      Real usum = 0;
      for (std::size_t m = 0; m < nmat; ++m) {
        Real rho_max = eos[m].MaximumDensity();

        // TODO(JMM): Note that this assumes sensible pressures were
        // passed in as an initial guess.
        Real sie, cv;
        GetSieCvFromTAndPreferred(eos[m], std::min(rho[m], rho_max), press[m], T,
                                  lambda[m], sie, cv);

        usum += rhobar[m] * sie;
        dudt += rhobar[m] * cv;
      }
      return T - robust::ratio((usum - utotal_scale * uscale), dudt);
    };

    // iteratively increase temperature guess until all rho's are above rho_at_pmin
    const Real Tfactor = 10.0;
    bool bad_vfrac_guess;
    for (std::size_t i = 0; i < 3; i++) {
      if (params_.iterate_t_guess) {
        Tguess =
            std::min(params_.temperature_limit, std::max(Tguess, newton_step(Tguess)));
      }
      for (std::size_t m = 0; m < nmat; ++m) {
        Tguess = std::max(eos[m].MinimumTemperature(), Tguess);
      }
      bad_vfrac_guess = SetVfracFromT(Tguess);
      if (bad_vfrac_guess) {
        Tguess *= Tfactor;
        continue;
      }
      for (std::size_t m = 0; m < nmat; ++m) {
        rho[m] = robust::ratio(rhobar[m], vfrac[m]);
      }
      // check to make sure the normalization didn't put us below rho_at_pmin
      bad_vfrac_guess = false;
      for (std::size_t m = 0; m < nmat; ++m) {
        const Real rho_min = eos[m].RhoPmin(Tguess);
        [[maybe_unused]] const Real rho_max = eos[m].MaximumDensity();
        PORTABLE_REQUIRE(rho_min < rho_max, "Valid density range must exist!");
        PORTABLE_REQUIRE(rho[m] < rho_max, "Density must be less than rho_max");
        if (rho[m] < rho_min) {
          bad_vfrac_guess = true;
          Tguess *= Tfactor;
          break;
        }
      }
      if (!bad_vfrac_guess) break;
    }

    if (bad_vfrac_guess && params_.verbose) {
      PORTABLE_ALWAYS_WARN(
          "rho < rho_min in PTE initialization!  Solver may not converge.\n");
    }

    return Tguess;
  }

  template <typename EOS_t, typename Indexer_t>
  PORTABLE_FORCEINLINE_FUNCTION static Real
  GetPressureFromPreferred(const EOS_t &eos, const Real rho, const Real T, Real sie,
                           Indexer_t lambda, const bool do_e_lookup) {
    Real P{};
    if (eos.PreferredInput() & thermalqs::specific_internal_energy) {
      if (do_e_lookup) {
        sie = eos.InternalEnergyFromDensityTemperature(rho, T, lambda);
      }
      P = eos.PressureFromDensityInternalEnergy(rho, sie, lambda);
    } else { // if (eos.PreferredInput() & thermalqs::temperature) {
      P = eos.PressureFromDensityTemperature(rho, T, lambda);
    }
    return P;
  }

  // Uses temperature always. Sometimes uses P sometimes uses rho
  // depending on what the EOS wants.
  template <typename EOS_t, typename Indexer_t>
  PORTABLE_FORCEINLINE_FUNCTION static void
  GetSieCvFromTAndPreferred(const EOS_t &eos, Real rho, const Real P, const Real T,
                            Indexer_t lambda, Real &sie, Real &cv) {
    if (eos.PreferredInput() & thermalqs::pressure) {
      Real Pmin = eos.MinimumPressure();
      Real Pmax = eos.MaximumPressureAtTemperature(T);
      Real Pguess;
      if (error_utils::bad_value(P, "Pressure in GetSieCvFromTAndPreferred")) {
        Pguess = 0.5 * (Pmin + Pmax);
      } else {
        Pguess = std::max(Pmin, std::min(Pmax, P));
      }
      eos.DensityEnergyFromPressureTemperature(Pguess, T, lambda, rho, sie);
      // T is KNOWN here, so read cv directly at (rho, T).  The former
      // SpecificHeatFromDensityInternalEnergy(rho, sie) RE-INVERTED (rho,sie)->T (a ~40-iteration
      // bisection) just to recover the T we already have -- pure redundant work in the cyclic
      // Init's T-guess loop (~3 inversions/cell).  cv is identical (the round trip returns T).
      cv = eos.SpecificHeatFromDensityTemperature(rho, T, lambda);
    } else { // use density, not pressure.
      sie = eos.InternalEnergyFromDensityTemperature(rho, T, lambda);
      cv = eos.SpecificHeatFromDensityTemperature(rho, T, lambda);
    }
  }

  // Initialize the volume fractions, avg densities, temperatures, energies, and
  // pressures of the materials.  Compute the total density and internal energy.
  // Scale the energy densities u = rho sie and pressures with the total internal energy
  // density.  This makes Sum(u(mat)) = 1.  Also scale the temperature with the initial
  // guess, so all the initial, scaled material temperatures are = 1.
  PORTABLE_INLINE_FUNCTION
  void InitBase() {

    // intialize rhobar array and final density
    InitRhoBarandRho();
    const Real utotal = rho_total * sie_total;
    // TODO(): Consider more carefully edge case when utotal \simeq 0
    uscale = std::max(std::abs(utotal), 1.0e-14);
    utotal_scale = robust::ratio(utotal, uscale);

    // guess some non-zero temperature to start
    const Real Tguess = GetTguess();
    // set the temperature normalization
    Tnorm = Tguess;

    for (std::size_t m = 0; m < nmat; m++) {
      // scaled initial guess for temperature is just 1
      temp[m] = 1.0;
      sie[m] = eos[m].InternalEnergyFromDensityTemperature(rho[m], Tguess, lambda[m]);
      // note the scaling of pressure
      auto prefinput = eos[m].PreferredInput();
      if ((prefinput & thermalqs::pressure) &&
          !(error_utils::bad_value(press[m], "press[m]"))) {
        // Use input pressure for this material as the guess
        press[m] = robust::ratio(press[m], uscale);
      } else {
        press[m] = robust::ratio(this->GetPressureFromPreferred(eos[m], rho[m], Tguess,
                                                                sie[m], lambda[m], false),
                                 uscale);
      }
    }

    // note the scaling of the material internal energy densities
    for (std::size_t m = 0; m < nmat; ++m)
      u[m] = sie[m] * robust::ratio(rhobar[m], uscale);
  }

  PORTABLE_INLINE_FUNCTION
  Real ResidualNorm() const {
    return 0.5 * sum_neumaier(residual, neq, 0, -1, [](const Real x) { return x * x; });
  }

  PORTABLE_FORCEINLINE_FUNCTION
  std::size_t MatIndex(const std::size_t &i, const std::size_t &j) const {
    return i * neq + j;
  }

  // Compute the equilibrium pressure and temperature assuming an ideal EOS for each
  // material.  Set Pideal and Tideal to the ***scaled*** solution.
  PORTABLE_INLINE_FUNCTION
  void GetIdealPTE(Real &Pideal, Real &Tideal) const {
    Real rhoBsum = 0.0;
    Real Asum = 0.0;
    for (std::size_t m = 0; m < nmat; m++) {
      Asum += vfrac[m] * robust::ratio(press[m], temp[m]);
      rhoBsum += rho[m] * vfrac[m] * robust::ratio(sie[m], temp[m]);
    }
    Asum *= uscale / Tnorm;
    rhoBsum /= Tnorm;
    Tideal = uscale / rhoBsum / Tnorm;
    Pideal = robust::ratio(Tnorm * Tideal * Asum, uscale * vfrac_total);
  }

  // Compute the ideal EOS PTE solution and replace the initial guess if it has a lower
  // residual.
  template <typename T>
  PORTABLE_INLINE_FUNCTION void TryIdealPTE(T *solver) {
    Real Pideal, Tideal;
    GetIdealPTE(Pideal, Tideal);

    // temporarily hijack some of the scratch space
    Real *etemp = jacobian;
    Real *ptemp = jacobian + nmat;
    Real *vtemp = jacobian + 2 * nmat;
    Real *rtemp = jacobian + 3 * nmat;
    Real *res = jacobian + 4 * nmat;
    // copy out the initial guess
    for (std::size_t m = 0; m < nmat; ++m) {
      etemp[m] = u[m];
      ptemp[m] = press[m];
      vtemp[m] = vfrac[m];
      rtemp[m] = rho[m];
    }
    Real res_norm_old = 0.0;
    for (std::size_t m = 0; m < neq; ++m) {
      res[m] = residual[m];
      res_norm_old += res[m] * res[m];
    }
    // check if the volume fractions are reasonable
    const Real alpha = robust::ratio(Pideal, Tideal);
    for (std::size_t m = 0; m < nmat; ++m) {
      vfrac[m] *= robust::ratio(press[m], (temp[m] * alpha));
      if (Tnorm * Tideal < 0 ||
          robust::ratio(rhobar[m], vfrac[m]) < eos[m].RhoPmin(Tnorm * Tideal)) {
        // abort because this is putting this material into a bad state
        for (std::size_t n = 0; n <= m; n++)
          vfrac[n] = vtemp[n];
        return;
      }
    }
    // fill in the rest of the state
    for (std::size_t m = 0; m < nmat; ++m) {
      rho[m] = robust::ratio(rhobar[m], vfrac[m]);

      const Real sie_m =
          eos[m].InternalEnergyFromDensityTemperature(rho[m], Tnorm * Tideal, lambda[m]);
      u[m] = rhobar[m] * robust::ratio(sie_m, uscale);
      press[m] =
          robust::ratio(this->GetPressureFromPreferred(eos[m], rho[m], Tnorm * Tideal,
                                                       sie_m, lambda[m], false),
                        uscale);
    }
    // fill in the residual
    solver->Residual();
    Real res_norm_new = 0.0;
    for (std::size_t m = 0; m < neq; m++)
      res_norm_new += residual[m] * residual[m];

    if (res_norm_new > res_norm_old) {
      // this didn't work out so reset everything
      for (std::size_t m = 0; m < nmat; ++m) {
        vfrac[m] = vtemp[m];
        rho[m] = rtemp[m];
        u[m] = etemp[m];
        press[m] = ptemp[m];
      }
      for (std::size_t m = 0; m < neq; ++m) {
        residual[m] = res[m];
      }
    } else {
      // did work, fill in temp and energy density
      for (std::size_t m = 0; m < nmat; ++m) {
        temp[m] = Tideal;
        sie[m] = uscale * robust::ratio(u[m], rhobar[m]);
      }
    }
  }

  PORTABLE_INLINE_FUNCTION
  Real *AssignIncrement(Real *&scratch, const std::size_t size) const {
    Real *p = scratch;
    scratch += size;
    return p;
  }

  const MixParams params_;
  const std::size_t nmat, neq;
  std::size_t niter;
  const Real vfrac_total, sie_total;
  const EOSIndexer &eos;
  const RealIndexer &rho;
  const RealIndexer &vfrac;
  const RealIndexer &sie;
  const RealIndexer &temp;
  const RealIndexer &press;
  LambdaIndexer &lambda;
  Real *jacobian, *dx, *sol_scratch, *residual, *u, *rhobar;
  CacheAccessor Cache;
  Real rho_total, uscale, utotal_scale, Tnorm;
};

} // namespace mix_impl

template <typename EOSIndexer, typename RealIndexer, typename LambdaIndexer = NullIndexer>
PORTABLE_INLINE_FUNCTION Real ApproxTemperatureFromRhoMatU(
    const std::size_t nmat, EOSIndexer &&eos, const Real u_tot, RealIndexer &&rho,
    RealIndexer &&vfrac, const Real Tguess = 0.0,
    LambdaIndexer &&lambda = NullIndexer()) {
  // should these be passed in?
  constexpr Real minimum_temperature = 1.e-9;
  constexpr Real maximum_temperature = 1.e9;

  // given material microphysical densities, volume fractions, and a total internal energy
  // density (rho e => erg/cm^3), solve for the temperature that gives the right sum
  // of material energies.  this should only be used for a rough guess since it has a
  // hard coded and fairly loose tolerance
  auto ufunc = [&](const Real T) {
    Real usum = 0.0;
    for (std::size_t m = 0; m < nmat; m++) {
      usum += rho[m] * vfrac[m] *
              eos[m].InternalEnergyFromDensityTemperature(rho[m], T, lambda[m]);
    }
    return usum;
  };

  Real ulo = ufunc(minimum_temperature);
  if (u_tot < ulo) return minimum_temperature;
  Real uhi = ufunc(maximum_temperature);
  if (u_tot > uhi) return maximum_temperature;
  Real lTlo = FastMath::lg(minimum_temperature);
  Real lThi = FastMath::lg(maximum_temperature);
  if (Tguess > minimum_temperature && Tguess < maximum_temperature) {
    const Real ug = ufunc(Tguess);
    if (ug < u_tot) {
      lTlo = FastMath::lg(Tguess);
      ulo = ug;
    } else {
      lThi = FastMath::lg(Tguess);
      uhi = ug;
    }
  }
  std::size_t iter = 0;
  constexpr std::size_t max_iter = 10;
  while (lThi - lTlo > 0.01 && iter < max_iter) {
    // apply bisection which is much better behaved
    // for materials that have a flat sie at low temperatures
    const Real lT = 0.5 * (lTlo + lThi);
    const Real uT = ufunc(FastMath::pow2(lT));
    if (uT < u_tot) {
      lTlo = lT;
      ulo = uT;
    } else {
      lThi = lT;
      uhi = uT;
    }
    iter++;
  }

  const Real alpha = robust::ratio((u_tot - ulo), (uhi - ulo));
  return FastMath::pow2((1.0 - alpha) * lTlo + alpha * lThi);
}

// clang-format off
/*
 * Indep variables are: T, and volume fractions for materials 0 through nmat-1
 * excluding one material index s
 * Residual equations are:
 *   sum(u_m) - utotal = 0
 *   P_m - P_s = 0 for all m > 0, m != s
 * subject to the constraint
 *   alpha_s = 1 - sum_{m=0, m!=s)^{nmat - 1} alpha_m
 * where alpha is volume fraction
 *
 * Jacobian is
 *
 * [ sum(du/dT)         du_1/dalpha_1 - du_s/dalpha_s ... du_m/dalpha_m - du_s/dalpha_s ]
 * [ dP_1/dT - dP_s/dT  dP_1/dalpha_1 + dP_s/dalpha_s dP_s/dalpha_s    ...              ]
 * [ .                                                            dP_s/dalpha_s         ]
 * [ .                            dP_s/dalpha_s                         .               ]
 * [ dP_m/dT - dP_s/dT            dP_s/dalpha_s          dP_n/dalpha_n + dP_s/dalpha_s  ]
 *
 */
// clang-format on
// ======================================================================
// PTE Solver RhoT
// ======================================================================
constexpr inline int PTESolverRhoTRequiredScratch(const std::size_t nmat) {
  std::size_t neq = nmat;
  return neq * neq   // jacobian
         + 4 * neq   // dx, residual, and sol_scratch
         + 6 * nmat; // all the nmat sized arrays
}
constexpr inline size_t PTESolverRhoTRequiredScratchInBytes(const std::size_t nmat) {
  return PTESolverRhoTRequiredScratch(nmat) * sizeof(Real);
}

template <typename EOSIndexer, typename RealIndexer, typename LambdaIndexer>
class PTESolverRhoT
    : public mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer> {
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::InitBase;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::AssignIncrement;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::nmat;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::neq;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::ResidualNorm;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::vfrac_total;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::NormalizeVfrac;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::sie_total;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::eos;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::rho;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::vfrac;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::sie;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::temp;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::press;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::rho_total;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::uscale;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::utotal_scale;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::MatIndex;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::TryIdealPTE;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::jacobian;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::residual;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::dx;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::u;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::rhobar;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::Cache;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::Tnorm;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::params_;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::lambda;

 public:
  // template the ctor to get type deduction/universal references prior to c++17
  template <typename EOS_t, typename Real_t, typename Lambda_t>
  PORTABLE_INLINE_FUNCTION
  PTESolverRhoT(const std::size_t nmat, EOS_t &&eos, const Real vfrac_tot,
                const Real sie_tot, Real_t &&rho, Real_t &&vfrac, Real_t &&sie,
                Real_t &&temp, Real_t &&press, Lambda_t &&lambda, Real *scratch,
                const Real Tnorm = 0.0, const MixParams &params = MixParams())
      : mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>(
            nmat, nmat, eos, vfrac_tot, sie_tot, rho, vfrac, sie, temp, press, lambda,
            scratch, Tnorm, params) {
    dpdv = AssignIncrement(scratch, nmat);
    dedv = AssignIncrement(scratch, nmat);
    dpdT = AssignIncrement(scratch, nmat);
    vtemp = AssignIncrement(scratch, nmat);
  }

  static inline std::string MethodType() { return std::string("PTESolveRhoT"); }

  // Create static member functions bound to the class for ease of use
  constexpr static inline int RequiredScratch(const std::size_t nmat) {
    return PTESolverRhoTRequiredScratch(nmat);
  }
  constexpr static inline size_t RequiredScratchInBytes(const std::size_t nmat) {
    return PTESolverRhoTRequiredScratchInBytes(nmat);
  }

  PORTABLE_INLINE_FUNCTION
  Real Init() {
    InitBase();

    // Decide which material's volume fraction should be set by the
    // sum of the others. If an index is passed in through mix params,
    // we use that. Otherwise we do our best.
    //
    // We probably want to exclude the material with the largest
    // volume fraction, but we don't know which one that will be. We
    // guestimate it as the material with the largest mass fraction,
    // though this may not always be correct.
    //
    // At later iterations we will check the volume fractions again
    // and pick the material with the largest one.
    if (params_.pte_reduced_system_exclude_idx >= 0) {
      ms = params_.pte_reduced_system_exclude_idx;
    } else {
      ms = 0;
      Real xmax = 0;
      for (std::size_t m = 0; m < nmat; ++m) {
        Real x = robust::ratio(rhobar[m], rho_total);
        if (x > xmax) {
          xmax = x;
          ms = m;
        }
      }
    }
    PORTABLE_REQUIRE(ms < nmat, "Index of material whose volume fraction will depend on "
                                "the others must be less than nmat");

    Residual();

    // Set the current guess for the equilibrium temperature.  Note that this is already
    // scaled.
    Tequil = temp[0];

    // Compute minimum scaled temperature from EOS table bounds (constant
    // for the lifetime of this solver instance).
    Tequil_min = 0.0;
    for (std::size_t m = 0; m < nmat; ++m) {
      Tequil_min = std::max(Tequil_min,
                            robust::ratio(eos[m].MinimumTemperature(), Tnorm));
    }

    return ResidualNorm();
  }

  PORTABLE_INLINE_FUNCTION
  void Fixup() {
    // Volume fractions normalized by construction in this
    // formulation, so no reason to normalize them here the way
    // PTESolverBase does.
    //
    // Re-pick the material to exclude based on largest volume
    // fraction, now that perhaps the volume fractions have been
    // updated
    if (params_.pte_reduced_system_exclude_idx < 0) {
      Real alphamax = 0;
      ms = 0;
      for (std::size_t m = 0; m < nmat; ++m) {
        if (vfrac[m] > alphamax) {
          alphamax = vfrac[m];
          ms = m;
        }
      }
      // With the excluded material changed, the residual must be
      // recomputed
      Residual();
    }
  }

  PORTABLE_INLINE_FUNCTION
  void Residual() const {
    Real esum = mix_impl::sum_neumaier(u, nmat);
    residual[0] = utotal_scale - esum;
    std::size_t ires = 1;
    for (std::size_t m = 0; m < nmat; ++m) {
      if (m == ms) continue;
      residual[ires++] = press[ms] - press[m];
    }
  }

  PORTABLE_INLINE_FUNCTION
  auto CheckPTE() const {
    using namespace mix_impl;
    Real mean_p = 0;
    for (std::size_t m = 0; m < nmat; ++m) {
      mean_p += vfrac[m] * press[m];
    }
    Real error_p = std::sqrt(
        sum_neumaier(residual, neq - 1, 1, -1, [](const Real x) { return x * x; }));
    Real error_u = std::abs(residual[0]);
    // Check for convergence
    bool converged_p = (error_p < params_.pte_rel_tolerance_p * std::abs(mean_p) ||
                        error_p < params_.pte_abs_tolerance_p);
    bool converged_u = (error_u < params_.pte_rel_tolerance_e);
    bool acceptable_p =
        (error_p < params_.pte_rel_tolerance_p_sufficient * std::abs(mean_p) ||
         error_p < params_.pte_abs_tolerance_p_sufficient);
    bool acceptable_u = (error_u < params_.pte_rel_tolerance_e_sufficient);
    return std::make_pair(converged_p && converged_u, acceptable_p && acceptable_u);
  }

  PORTABLE_INLINE_FUNCTION
  void Jacobian() const {
    Real dedT_sum = 0.0;
    for (std::size_t m = 0; m < nmat; m++) {
      //////////////////////////////
      // perturb volume fractions
      //////////////////////////////
      Real dv = (vfrac[m] < 0.5 ? 1.0 : -1.0) * vfrac[m] * params_.derivative_eps;
      const Real vf_pert = vfrac[m] + dv;
      const Real rho_pert = robust::ratio(rhobar[m], vf_pert);

      Real e_pert = eos[m].InternalEnergyFromDensityTemperature(rho_pert, Tnorm * Tequil,
                                                                lambda[m]);
      Real p_pert =
          robust::ratio(this->GetPressureFromPreferred(eos[m], rho_pert, Tnorm * Tequil,
                                                       e_pert, lambda[m], false),
                        uscale);
      dpdv[m] = std::min(robust::ratio((p_pert - press[m]), dv),
                         -std::abs(params_.pte_max_dpdv));
      dedv[m] = robust::ratio(rhobar[m] * robust::ratio(e_pert, uscale) - u[m], dv);

      //////////////////////////////
      // perturb temperature
      //////////////////////////////
      // Use at least derivative_eps in scaled units to prevent vanishing
      // perturbation when Tequil < 1 (which would place both T and T+dT
      // below the EOS table floor, yielding zero derivatives).
      Real dT = std::max(Tequil, 1.0) * params_.derivative_eps;
      e_pert = eos[m].InternalEnergyFromDensityTemperature(rho[m], Tnorm * (Tequil + dT),
                                                           lambda[m]);
      p_pert = robust::ratio(this->GetPressureFromPreferred(eos[m], rho[m],
                                                            Tnorm * (Tequil + dT), e_pert,
                                                            lambda[m], false),
                             uscale);
      dpdT[m] = robust::ratio((p_pert - press[m]), dT);
      dedT_sum += robust::ratio(rhobar[m] * robust::ratio(e_pert, uscale) - u[m], dT);
    }
    // Enforce cv > 0: if EOS clamping made the finite-difference derivative
    // zero or negative, use a small positive floor to prevent singular Jacobian.
    dedT_sum = std::max(dedT_sum, params_.min_dtde);

    // Fill in the Jacobian
    for (std::size_t i = 0; i < neq * neq; ++i)
      jacobian[i] = 0.0;
    // first row is energy residual eqn
    jacobian[0] = dedT_sum;
    std::size_t col = 1;
    for (std::size_t m = 0; m < nmat; ++m) {
      if (m == ms) continue;
      jacobian[col++] = dedv[m] - dedv[ms];
    }
    // remaining rows are pressure residual
    std::size_t row = 1;
    for (std::size_t m = 0; m < nmat; m++) {
      if (m == ms) continue;
      // 0th column of rows 1+ is temp derivative
      jacobian[MatIndex(row, 0)] = dpdT[m] - dpdT[ms];
      // diagonal of J[1:,1:] is dpdvs.
      // + sign here is because dp_s/dalpha_i = -dp_s/dalpha_s
      // ALL off diagonal terms in J[1:,1:] are dpdv[s] because Ps
      // depends on ALL the independent volume fractions
      std::size_t col = 1;
      for (std::size_t mm = 0; mm < nmat; ++mm) {
        if (mm == ms) continue;
        jacobian[MatIndex(row, col)] = (mm == m) ? dpdv[m] + dpdv[ms] : dpdv[ms];
        col++;
      }
      row++;
    }
  }

  PORTABLE_INLINE_FUNCTION
  Real ScaleDx() const {
    // Each check reduces the scale further if necessary
    Real scale = 1.0;
    // control how big of a step toward vfrac = 0 is allowed
    // 0th material is special as delta volume fraction for it is
    // minus the sum of the deltas for the others
    Real dalphaskip = -mix_impl::sum_neumaier(dx, nmat - 1, 1);
    std::size_t idx = 1;
    for (std::size_t m = 0; m < nmat; ++m) {
      Real mydx = (m == ms) ? dalphaskip : dx[idx++];
      if (scale * mydx < -params_.vfrac_safety_fac * vfrac[m]) {
        scale = std::min(
            scale, std::abs(params_.vfrac_safety_fac * robust::ratio(vfrac[m], mydx)));
      }
    }

    const Real Tnew = Tequil + scale * dx[0];
    // control how big of a step toward rho = rho(Pmin) is allowed
    idx = 1;
    for (std::size_t m = 0; m < nmat; m++) {
      const Real rho_min =
          std::max(eos[m].RhoPmin(Tnorm * Tequil), eos[m].RhoPmin(Tnorm * Tnew));
      const Real alpha_max = std::min(1.0, robust::ratio(rhobar[m], rho_min));
      if (alpha_max < vfrac[m]) {
        // Despite our best efforts, we're already in the unstable regime (i.e.
        // dPdV_T > 0) so we would actually want to *increase* the step instead
        // of decreasing it. As a result, this code doesn't work as intended and
        // should be skipped.
        continue;
      }
      Real mydx = (m == ms) ? dalphaskip : dx[idx++];
      if (scale * mydx > 0.5 * (alpha_max - vfrac[m])) {
        scale = std::min(robust::ratio(0.5 * (alpha_max - vfrac[m]), mydx), scale);
      }
    }
    // Minimum allowed scaled temperature: max of the relative floor (5% of
    // current T) and the EOS table minimum (precomputed in Init()).
    const Real T_floor = std::max(0.05 * Tequil, Tequil_min);
    if (Tequil + scale * dx[0] < T_floor) {
      if (dx[0] < 0.0 && Tequil > T_floor) {
        // Limit step so we land exactly at the floor
        scale = std::min(scale, (Tequil - T_floor) / (-dx[0]));
      } else if (dx[0] < 0.0) {
        // Already at or below the floor with a downward step: zero it out
        // so stagnation detection fires promptly
        dx[0] = 0.0;
      }
    }
    // Now apply the overall scaling
    for (std::size_t i = 0; i < neq; ++i)
      dx[i] *= scale;
    return scale;
  }

  // Update the solution and return new residual.  Possibly called repeatedly with
  // different scale factors as part of a line search
  PORTABLE_INLINE_FUNCTION
  Real TestUpdate(const Real scale, bool const cache_state = false) {
    if (cache_state) {
      // Store the current state in temp variables for first iteration of line
      // search
      Ttemp = Tequil;
      for (std::size_t m = 0; m < nmat; ++m)
        vtemp[m] = vfrac[m];
    }
    Tequil = Ttemp + scale * dx[0];
    std::size_t idx = 1;
    for (std::size_t m = 0; m < nmat; ++m) {
      if (ms == m) continue;
      vfrac[m] = vtemp[m] + scale * dx[idx++];
    }
    vfrac[ms] = mix_impl::sum_neumaier(vfrac, nmat, 0, ms);
    vfrac[ms] = 1 - vfrac[ms];
    for (std::size_t m = 0; m < nmat; ++m) {
      rho[m] = robust::ratio(rhobar[m], vfrac[m]);
      u[m] = rhobar[m] * eos[m].InternalEnergyFromDensityTemperature(
                             rho[m], Tnorm * Tequil, lambda[m]);
      sie[m] = robust::ratio(u[m], rhobar[m]);
      u[m] = robust::ratio(u[m], uscale);
      temp[m] = Tequil;
      press[m] =
          robust::ratio(this->GetPressureFromPreferred(eos[m], rho[m], Tnorm * Tequil,
                                                       sie[m], lambda[m], false),
                        uscale);
    }
    Residual();

    return ResidualNorm();
  }

  PORTABLE_INLINE_FUNCTION
  Real GetPhysicalT() const { return Tnorm * Tequil; }

  // === DEBUG PRINTS (enabled with -DPTE_DEBUG_TRACE) ===
  PORTABLE_INLINE_FUNCTION
  void DebugPrintHeader() const {
#ifdef PTE_DEBUG_TRACE
    std::printf("=== PTE SOLVER ITERATION TRACE (PTESolverRhoT) ===\n");
    std::printf("  nmat=%zu  ms(excluded)=%zu  Tnorm=%.6e  uscale=%.6e\n", nmat, ms,
                Tnorm, uscale);
    std::printf("  %-5s %-12s %-12s %-5s %-5s", "iter", "err", "Tequil*Tnorm", "conv",
                "close");
    for (std::size_t m = 0; m < nmat; ++m)
      std::printf("  rho[%zu]       vfrac[%zu]     P[%zu]*uscale  ", m, m, m);
    std::printf("\n");
#endif
  }

  PORTABLE_INLINE_FUNCTION
  void DebugPrintState(std::size_t iter, Real err, bool converged,
                       bool close_enough) const {
#ifdef PTE_DEBUG_TRACE
    if (iter == 0) DebugPrintHeader();
    std::printf("  %-5zu %-12.6e %-12.6e %-5d %-5d", iter, err, Tequil * Tnorm,
                (int)converged, (int)close_enough);
    for (std::size_t m = 0; m < nmat; ++m)
      std::printf("  %-12.6e %-12.6e %-12.6e", rho[m], vfrac[m], press[m] * uscale);
    std::printf("\n");
    // Print residual vector
    std::printf("    residual=[");
    for (std::size_t i = 0; i < neq; ++i)
      std::printf("%.6e%s", residual[i], i < neq - 1 ? ", " : "");
    std::printf("]\n");
    // Print dx (Newton step)
    std::printf("    dx=[");
    for (std::size_t i = 0; i < neq; ++i)
      std::printf("%.6e%s", dx[i], i < neq - 1 ? ", " : "");
    std::printf("]\n");
    // Print dpdv and dpdT
    std::printf("    dpdv=[");
    for (std::size_t m = 0; m < nmat; ++m)
      std::printf("%.6e%s", dpdv[m], m < nmat - 1 ? ", " : "");
    std::printf("]  dpdT=[");
    for (std::size_t m = 0; m < nmat; ++m)
      std::printf("%.6e%s", dpdT[m], m < nmat - 1 ? ", " : "");
    std::printf("]\n");
    // Print Jacobian matrix
    std::printf("    Jacobian (%zux%zu):\n", neq, neq);
    for (std::size_t i = 0; i < neq; ++i) {
      std::printf("      [");
      for (std::size_t j = 0; j < neq; ++j)
        std::printf("%-14.6e", jacobian[i * neq + j]);
      std::printf("]\n");
    }
    std::fflush(stdout);
#endif
  }

  PORTABLE_INLINE_FUNCTION
  void DebugPrintStep(Real scale) const {
#ifdef PTE_DEBUG_TRACE
    std::printf("    ScaleDx -> scale=%.6e\n", scale);
    std::fflush(stdout);
#endif
  }
  // === END DEBUG PRINTS ===

  // Returns true when the equilibrium temperature is at or below the
  // EOS minimum temperature.  Used by the Newton loop to skip futile
  // pressure jumps when the solver is stuck at the temperature floor.
  PORTABLE_INLINE_FUNCTION
  bool IsAtTFloor() const {
    return Tequil <= Tequil_min * (1.0 + robust::EPS());
  }

 private:
  Real *dpdv, *dedv, *dpdT, *vtemp;
  Real Tequil, Ttemp, Tequil_min;
  std::size_t ms;
};

// ======================================================================
// PT space solver
// ======================================================================
constexpr inline int PTESolverPTRequiredScratch(const std::size_t nmat) {
  constexpr int neq = 2;
  return neq * neq   // jacobian
         + 4 * neq   // dx, residual, and sol_scratch
         + 2 * nmat; // all the nmat sized arrays
}
constexpr inline size_t PTESolverPTRequiredScratchInBytes(const std::size_t nmat) {
  return PTESolverPTRequiredScratch(nmat) * sizeof(Real);
}
template <typename EOSIndexer, typename RealIndexer, typename LambdaIndexer>
class PTESolverPT
    : public mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer> {
  // protected (not private) so subclasses (PTESolverPTAnalytic) inherit these base-member
  // handles and can override Jacobian()/etc.
 protected:
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::InitBase;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::AssignIncrement;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::nmat;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::neq;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::ResidualNorm;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::vfrac_total;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::sie_total;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::eos;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::rho;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::vfrac;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::sie;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::temp;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::press;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::rho_total;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::uscale;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::utotal_scale;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::MatIndex;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::TryIdealPTE;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::jacobian;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::residual;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::dx;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::u;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::rhobar;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::Cache;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::Tnorm;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::params_;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::lambda;

  enum RES { RV = 0, RSIE = 1 };

 public:
  // template the ctor to get type deduction/universal references prior to c++17
  template <typename EOS_t, typename Real_t, typename Lambda_t>
  PORTABLE_INLINE_FUNCTION
  PTESolverPT(const std::size_t nmat, EOS_t &&eos, const Real vfrac_tot,
              const Real sie_tot, Real_t &&rho, Real_t &&vfrac, Real_t &&sie,
              Real_t &&temp, Real_t &&press, Lambda_t &&lambda, Real *scratch,
              const Real Tnorm = 0.0, const MixParams &params = MixParams())
      : mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>(
            nmat, 2, eos, vfrac_tot, sie_tot, rho, vfrac, sie, temp, press, lambda,
            scratch, Tnorm, params) {}

  static inline std::string MethodType() { return std::string("PTESolverPT"); }

  // Create a static member function bound to the class for ease of use
  constexpr static inline int RequiredScratch(const std::size_t nmat) {
    return PTESolverPTRequiredScratch(nmat);
  }
  constexpr static inline size_t RequiredScratchInBytes(const std::size_t nmat) {
    return PTESolverPTRequiredScratchInBytes(nmat);
  }

  PORTABLE_INLINE_FUNCTION
  Real Init() {
    InitBase();
    Residual();

    // TODO(JMM): Suggestion from Jeff:
    // I suspect that you could multiply the bulk modulus by the
    // volume fraction to produce a weighting that wasn't a bad
    // guess. Worth keeping in mind for the future maybe if the
    // iteration count becomes an issue.
    Pequil = 0;
    for (std::size_t m = 0; m < nmat; ++m) {
      // always approach from >0 side
      Pequil += std::abs(press[m]) * vfrac[m];
    }
    Real vsum = mix_impl::sum_neumaier(vfrac, nmat);
    Pequil /= vsum;
    Tequil = 1; // Because it's = Tnorm = initial guess

    // Set the state based on the P/T chosen
    for (std::size_t m = 0; m < nmat; ++m) {
      eos[m].DensityEnergyFromPressureTemperature(Pequil * uscale, Tequil * Tnorm,
                                                  lambda[m], rho[m], sie[m]);
      vfrac[m] = robust::ratio(rhobar[m], rho[m]);
      u[m] = robust::ratio(sie[m] * rhobar[m], uscale);
      temp[m] = Tequil;
    }
    Residual();

    // Set the current guess for the equilibrium temperature.  Note
    // that this is already scaled.
    return ResidualNorm();
  }

  PORTABLE_INLINE_FUNCTION
  void Finalize() {
    for (std::size_t m = 0; m < nmat; m++) {
      if (!variadic_utils::is_nullptr(press)) {
        press[m] = Pequil * uscale;
      }
      if (!variadic_utils::is_nullptr(temp)) {
        temp[m] = Tequil * Tnorm;
      }
      if (!variadic_utils::is_nullptr(u)) {
        u[m] *= uscale;
      }
    }
  }

  PORTABLE_INLINE_FUNCTION
  void Residual() const {
    Real vsum = mix_impl::sum_neumaier(vfrac, nmat);
    Real esum = mix_impl::sum_neumaier(u, nmat);
    residual[RV] = vfrac_total - vsum;
    residual[RSIE] = utotal_scale - esum;
  }

  PORTABLE_INLINE_FUNCTION
  auto CheckPTE() const {
    Real error_v = std::abs(residual[RV]);
    Real error_u = std::abs(residual[RSIE]);
    // this may not be quite right as we may need an energy scaling factor
    // Check for convergence
    bool converged_u = error_u < params_.pte_rel_tolerance_e;
    bool converged_v = error_v < params_.pte_rel_tolerance_v;
    bool acceptable_u = error_u < params_.pte_rel_tolerance_e_sufficient;
    bool acceptable_v = error_v < params_.pte_rel_tolerance_v_sufficient;

    return std::make_pair(converged_v && converged_u, acceptable_u && acceptable_v);
  }

  PORTABLE_INLINE_FUNCTION
  void Jacobian() const {
    // sum_m d u_m / dT )_P
    Real dudT_P_sum = 0.0;
    // sum_m d u_m / dP )_T
    Real dudP_T_sum = 0.0;
    // - sum_m rhobar / rho_m^2 * d rho_m / dT )_P
    Real rbor2_dr_dT_P_sum = 0.0;
    // - sum_m rhobar / rho_m^2 d rho_m / dP )_T
    Real rbor2_dr_dP_T_sum = 0.0;

    // JMM: Note rescaling for u, P, T

    // TODO(JMM): Should we use the thermodynamic derivatives rather
    // than finite differences?
    for (std::size_t m = 0; m < nmat; ++m) {
      Real r_pert = rho[m]; // provide initial guesses
      Real e_pert = sie[m];

      //////////////////////////////
      // perturb pressures
      //////////////////////////////
      Real dp = -Pequil * params_.derivative_eps; // always move towards phase transition
      eos[m].DensityEnergyFromPressureTemperature(uscale * (Pequil + dp), Tnorm * Tequil,
                                                  lambda[m], r_pert, e_pert);
      Real drdp = robust::ratio(r_pert - rho[m], dp);
      Real dudp = robust::ratio(robust::ratio(rhobar[m] * e_pert, uscale) - u[m], dp);

      rbor2_dr_dP_T_sum += robust::ratio(rhobar[m], rho[m] * rho[m]) * drdp;
      dudP_T_sum += dudp;

      //////////////////////////////
      // perturb temperatures
      //////////////////////////////
      Real dT = std::max(Tequil, 1.0) * params_.derivative_eps;
      eos[m].DensityEnergyFromPressureTemperature(uscale * Pequil, Tnorm * (Tequil + dT),
                                                  lambda[m], r_pert, e_pert);
      Real drdT = robust::ratio(r_pert - rho[m], dT);
      Real dudT = robust::ratio(robust::ratio(rhobar[m] * e_pert, uscale) - u[m], dT);

      rbor2_dr_dT_P_sum += robust::ratio(rhobar[m], rho[m] * rho[m]) * drdT;
      dudT_P_sum += dudT;
    }

    // Fill in the Jacobian
    jacobian[0] = -rbor2_dr_dT_P_sum; // TODO(JMM): Check positions
    jacobian[1] = -rbor2_dr_dP_T_sum;
    jacobian[2] = dudT_P_sum;
    jacobian[3] = dudP_T_sum;
  }

  PORTABLE_INLINE_FUNCTION
  Real ScaleDx() const {
    Real scale = 1.0;
    if (scale * dx[0] < -0.95 * Tequil) {
      scale = robust::ratio(-0.95 * Tequil, dx[0]);
    }
    auto bounded = [=](Real Pbound, Real delta) {
      return robust::ratio(robust::ratio(Pbound, uscale) - Pequil, delta);
    };

    for (std::size_t m = 0; m < nmat; ++m) {
      Real Pmin = eos[m].MinimumPressure();
      scale = std::min(std::abs(scale), std::abs(0.95 * bounded(Pmin, dx[1])));
    }
    for (std::size_t m = 0; m < nmat; ++m) {
      Real Ttest = (Tequil + scale * dx[0]) * Tnorm;
      Real Pmax = eos[m].MaximumPressureAtTemperature(Ttest);
      scale = std::min(std::abs(scale), std::abs(0.95 * bounded(Pmax, dx[0])));
    }

    for (std::size_t i = 0; i < neq; ++i) {
      dx[i] *= scale;
    }
    return scale;
  }

  // Update the solution and return new residual.  Possibly called repeatedly with
  // different scale factors as part of a line search
  PORTABLE_INLINE_FUNCTION
  Real TestUpdate(const Real scale, bool const cache_state = false) {
    if (cache_state) {
      // Store the current state in temp variables for first iteration of line
      // search
      Ttemp = Tequil;
      Ptemp = Pequil;
    }
    Tequil = Ttemp + scale * dx[0];
    Pequil = Ptemp + scale * dx[1];
    for (std::size_t m = 0; m < nmat; ++m) {
      eos[m].DensityEnergyFromPressureTemperature(Pequil * uscale, Tequil * Tnorm,
                                                  lambda[m], rho[m], sie[m]);
      vfrac[m] = robust::ratio(rhobar[m], rho[m]);
      u[m] = robust::ratio(sie[m] * rhobar[m], uscale);
      temp[m] = Tequil;
      press[m] = Pequil;
    }
    Residual();
    return ResidualNorm();
  }

  PORTABLE_INLINE_FUNCTION
  Real GetPhysicalT() const { return Tnorm * Tequil; }
  // Physical equilibrium pressure (scaled Pequil * uscale) -- for the cyclic-solver
  // per-iteration diagnostic in PTESolverPTCyclicSolve.
  PORTABLE_INLINE_FUNCTION
  Real GetPhysicalP() const { return Pequil * uscale; }

  // No-op debug stubs (only PTESolverRhoT has real prints)
  PORTABLE_INLINE_FUNCTION
  void DebugPrintState(std::size_t, Real, bool, bool) const {}
  PORTABLE_INLINE_FUNCTION
  void DebugPrintStep(Real) const {}
  PORTABLE_INLINE_FUNCTION
  bool IsAtTFloor() const { return false; }

 protected:
  // TODO(JMM): Should these have trailing underscores?
  // Current P, T state. Protected (not private) so PTESolverPTAnalytic can override
  // Jacobian() using the equilibrium (P,T) directly.
  Real Pequil;
  Real Tequil;
  // Scratch states for test update
  Real Ptemp;
  Real Ttemp;
  // TODO(JMM): Should there be a P norm as well as a Tnorm?
};

// ======================================================================
// P-T solver with an ANALYTIC Jacobian (fast path / fallback for the cyclic solver)
// ======================================================================
// Identical to PTESolverPT except Jacobian() uses the EOS's exact (P,T) partials
// (DensityEnergyDerivativesFromPressureTemperature, e.g. TableDependsPT) instead of a
// finite difference. Because those partials are the exact derivatives of the same
// DensityEnergyFromPressureTemperature the residual is built from, the Newton step is
// consistent, and we avoid the base solver's FD stencil that deliberately perturbs P
// toward a phase transition (a source of noise near transitions). Requires an EOS that
// provides the derivatives method; use only on (P,T)-table mixtures (TableDependsPT).
template <typename EOSIndexer, typename RealIndexer, typename LambdaIndexer>
class PTESolverPTAnalytic
    : public PTESolverPT<EOSIndexer, RealIndexer, LambdaIndexer> {
  using Base = PTESolverPT<EOSIndexer, RealIndexer, LambdaIndexer>;
  // Base-member handles (nmat, eos, rho, rhobar, uscale, Tnorm, lambda, jacobian,
  // Pequil, Tequil) are inherited protected from PTESolverPT / PTESolverBase.
  using Base::eos;
  using Base::jacobian;
  using Base::lambda;
  using Base::nmat;
  using Base::Pequil;
  using Base::rho;
  using Base::rhobar;
  using Base::Tequil;
  using Base::Tnorm;
  using Base::uscale;

 public:
  template <typename EOS_t, typename Real_t, typename Lambda_t>
  PORTABLE_INLINE_FUNCTION
  PTESolverPTAnalytic(const std::size_t nmat, EOS_t &&eos, const Real vfrac_tot,
                      const Real sie_tot, Real_t &&rho, Real_t &&vfrac, Real_t &&sie,
                      Real_t &&temp, Real_t &&press, Lambda_t &&lambda, Real *scratch,
                      const Real Tnorm = 0.0, const MixParams &params = MixParams())
      : Base(nmat, std::forward<EOS_t>(eos), vfrac_tot, sie_tot, std::forward<Real_t>(rho),
             std::forward<Real_t>(vfrac), std::forward<Real_t>(sie),
             std::forward<Real_t>(temp), std::forward<Real_t>(press),
             std::forward<Lambda_t>(lambda), scratch, Tnorm, params) {}

  static inline std::string MethodType() { return std::string("PTESolverPTAnalytic"); }

  PORTABLE_INLINE_FUNCTION
  void Jacobian() const {
    Real dudT_P_sum = 0.0, dudP_T_sum = 0.0;
    Real rbor2_dr_dT_P_sum = 0.0, rbor2_dr_dP_T_sum = 0.0;
    for (std::size_t m = 0; m < nmat; ++m) {
      Real r, e, drho_dP, drho_dT, de_dP, de_dT;
      eos[m].DensityEnergyDerivativesFromPressureTemperature(
          uscale * Pequil, Tnorm * Tequil, lambda[m], r, e, drho_dP, drho_dT, de_dP, de_dT);
      // Rescale to the solver's nondimensional (Pequil, Tequil) variables, matching the
      // finite-difference convention in PTESolverPT::Jacobian: physical P = uscale*Pequil,
      // physical T = Tnorm*Tequil, and u_m = rhobar_m * sie_m / uscale.
      const Real drdp = drho_dP * uscale;                 // d rho_m / dPequil
      const Real drdT = drho_dT * Tnorm;                  // d rho_m / dTequil
      const Real dudp = rhobar[m] * de_dP;                // d u_m / dPequil
      const Real dudT = rhobar[m] * (Tnorm / uscale) * de_dT; // d u_m / dTequil
      const Real rbor2 = robust::ratio(rhobar[m], rho[m] * rho[m]);
      rbor2_dr_dP_T_sum += rbor2 * drdp;
      rbor2_dr_dT_P_sum += rbor2 * drdT;
      dudP_T_sum += dudp;
      dudT_P_sum += dudT;
    }
    jacobian[0] = -rbor2_dr_dT_P_sum;
    jacobian[1] = -rbor2_dr_dP_T_sum;
    jacobian[2] = dudT_P_sum;
    jacobian[3] = dudP_T_sum;
  }
};

// ======================================================================
// Cyclic P-T solver (Clayton-McConnell-Solomon Alg 5.2) -- PRIMARY PTE method
// ======================================================================
// The cyclic method is NOT a 2x2 Newton (that is PTESolverPT): each iteration takes a
// 1-D Newton step in P on the density residual varrho = 1/tau - 1/tau0, a 1-D Newton step
// in T on the enthalpy residual h (d_T h = c_p > 0), then intersects the tau-level-set
// tangent (slope (dP/dT)_varrho) with the isentrope tangent (slope (dT/dP)_s) for the next
// (P,T). It converges from far seeds where the 2-D Newton fails (Theorem 5.2), given a
// thermodynamically admissible mixture (each EOS MP-stable -> unique root, Theorem 2.26).
// It therefore uses a dedicated loop (PTESolverPTCyclicSolve) instead of the generic
// PTESolver Newton+line-search driver (the cyclic step is not a descent direction on the
// residual norm, so a line search would fight it). Ports pfc _pte.py::_cyclic_step exactly.
// Requires an EOS providing DensityEnergyDerivativesFromPressureTemperature (TableDependsPT).
template <typename EOSIndexer, typename RealIndexer, typename LambdaIndexer>
class PTESolverPTCyclic
    : public PTESolverPT<EOSIndexer, RealIndexer, LambdaIndexer> {
  using Base = PTESolverPT<EOSIndexer, RealIndexer, LambdaIndexer>;
  using Base::eos;
  using Base::lambda;
  using Base::nmat;
  using Base::params_;
  using Base::Pequil;
  using Base::press;
  using Base::rho;
  using Base::rho_total;
  using Base::rhobar;
  using Base::sie;
  using Base::sie_total;
  using Base::temp;
  using Base::Tequil;
  using Base::Tnorm;
  using Base::u;
  using Base::uscale;
  using Base::vfrac;

  struct Mix {
    Real tau, e, dtau_dp, dtau_dt, de_dp, de_dt;
  };

  // Mass-fraction-weighted mixture tau, e and (P,T) partials at physical (P,T).
  PORTABLE_INLINE_FUNCTION Mix mixture_(const Real P, const Real T) const {
    Mix mx{0., 0., 0., 0., 0., 0.};
    for (std::size_t m = 0; m < nmat; ++m) {
      Real rho_m, sie_m, drho_dP, drho_dT, de_dP, de_dT;
      eos[m].DensityEnergyDerivativesFromPressureTemperature(
          P, T, lambda[m], rho_m, sie_m, drho_dP, drho_dT, de_dP, de_dT);
      const Real Y = robust::ratio(rhobar[m], rho_total);
      const Real tau_m = robust::ratio(1.0, rho_m);
      const Real inv_rho2 = robust::ratio(1.0, rho_m * rho_m);
      mx.tau += Y * tau_m;
      mx.e += Y * sie_m;
      mx.dtau_dp += Y * (-drho_dP * inv_rho2); // dtau/dP = -(drho/dP)/rho^2
      mx.dtau_dt += Y * (-drho_dT * inv_rho2);
      mx.de_dp += Y * de_dP;
      mx.de_dt += Y * de_dT;
    }
    return mx;
  }

  // Intersect line A: P = pA + a (T - tA) with line B: T = tB + b (P - pB).
  PORTABLE_INLINE_FUNCTION bool intersect_(const Real pA, const Real tA, const Real a,
                                           const Real pB, const Real tB, const Real b,
                                           Real &P, Real &T) const {
    const Real ab = a * b;
    if (std::abs(1.0 - ab) < 1.0e-300 || !std::isfinite(ab)) return false;
    T = (tB + b * (pA - pB) - ab * tA) / (1.0 - ab);
    P = pA + a * (T - tA);
    return std::isfinite(P) && std::isfinite(T);
  }

 public:
  template <typename EOS_t, typename Real_t, typename Lambda_t>
  PORTABLE_INLINE_FUNCTION
  PTESolverPTCyclic(const std::size_t nmat, EOS_t &&eos, const Real vfrac_tot,
                    const Real sie_tot, Real_t &&rho, Real_t &&vfrac, Real_t &&sie,
                    Real_t &&temp, Real_t &&press, Lambda_t &&lambda, Real *scratch,
                    const Real Tnorm = 0.0, const MixParams &params = MixParams())
      : Base(nmat, std::forward<EOS_t>(eos), vfrac_tot, sie_tot, std::forward<Real_t>(rho),
             std::forward<Real_t>(vfrac), std::forward<Real_t>(sie),
             std::forward<Real_t>(temp), std::forward<Real_t>(press),
             std::forward<Lambda_t>(lambda), scratch, Tnorm, params) {}

  static inline std::string MethodType() { return std::string("PTESolverPTCyclic"); }

  // Lean (P,T)-native initialization for the cyclic solver.
  //
  // PTESolverPT::Init() (inherited) routes through PTESolverBase::InitBase(),
  // which for EVERY material does
  //     sie[m] = eos[m].InternalEnergyFromDensityTemperature(rho[m], Tguess),
  // and gets its temperature guess from GetTguess() -> a Newton step that calls
  // SpecificHeatFromDensityInternalEnergy(rho, sie).  On a (P,T)-indexed
  // TableDependsPT both are (rho,T)/(rho,sie) inversions: the first bisects
  // pressureOfRhoT_ ~100x; the second nests TemperatureFromDensityInternalEnergy
  // (~100x) inside SpecificHeatFromDensityTemperature (~100x) for ~1e4 native
  // lookups per material.  For the cyclic solver ALL of that work is discarded --
  // Init() immediately overwrites rho/sie/vfrac/u from
  // DensityEnergyFromPressureTemperature(Pequil, Tnorm).  Since TableDependsPT is
  // pressure-preferred and the caller supplies a positive P seed in press[], we
  // seed directly in the native (P,T) direction: one
  // DensityEnergyFromPressureTemperature per material, no (rho,T) inversion
  // anywhere.  This mirrors pfc _pte.py::solve_pte, which seeds from
  // (pressure_guess, temperature_guess) with no init inversion.  Isolated to the
  // cyclic solver: InitBase()/GetTguess() and every other PTE solver are unchanged.
  PORTABLE_INLINE_FUNCTION
  Real Init() {
    // rhobar[m] and rho_total (fixed mixture quantities).
    this->InitRhoBarandRho();
    // Energy normalization (as in InitBase()): make Sum(u_m) ~ 1.
    const Real utotal = rho_total * sie_total;
    uscale = std::max(std::abs(utotal), 1.0e-14);
    this->utotal_scale = robust::ratio(utotal, uscale);

    // Temperature normalization.  Use the caller's guess (stored in Tnorm) when
    // positive, else the largest per-material temperature, else the default --
    // the same fallback ladder as GetTguess() minus its Newton refinement (the
    // first CyclicStep does a T-Newton step on the enthalpy residual anyway).
    Real Tguess = (Tnorm > 0.0) ? Tnorm : params_.default_tguess;
    for (std::size_t m = 0; m < nmat; ++m) {
      Tguess = std::max(Tguess, temp[m]);
      Tguess = std::max(eos[m].MinimumTemperature(), Tguess);
    }
    PORTABLE_REQUIRE(Tguess > 0., "Non-positive temperature guess for PTE");
    PORTABLE_REQUIRE(Tguess < params_.temperature_limit,
                     "Very large input temperature or temperature guess");

    // Equilibrium-pressure seed: vfrac-weighted |press[m]| (identical to
    // PTESolverPT::Init()).  The caller seeds a single common mixture-match
    // pressure, so this reduces to that pressure.
    Real Pseed = 0.0;
    Real vsum = 0.0;
    for (std::size_t m = 0; m < nmat; ++m) {
      Pseed += std::abs(press[m]) * vfrac[m];
      vsum += vfrac[m];
    }
    Pseed = robust::ratio(Pseed, vsum); // physical

    // Energy-based T refinement -- the Newton step GetTguess() takes, which this lean cyclic
    // Init previously skipped (assuming the first CyclicStep would climb T).  That assumption
    // fails for a cold cell whose STORED energy implies a much hotter ion equilibrium (e.g. the
    // fuel's large ion-energy reference): seeded at the raw cell T, the first CyclicStep's
    // T-Newton overshoots DOWN to the T-floor (Tmin) and stalls -- reproduced offline (pfc
    // _pte.py collapses identically from a cold Tguess, and converges from a warm one).  Take up
    // to 3 Newton steps on the energy residual at Pseed, accepting ONLY increases (GetTguess
    // convention), so we seed near the hot energy-consistent root.  Cheap: 3 * nmat EOS evals.
    if (params_.iterate_t_guess) {
      // Accept-only-increases Newton on the energy residual (GetTguess convention;
      // utotal = rho_total*sie_total is the target total energy computed above). Self-limiting
      // for a GOOD (already-warm) guess: the first step that does not raise Tguess breaks the
      // loop, so a warm / warm-started cell pays ~one nmat-eval round, not three -- no meaningful
      // slowdown when the initial guess is already near the equilibrium; the extra work is spent
      // only where it is needed (a cold guess that must climb to a hot root).
      for (int it = 0; it < 3; ++it) {
        Real usum = 0.0, dudt = 0.0;
        for (std::size_t m = 0; m < nmat; ++m) {
          Real sie_m, cv_m;
          this->GetSieCvFromTAndPreferred(eos[m], std::min(rho[m], eos[m].MaximumDensity()),
                                          Pseed, Tguess, lambda[m], sie_m, cv_m);
          usum += rhobar[m] * sie_m;
          dudt += rhobar[m] * cv_m;
        }
        const Real Tnew = Tguess - robust::ratio(usum - utotal, dudt);
        if (!(Tnew > Tguess)) break; // good guess (no increase) -> stop; avoid wasted steps
        Tguess = std::min(params_.temperature_limit, Tnew);
      }
      for (std::size_t m = 0; m < nmat; ++m)
        Tguess = std::max(eos[m].MinimumTemperature(), Tguess);
    }

    Tnorm = Tguess;
    Pequil = robust::ratio(Pseed, uscale); // scaled
    Tequil = 1.0;                          // == Tguess / Tnorm

    // One native (P,T) lookup per material fixes (rho, sie); vfrac/u/temp/press
    // follow.  No InternalEnergyFromDensityTemperature / pressureOfRhoT_ bisection.
    for (std::size_t m = 0; m < nmat; ++m) {
      eos[m].DensityEnergyFromPressureTemperature(Pseed, Tguess, lambda[m], rho[m],
                                                  sie[m]);
      vfrac[m] = robust::ratio(rhobar[m], rho[m]);
      u[m] = robust::ratio(sie[m] * rhobar[m], uscale);
      temp[m] = Tequil;
      press[m] = Pequil;
    }
    this->Residual();
    return this->ResidualNorm();
  }

  // One cyclic iteration: update (Pequil, Tequil), re-evaluate per-material state and the
  // mixture residual. Returns the new residual norm. (Ports _pte.py::_cyclic_step.)
  PORTABLE_INLINE_FUNCTION
  Real CyclicStep() {
    const Real tau0 = robust::ratio(1.0, rho_total); // mixture specific volume target
    const Real e0 = sie_total;
    const Real P = Pequil * uscale, T = Tequil * Tnorm; // physical
    const Mix mix = mixture_(P, T);

    // Step 1: Newton step in P on the density residual varrho = 1/tau - 1/tau0.
    const Real varrho = robust::ratio(1.0, mix.tau) - robust::ratio(1.0, tau0);
    const Real dvarrho_dp = -mix.dtau_dp / (mix.tau * mix.tau);
    Real p_tilde = (dvarrho_dp == 0.0) ? P : P - varrho / dvarrho_dp;

    // Step 2: Newton step in T on the enthalpy residual h at (p_tilde, T); d_T h = c_p.
    const Mix mix_p = mixture_(p_tilde, T);
    const Real enth = (mix_p.e + p_tilde * mix_p.tau) - (e0 + p_tilde * tau0);
    const Real c_p = mix_p.de_dt + p_tilde * mix_p.dtau_dt;
    const Real t_tilde = (c_p == 0.0) ? T : T - enth / c_p;

    // Step 3: intersect the tau-level-set tangent at (p_tilde, T) with the isentrope
    // tangent at (p_tilde, t_tilde); fall back to the enthalpy-curve tangent.
    const Real slope_a =
        (mix_p.dtau_dp == 0.0) ? 0.0 : -mix_p.dtau_dt / mix_p.dtau_dp; // (dP/dT)_varrho
    const Mix mix_pt = mixture_(p_tilde, t_tilde);
    const Real c_p_pt = mix_pt.de_dt + p_tilde * mix_pt.dtau_dt;
    const Real slope_b_s =
        (c_p_pt == 0.0) ? 0.0 : t_tilde * mix_pt.dtau_dt / c_p_pt; // (dT/dP)_s
    Real Pn, Tn;
    bool ok = intersect_(p_tilde, T, slope_a, p_tilde, t_tilde, slope_b_s, Pn, Tn);
    if (!ok || !InBox_(Pn, Tn)) {
      const Real slope_b_h =
          (c_p_pt == 0.0) ? 0.0 : (t_tilde * mix_pt.dtau_dt - mix_pt.tau) / c_p_pt;
      Real Pa, Ta;
      if (intersect_(p_tilde, T, slope_a, p_tilde, t_tilde, slope_b_h, Pa, Ta)) {
        Pn = Pa;
        Tn = Ta;
        ok = true;
      }
    }
    if (!ok) {
      Pn = p_tilde;
      Tn = t_tilde;
    }
    ClampToBox_(Pn, Tn);

    // Commit the new state (physical -> scaled) and refresh per-material quantities.
    Pequil = robust::ratio(Pn, uscale);
    Tequil = robust::ratio(Tn, Tnorm);
    for (std::size_t m = 0; m < nmat; ++m) {
      eos[m].DensityEnergyFromPressureTemperature(Pn, Tn, lambda[m], rho[m], sie[m]);
      vfrac[m] = robust::ratio(rhobar[m], rho[m]);
      u[m] = robust::ratio(sie[m] * rhobar[m], uscale);
      temp[m] = Tequil;
      press[m] = Pequil;
    }
    this->Residual();
    return this->ResidualNorm();
  }

  // Aggregate EOS-domain box over all materials (tightest common (P,T) rectangle).
  PORTABLE_INLINE_FUNCTION void Box_(Real &Plo, Real &Phi, Real &Tlo, Real &Thi,
                                     const Real Tref) const {
    Plo = eos[0].MinimumPressure();
    Phi = eos[0].MaximumPressureAtTemperature(Tref);
    Tlo = eos[0].MinimumTemperature();
    // Upper T bound: the solver's temperature_limit (the EOS variant does not forward a
    // MaximumTemperature accessor; the lower T/P bounds are what guard the failure modes).
    Thi = params_.temperature_limit;
    for (std::size_t m = 1; m < nmat; ++m) {
      Plo = std::max(Plo, eos[m].MinimumPressure());
      Phi = std::min(Phi, eos[m].MaximumPressureAtTemperature(Tref));
      Tlo = std::max(Tlo, eos[m].MinimumTemperature());
    }
  }
  PORTABLE_INLINE_FUNCTION bool InBox_(const Real P, const Real T) const {
    Real Plo, Phi, Tlo, Thi;
    Box_(Plo, Phi, Tlo, Thi, T);
    return P >= Plo && P <= Phi && T >= Tlo && T <= Thi;
  }
  PORTABLE_INLINE_FUNCTION void ClampToBox_(Real &P, Real &T) const {
    Real Plo, Phi, Tlo, Thi;
    Box_(Plo, Phi, Tlo, Thi, T);
    P = std::min(std::max(P, Plo), Phi);
    T = std::min(std::max(T, Tlo), Thi);
  }
};

// Dedicated driver for the cyclic P-T solver: iterate the cyclic step to convergence.
// (Mirrors pfc _pte.py::solve_pte; distinct from the Newton PTESolver above because the
// cyclic step is not a line-search descent direction.)
template <class System>
PORTABLE_INLINE_FUNCTION SolverStatus PTESolverPTCyclicSolve(System &s) {
  SolverStatus status;
  Real &err = status.residual;
  err = s.Init();
  const MixParams &params = s.GetParams();
  auto &niter = s.Niter();
  const std::size_t pte_max_iter = s.Nmat() * params.pte_max_iter_per_mat;
  status.converged = false;
  // Per-iteration diagnostic (env PTE_CYCLIC_DEBUG): trace (P,T,residual) so a stall can be
  // compared to the offline pfc _pte.py::solve_pte trajectory (does T climb to equilibrium?).
  // CPU-only; checked once. Prints seed + every 20th iter + the final state, per solve.
  static const bool cyc_dbg = (std::getenv("PTE_CYCLIC_DEBUG") != nullptr);
  if (cyc_dbg) {
    std::printf("[CYC] seed  P=%.6g T=%.6g r=%.4g  (pte_max_iter=%zu)\n", s.GetPhysicalP(),
                s.GetPhysicalT(), err, pte_max_iter);
  }
  for (niter = 0; niter < pte_max_iter; ++niter) {
    status.max_niter = std::max(status.max_niter, niter);
    auto check = s.CheckPTE();
    status.converged = check.first;
    if (status.converged) break;
    err = s.CyclicStep();
    if (cyc_dbg && (niter % 20 == 0)) {
      std::printf("[CYC] n=%zu P=%.6g T=%.6g r=%.4g\n", niter, s.GetPhysicalP(),
                  s.GetPhysicalT(), err);
    }
  }
  if (cyc_dbg) {
    std::printf("[CYC] end   conv=%d niter=%zu P=%.6g T=%.6g r=%.4g\n",
                static_cast<int>(status.converged), niter, s.GetPhysicalP(), s.GetPhysicalT(),
                err);
  }
  s.Finalize();
  return status;
}

// ======================================================================
// fixed temperature solver
// ======================================================================
constexpr inline std::size_t PTESolverFixedTRequiredScratch(const std::size_t nmat) {
  std::size_t neq = nmat;
  return neq * neq   // jacobian
         + 4 * neq   // dx, residual, and sol_scratch
         + 2 * nmat  // rhobar and u in base
         + 2 * nmat; // nmat sized arrays in fixed T solver
}
constexpr inline size_t PTESolverFixedTRequiredScratchInBytes(const std::size_t nmat) {
  return PTESolverFixedTRequiredScratch(nmat) * sizeof(Real);
}

template <typename EOSIndexer, typename RealIndexer, typename LambdaIndexer>
class PTESolverFixedT
    : public mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer> {
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::AssignIncrement;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::nmat;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::neq;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::ResidualNorm;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::vfrac_total;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::eos;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::rho;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::vfrac;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::sie;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::temp;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::press;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::rho_total;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::uscale;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::MatIndex;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::jacobian;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::residual;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::dx;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::u;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::rhobar;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::Cache;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::Tnorm;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::params_;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::lambda;

 public:
  // template the ctor to get type deduction/universal references prior to c++17
  // allow the type of the temperature array to be different, potentially a const Real*
  template <typename EOS_t, typename Real_t, typename CReal_t, typename Lambda_t>
  PORTABLE_INLINE_FUNCTION
  PTESolverFixedT(const std::size_t nmat, EOS_t &&eos, const Real vfrac_tot,
                  const Real T_true, Real_t &&rho, Real_t &&vfrac, Real_t &&sie,
                  CReal_t &&temp, Real_t &&press, Lambda_t &lambda, Real *scratch,
                  const MixParams &params = MixParams())
      : mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>(
            nmat, nmat, eos, vfrac_tot, 1.0, rho, vfrac, sie, temp, press, lambda,
            scratch, T_true, params) {
    dpdv = AssignIncrement(scratch, nmat);
    vtemp = AssignIncrement(scratch, nmat);
    Tequil = T_true;
    Ttemp = T_true;
    Tnorm = 1.0;
  }

  static inline std::string MethodType() { return std::string("PTESolverFixedT"); }

  // Create a static member function bound to the class for ease of use
  constexpr static inline int RequiredScratch(const std::size_t nmat) {
    return PTESolverFixedTRequiredScratch(nmat);
  }
  constexpr static inline size_t RequiredScratchInBytes(const std::size_t nmat) {
    return PTESolverFixedTRequiredScratchInBytes(nmat);
  }

  PORTABLE_INLINE_FUNCTION
  Real Init() {
    // InitBase();
    // fake init base
    // rhobar is a fixed quantity: the average density of
    // material m averaged over the full PTE volume
    Tnorm = 1.0;
    this->InitRhoBarandRho();
    this->SetVfracFromT(Tequil);
    uscale = 0.0;
    for (std::size_t m = 0; m < nmat; m++) {
      // volume fractions have been potentially reset to ensure densitites are
      // larger than rho(Pmin(Tequil)); set the physical density to reflect
      // this change in volume fraction
      rho[m] = robust::ratio(rhobar[m], vfrac[m]);
      sie[m] = eos[m].InternalEnergyFromDensityTemperature(rho[m], Tequil, lambda[m]);
      // Guess uscale from initial density guess. Make sure it's positive
      uscale += std::abs(sie[m]) * rho[m];
      // note the scaling of pressure
      press[m] = eos[m].PressureFromDensityTemperature(rho[m], Tequil, lambda[m]);
    }
    uscale = std::max(uscale, 1.0e-14);
    for (std::size_t m = 0; m < nmat; ++m) {
      press[m] = robust::ratio(press[m], uscale);
      u[m] = sie[m] * robust::ratio(rhobar[m], uscale);
    }
    Residual();
    return ResidualNorm();
  }

  PORTABLE_INLINE_FUNCTION
  void Residual() const {
    Real vsum = mix_impl::sum_neumaier(vfrac, nmat);
    residual[0] = vfrac_total - vsum;
    for (std::size_t m = 0; m < nmat - 1; ++m) {
      residual[1 + m] = press[m] - press[m + 1];
    }
  }

  PORTABLE_INLINE_FUNCTION
  auto CheckPTE() const {
    Real mean_p = vfrac[0] * press[0];
    Real error_p = 0;
    for (std::size_t m = 1; m < nmat; ++m) {
      mean_p += vfrac[m] * press[m];
      error_p += residual[m] * residual[m];
    }
    error_p = std::sqrt(error_p);
    Real error_v = std::abs(residual[0]);
    // Check for convergence
    bool converged_p = (error_p < params_.pte_rel_tolerance_p * std::abs(mean_p) ||
                        error_p < params_.pte_abs_tolerance_p);
    bool converged_v = error_v < params_.pte_rel_tolerance_v;
    bool acceptable_p =
        (error_p < params_.pte_rel_tolerance_p_sufficient * std::abs(mean_p) ||
         error_p < params_.pte_abs_tolerance_p_sufficient);
    bool acceptable_v = error_v < params_.pte_rel_tolerance_v_sufficient;
    return std::make_pair(converged_p && converged_v, acceptable_p && acceptable_v);
  }

  PORTABLE_INLINE_FUNCTION
  void Jacobian() const {
    for (std::size_t m = 0; m < nmat; m++) {
      //////////////////////////////
      // perturb volume fractions
      //////////////////////////////
      Real dv = (vfrac[m] < 0.5 ? 1.0 : -1.0) * vfrac[m] * params_.derivative_eps;
      const Real vf_pert = vfrac[m] + dv;
      const Real rho_pert = robust::ratio(rhobar[m], vf_pert);

      Real p_pert = robust::ratio(
          eos[m].PressureFromDensityTemperature(rho_pert, Tequil, lambda[m]), uscale);
      dpdv[m] = std::min(robust::ratio((p_pert - press[m]), dv),
                         -std::abs(params_.pte_max_dpdv));
    }

    // Fill in the Jacobian
    for (std::size_t i = 0; i < neq * neq; ++i)
      jacobian[i] = 0.0;
    for (std::size_t m = 0; m < nmat; ++m) {
      jacobian[m] = 1.0;
    }
    for (std::size_t m = 0; m < nmat - 1; m++) {
      jacobian[MatIndex(m + 1, m)] = -dpdv[m];
      jacobian[MatIndex(m + 1, m + 1)] = dpdv[m + 1];
    }
  }

  PORTABLE_INLINE_FUNCTION
  Real ScaleDx() const {
    // Each check reduces the scale further if necessary
    Real scale = 1.0;
    // control how big of a step toward vfrac = 0 is allowed
    for (std::size_t m = 0; m < nmat; ++m) {
      if (scale * dx[m] < -params_.vfrac_safety_fac * vfrac[m]) {
        scale = robust::ratio(-params_.vfrac_safety_fac * vfrac[m], dx[m]);
      }
    }
    // control how big of a step toward rho = rho(Pmin) is allowed
    for (std::size_t m = 0; m < nmat; m++) {
      const Real rho_min = eos[m].RhoPmin(Tequil);
      const Real alpha_max = robust::ratio(rhobar[m], rho_min);
      if (alpha_max < vfrac[m]) {
        // Despite our best efforts, we're already in the unstable regime (i.e.
        // dPdV_T > 0) so we would actually want to *increase* the step instead
        // of decreasing it. As a result, this code doesn't work as intended and
        // should be skipped.
        continue;
      }
      if (scale * dx[m] > 0.5 * (alpha_max - vfrac[m])) {
        scale = robust::ratio(0.5 * (alpha_max - vfrac[m]), dx[m]);
      }
    }
    // Now apply the overall scaling
    for (std::size_t i = 0; i < neq; ++i)
      dx[i] *= scale;
    return scale;
  }

  // Update the solution and return new residual.  Possibly called repeatedly with
  // different scale factors as part of a line search
  PORTABLE_INLINE_FUNCTION
  Real TestUpdate(const Real scale, bool const cache_state = false) {
    if (cache_state) {
      // Store the current state in temp variables for first iteration of line
      // search
      for (std::size_t m = 0; m < nmat; ++m)
        vtemp[m] = vfrac[m];
    }
    for (std::size_t m = 0; m < nmat; ++m) {
      vfrac[m] = vtemp[m] + scale * dx[m];
      rho[m] = robust::ratio(rhobar[m], vfrac[m]);
      u[m] = rhobar[m] *
             eos[m].InternalEnergyFromDensityTemperature(rho[m], Tequil, lambda[m]);
      sie[m] = robust::ratio(u[m], rhobar[m]);
      u[m] = robust::ratio(u[m], uscale);
      press[m] = robust::ratio(
          eos[m].PressureFromDensityTemperature(rho[m], Tequil, lambda[m]), uscale);
    }
    Residual();
    return ResidualNorm();
  }

  PORTABLE_INLINE_FUNCTION
  Real GetPhysicalT() const { return Tnorm * Tequil; }

  // No-op debug stubs (only PTESolverRhoT has real prints)
  PORTABLE_INLINE_FUNCTION
  void DebugPrintState(std::size_t, Real, bool, bool) const {}
  PORTABLE_INLINE_FUNCTION
  void DebugPrintStep(Real) const {}
  PORTABLE_INLINE_FUNCTION
  bool IsAtTFloor() const { return false; }

 private:
  Real *dpdv, *vtemp;
  Real Tequil, Ttemp;
};

// ======================================================================
// fixed P solver
// ======================================================================
constexpr inline std::size_t PTESolverFixedPRequiredScratch(const std::size_t nmat) {
  std::size_t neq = nmat + 1;
  return neq * neq   // jacobian
         + 4 * neq   // dx, residual, and sol_scratch
         + 2 * nmat  // all the nmat sized arrays in base
         + 3 * nmat; // all the nmat sized arrays in fixedP
}
constexpr inline size_t PTESolverFixedPRequiredScratchInBytes(const std::size_t nmat) {
  return PTESolverFixedPRequiredScratch(nmat) * sizeof(Real);
}

template <typename EOSIndexer, typename RealIndexer, typename LambdaIndexer>
class PTESolverFixedP
    : public mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer> {
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::InitBase;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::AssignIncrement;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::nmat;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::neq;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::ResidualNorm;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::vfrac_total;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::sie_total;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::eos;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::rho;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::vfrac;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::sie;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::temp;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::press;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::rho_total;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::uscale;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::MatIndex;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::TryIdealPTE;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::jacobian;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::residual;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::dx;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::u;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::rhobar;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::Cache;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::Tnorm;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::params_;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::lambda;

 public:
  // template the ctor to get type deduction/universal references prior to c++17
  template <typename EOS_t, typename Real_t, typename CReal_t, typename Lambda_t>
  PORTABLE_INLINE_FUNCTION
  PTESolverFixedP(const std::size_t nmat, EOS_t &&eos, const Real vfrac_tot, const Real P,
                  Real_t &&rho, Real_t &&vfrac, Real_t &&sie, Real_t &&temp,
                  CReal_t &&press, Lambda_t &lambda, Real *scratch,
                  const MixParams &params = MixParams())
      : mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>(
            nmat, nmat + 1, eos, vfrac_tot, 1.0, rho, vfrac, sie, temp, press, lambda,
            scratch, 0.0, params) {
    dpdv = AssignIncrement(scratch, nmat);
    dpdT = AssignIncrement(scratch, nmat);
    vtemp = AssignIncrement(scratch, nmat);
    Pequil = P;
  }

  static inline std::string MethodType() { return std::string("PTESolverFixedP"); }

  // Create a static member function bound to the class for ease of use
  constexpr static inline int RequiredScratch(const std::size_t nmat) {
    return PTESolverFixedPRequiredScratch(nmat);
  }
  constexpr static inline size_t RequiredScratchInBytes(const std::size_t nmat) {
    return PTESolverFixedPRequiredScratchInBytes(nmat);
  }

  PORTABLE_INLINE_FUNCTION
  Real Init() {
    // InitBase();
    // fake init base
    // rhobar is a fixed quantity: the average density of
    // material m averaged over the full PTE volume
    this->InitRhoBarandRho();

    // guess some non-zero temperature to start
    const Real Tguess = this->GetTguess();
    // set the temperature normalization
    Tnorm = Tguess;
    // Assume pressure and energy scales are similar
    uscale = std::max(std::abs(Pequil), 1.0e-14);
    for (std::size_t m = 0; m < nmat; m++) {
      // scaled initial guess for temperature is just 1
      temp[m] = 1.0;
      sie[m] = eos[m].InternalEnergyFromDensityTemperature(rho[m], Tguess, lambda[m]);
    }

    // note the scaling of the material internal energy densities
    for (std::size_t m = 0; m < nmat; ++m) {
      u[m] = robust::ratio(sie[m] * rhobar[m], uscale);
      press[m] = robust::ratio(
          eos[m].PressureFromDensityTemperature(rho[m], Tguess, lambda[m]), uscale);
    }
    Residual();
    // Set the current guess for the equilibrium temperature.  Note that this is already
    // scaled.
    Tequil = temp[0];
    Ttemp = Tequil;
    return ResidualNorm();
  }

  PORTABLE_INLINE_FUNCTION
  void Residual() const {
    Real vsum = mix_impl::sum_neumaier(vfrac, nmat);
    for (std::size_t m = 0; m < nmat; ++m) {
      residual[m] = robust::ratio(Pequil, uscale) - press[m];
    }
    residual[nmat] = vfrac_total - vsum;
  }

  PORTABLE_INLINE_FUNCTION
  auto CheckPTE() const {
    Real error_p = 0;
    for (std::size_t m = 0; m < nmat; ++m) {
      error_p += residual[m] * residual[m];
    }
    error_p = std::sqrt(error_p);
    error_p *= uscale;
    Real error_v = std::abs(residual[neq - 1]);
    // Check for convergence
    bool converged_p =
        (error_p < params_.pte_rel_tolerance_p || error_p < params_.pte_abs_tolerance_p);
    bool converged_v = error_v < params_.pte_rel_tolerance_v;
    bool acceptable_p = (error_p < params_.pte_rel_tolerance_p_sufficient ||
                         error_p < params_.pte_abs_tolerance_p_sufficient);
    bool acceptable_v = error_v < params_.pte_rel_tolerance_v_sufficient;
    return std::make_pair(converged_p && converged_v, acceptable_p && acceptable_v);
  }

  PORTABLE_INLINE_FUNCTION
  void Jacobian() const {
    for (std::size_t m = 0; m < nmat; m++) {
      //////////////////////////////
      // perturb volume fractions
      //////////////////////////////
      Real dv = (vfrac[m] < 0.5 ? 1.0 : -1.0) * vfrac[m] * params_.derivative_eps;
      const Real vf_pert = vfrac[m] + dv;
      const Real rho_pert = robust::ratio(rhobar[m], vf_pert);

      Real p_pert{};
      Real e_pert{};
      p_pert =
          robust::ratio(this->GetPressureFromPreferred(eos[m], rho_pert, Tnorm * Tequil,
                                                       e_pert, lambda[m], true),
                        uscale);
      dpdv[m] = std::min(robust::ratio((p_pert - press[m]), dv),
                         -std::abs(params_.pte_max_dpdv));
      //////////////////////////////
      // perturb temperature
      //////////////////////////////
      Real dT = std::max(Tequil, 1.0) * params_.derivative_eps;

      p_pert = robust::ratio(this->GetPressureFromPreferred(eos[m], rho[m],
                                                            Tnorm * (Tequil + dT), e_pert,
                                                            lambda[m], true),
                             uscale);
      dpdT[m] = robust::ratio((p_pert - press[m]), dT);
    }

    // Fill in the Jacobian
    for (std::size_t i = 0; i < neq * neq; ++i)
      jacobian[i] = 0.0;
    for (std::size_t m = 0; m < nmat; m++) {
      jacobian[MatIndex(m, m)] = dpdv[m];
      jacobian[MatIndex(m, nmat)] = dpdT[m];
    }
    for (std::size_t m = 0; m < neq - 1; ++m) {
      jacobian[MatIndex(neq - 1, m)] = 1.0;
    }
  }

  PORTABLE_INLINE_FUNCTION
  Real ScaleDx() const {
    // Each check reduces the scale further if necessary
    Real scale = 1.0;
    // control how big of a step toward vfrac = 0 is allowed
    for (std::size_t m = 0; m < nmat; ++m) {
      if (scale * dx[m] < -params_.vfrac_safety_fac * vfrac[m]) {
        scale = -params_.vfrac_safety_fac * robust::ratio(vfrac[m], dx[m]);
      }
    }
    const Real Tnew = Tequil + scale * dx[nmat];
    // control how big of a step toward rho = rho(Pmin) is allowed
    for (std::size_t m = 0; m < nmat; m++) {
      const Real rho_min =
          std::max(eos[m].RhoPmin(Tnorm * Tequil), eos[m].RhoPmin(Tnorm * Tnew));
      const Real alpha_max = robust::ratio(rhobar[m], rho_min);
      if (alpha_max < vfrac[m]) {
        // Despite our best efforts, we're already in the unstable regime (i.e.
        // dPdV_T > 0) so we would actually want to *increase* the step instead
        // of decreasing it. As a result, this code doesn't work as intended and
        // should be skipped.
        continue;
      }
      if (scale * dx[m] > 0.5 * (alpha_max - vfrac[m])) {
        scale = 0.5 * robust::ratio(alpha_max - vfrac[m], dx[m]);
      }
    }
    // control how big of a step toward T = 0 is allowed
    if (scale * dx[nmat] < -0.95 * Tequil) {
      scale = -0.95 * robust::ratio(Tequil, dx[nmat]);
    }
    // Now apply the overall scaling
    for (std::size_t i = 0; i < neq; ++i)
      dx[i] *= scale;
    return scale;
  }

  // Update the solution and return new residual.  Possibly called repeatedly with
  // different scale factors as part of a line search
  PORTABLE_INLINE_FUNCTION
  Real TestUpdate(const Real scale, bool const cache_state = false) {
    if (cache_state) {
      // Store the current state in temp variables for first iteration of line
      // search
      Ttemp = Tequil;
      for (std::size_t m = 0; m < nmat; ++m)
        vtemp[m] = vfrac[m];
    }
    Tequil = Ttemp + scale * dx[nmat];
    for (std::size_t m = 0; m < nmat; ++m) {
      vfrac[m] = vtemp[m] + scale * dx[m];
      rho[m] = robust::ratio(rhobar[m], vfrac[m]);
      u[m] = rhobar[m] * eos[m].InternalEnergyFromDensityTemperature(
                             rho[m], Tnorm * Tequil, lambda[m]);
      press[m] = robust::ratio(
          eos[m].PressureFromDensityTemperature(rho[m], Tnorm * Tequil, lambda[m]),
          uscale);
      sie[m] = robust::ratio(u[m], rhobar[m]);
      u[m] = robust::ratio(u[m], uscale);
      temp[m] = Tequil;
    }
    Residual();
    return ResidualNorm();
  }

  PORTABLE_INLINE_FUNCTION
  Real GetPhysicalT() const { return Tnorm * Tequil; }

  // No-op debug stubs (only PTESolverRhoT has real prints)
  PORTABLE_INLINE_FUNCTION
  void DebugPrintState(std::size_t, Real, bool, bool) const {}
  PORTABLE_INLINE_FUNCTION
  void DebugPrintStep(Real) const {}
  PORTABLE_INLINE_FUNCTION
  bool IsAtTFloor() const { return false; }

 private:
  Real *dpdv, *dpdT, *vtemp;
  Real Tequil, Ttemp, Pequil;
};

// ======================================================================
// RhoU Solver
// ======================================================================
constexpr inline std::size_t PTESolverRhoURequiredScratch(const std::size_t nmat) {
  std::size_t neq = 2 * nmat;
  return neq * neq   // jacobian
         + 4 * neq   // dx, residual, and sol_scratch
         + 8 * nmat; // all the nmat sized arrays
}
constexpr inline size_t PTESolverRhoURequiredScratchInBytes(const std::size_t nmat) {
  return PTESolverRhoURequiredScratch(nmat) * sizeof(Real);
}

template <typename EOSIndexer, typename RealIndexer, typename LambdaIndexer>
class PTESolverRhoU
    : public mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer> {
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::InitBase;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::AssignIncrement;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::nmat;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::neq;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::ResidualNorm;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::vfrac_total;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::sie_total;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::eos;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::rho;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::vfrac;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::sie;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::temp;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::press;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::rho_total;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::uscale;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::utotal_scale;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::TryIdealPTE;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::MatIndex;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::jacobian;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::residual;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::dx;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::u;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::rhobar;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::Cache;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::Tnorm;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::params_;
  using mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>::lambda;

 public:
  // template the ctor to get type deduction/universal references prior to c++17
  template <typename EOS_t, typename Real_t, typename Lambda_t>
  PORTABLE_INLINE_FUNCTION
  PTESolverRhoU(const std::size_t nmat, const EOS_t &&eos, const Real vfrac_tot,
                const Real sie_tot, Real_t &&rho, Real_t &&vfrac, Real_t &&sie,
                Real_t &&temp, Real_t &&press, Lambda_t &&lambda, Real *scratch,
                const Real Tnorm = 0.0, const MixParams &params = MixParams())
      : mix_impl::PTESolverBase<EOSIndexer, RealIndexer, LambdaIndexer>(
            nmat, 2 * nmat, eos, vfrac_tot, sie_tot, rho, vfrac, sie, temp, press, lambda,
            scratch, Tnorm, params) {
    dpdv = AssignIncrement(scratch, nmat);
    dtdv = AssignIncrement(scratch, nmat);
    dpde = AssignIncrement(scratch, nmat);
    dtde = AssignIncrement(scratch, nmat);
    vtemp = AssignIncrement(scratch, nmat);
    utemp = AssignIncrement(scratch, nmat);
  }

  static inline std::string MethodType() { return std::string("PTESolverRhoU"); }

  // Create a static member function bound to the class for ease of use
  constexpr static inline int RequiredScratch(const std::size_t nmat) {
    return PTESolverRhoURequiredScratch(nmat);
  }
  constexpr static inline size_t RequiredScratchInBytes(const std::size_t nmat) {
    return PTESolverRhoURequiredScratchInBytes(nmat);
  }

  PORTABLE_INLINE_FUNCTION
  Real Init() {
    InitBase();
    Residual();
    TryIdealPTE(this);
    return ResidualNorm();
  }

  PORTABLE_INLINE_FUNCTION
  void Residual() const {
    Real vsum = mix_impl::sum_neumaier(vfrac, nmat);
    Real esum = mix_impl::sum_neumaier(u, nmat);
    residual[0] = vfrac_total - vsum;
    residual[1] = utotal_scale - esum;
    for (std::size_t m = 0; m < nmat - 1; ++m) {
      residual[2 + m] = press[m + 1] - press[m];
    }
    for (std::size_t m = nmat + 1; m < neq; m++) {
      residual[m] = temp[m - nmat] - temp[m - nmat - 1];
    }
  }

  PORTABLE_INLINE_FUNCTION
  auto CheckPTE() const {
    Real mean_p = vfrac[0] * press[0];
    Real mean_t = rhobar[0] * temp[0];
    Real error_p = 0.0;
    Real error_t = 0.0;
    for (std::size_t m = 1; m < nmat; ++m) {
      mean_p += vfrac[m] * press[m];
      mean_t += rhobar[m] * temp[m];
      error_p += residual[m + 1] * residual[m + 1];
      error_t += residual[m + nmat] * residual[m + nmat];
    }
    mean_t = robust::ratio(mean_t, rho_total);
    error_p = std::sqrt(error_p);
    error_t = std::sqrt(error_t);
    // Check for convergence
    bool converged_p = (error_p < params_.pte_rel_tolerance_p * std::abs(mean_p) ||
                        error_p < params_.pte_abs_tolerance_p);
    bool converged_t = (error_t < params_.pte_rel_tolerance_t * mean_t ||
                        error_t < params_.pte_abs_tolerance_t);
    bool acceptable_p =
        (error_p < params_.pte_rel_tolerance_p_sufficient * std::abs(mean_p) ||
         error_p < params_.pte_abs_tolerance_p_sufficient);
    bool acceptable_t = (error_t < params_.pte_rel_tolerance_t_sufficient * mean_t ||
                         error_t < params_.pte_abs_tolerance_t_sufficient);
    return std::make_pair(converged_p && converged_t, acceptable_p && acceptable_t);
  }

  PORTABLE_INLINE_FUNCTION
  void Jacobian() const {
    for (std::size_t m = 0; m < nmat; m++) {
      //////////////////////////////
      // perturb volume fractions
      //////////////////////////////
      Real dv = (vfrac[m] < 0.5 ? 1.0 : -1.0) * vfrac[m] * params_.derivative_eps;
      const Real vf_pert = vfrac[m] + dv;
      const Real rho_pert = robust::ratio(rhobar[m], vf_pert);

      Real p_pert;
      Real t_pert = robust::ratio(
          eos[m].TemperatureFromDensityInternalEnergy(rho_pert, sie[m], lambda[m]),
          Tnorm);
      p_pert =
          robust::ratio(this->GetPressureFromPreferred(eos[m], rho_pert, Tnorm * t_pert,
                                                       sie[m], lambda[m], false),
                        uscale);
      dpdv[m] =
          std::min(robust::ratio(p_pert - press[m], dv), -std::abs(params_.pte_max_dpdv));
      dtdv[m] = robust::ratio(t_pert - temp[m], dv);
      //////////////////////////////
      // perturb energies
      //////////////////////////////
      const Real de = std::abs(u[m]) * params_.derivative_eps;
      Real e_pert = robust::ratio(u[m] + de, rhobar[m]);

      t_pert = robust::ratio(
          eos[m].TemperatureFromDensityInternalEnergy(rho[m], uscale * e_pert, lambda[m]),
          Tnorm);
      p_pert =
          robust::ratio(this->GetPressureFromPreferred(eos[m], rho[m], Tnorm * t_pert,
                                                       uscale * e_pert, lambda[m], false),
                        uscale);
      dpde[m] = robust::ratio(p_pert - press[m], de);
      dtde[m] = robust::ratio(t_pert - temp[m], de);
      if (std::abs(dtde[m]) < params_.min_dtde) { // must be on the cold curve
        dtde[m] = params_.derivative_eps;
      }
    }
    for (std::size_t i = 0; i < neq * neq; ++i)
      jacobian[i] = 0.0;
    // TODO(JCD): clean all this up with MatIndex
    for (std::size_t m = 0; m < nmat; ++m) {
      jacobian[m] = 1.0;
      jacobian[2 * nmat + nmat + m] = 1.0;
    }
    jacobian[2 * 2 * nmat] = dpdv[0];
    jacobian[nmat * 2 * nmat + nmat - 1] = -dpdv[nmat - 1];
    jacobian[(nmat + 1) * 2 * nmat] = dtdv[0];
    jacobian[(2 * nmat - 1) * 2 * nmat + nmat - 1] = -dtdv[nmat - 1];
    jacobian[2 * 2 * nmat + nmat] = dpde[0];
    jacobian[nmat * 2 * nmat + 2 * nmat - 1] = -dpde[nmat - 1];
    jacobian[(nmat + 1) * 2 * nmat + nmat] = dtde[0];
    jacobian[(2 * nmat - 1) * 2 * nmat + 2 * nmat - 1] = -dtde[nmat - 1];
    for (std::size_t m = 1; m < nmat - 1; ++m) {
      jacobian[(1 + m) * 2 * nmat + m] = -dpdv[m];
      jacobian[(2 + m) * 2 * nmat + m] = dpdv[m];
      jacobian[(nmat + m) * 2 * nmat + m] = -dtdv[m];
      jacobian[(nmat + m + 1) * 2 * nmat + m] = dtdv[m];
      jacobian[(1 + m) * 2 * nmat + nmat + m] = -dpde[m];
      jacobian[(2 + m) * 2 * nmat + nmat + m] = dpde[m];
      jacobian[(nmat + m) * 2 * nmat + nmat + m] = -dtde[m];
      jacobian[(nmat + m + 1) * 2 * nmat + nmat + m] = dtde[m];
    }
  }

  PORTABLE_INLINE_FUNCTION
  Real ScaleDx() const {
    // Each check reduces the scale further if necessary
    Real scale = 1.0;
    for (std::size_t m = 0; m < nmat; ++m) {
      // control how big of a step toward vfrac = 0 is allowed
      if (scale * dx[m] < -0.1 * vfrac[m]) {
        scale = -0.1 * robust::ratio(vfrac[m], dx[m]);
      }
      // try to control steps toward T = 0
      const Real dt = (dtdv[m] * dx[m] + dtde[m] * dx[m + nmat]);
      if (scale * dt < -0.1 * temp[m]) {
        scale = -0.1 * robust::ratio(temp[m], dt);
      }
      const Real tt = temp[m] + scale * dt;
      const Real rho_min =
          std::max(eos[m].RhoPmin(Tnorm * temp[m]), eos[m].RhoPmin(Tnorm * tt));
      const Real alpha_max = robust::ratio(rhobar[m], rho_min);
      // control how big of a step toward rho = rho(Pmin) is allowed
      if (alpha_max < vfrac[m]) {
        // Despite our best efforts, we're already in the unstable regime (i.e.
        // dPdV_T > 0) so we would actually want to *increase* the step instead
        // of decreasing it. As a result, this code doesn't work as intended and
        // should be skipped.
        continue;
      }
      if (scale * dx[m] > 0.5 * (alpha_max - vfrac[m])) {
        scale = 0.5 * robust::ratio(alpha_max - vfrac[m], dx[m]);
      }
    }
    // Now apply the overall scaling
    for (std::size_t i = 0; i < neq; ++i)
      dx[i] *= scale;
    return scale;
  }

  // Update the solution and return new residual.  Possibly called repeatedly with
  // different scale factors as part of a line search
  PORTABLE_INLINE_FUNCTION
  Real TestUpdate(const Real scale, bool const cache_state = false) const {
    if (cache_state) {
      // Store the current state in temp variables for first iteration of line
      // search
      for (std::size_t m = 0; m < nmat; ++m) {
        vtemp[m] = vfrac[m];
        utemp[m] = u[m];
      }
    }
    for (std::size_t m = 0; m < nmat; ++m) {
      vfrac[m] = vtemp[m] + scale * dx[m];
      rho[m] = robust::ratio(rhobar[m], vfrac[m]);
      u[m] = utemp[m] + scale * dx[nmat + m];
      sie[m] = uscale * robust::ratio(u[m], rhobar[m]);
      temp[m] = robust::ratio(
          eos[m].TemperatureFromDensityInternalEnergy(rho[m], sie[m], lambda[m]), Tnorm);
      press[m] =
          robust::ratio(this->GetPressureFromPreferred(eos[m], rho[m], Tnorm * temp[m],
                                                       sie[m], lambda[m], false),
                        uscale);
    }
    Residual();
    return ResidualNorm();
  }

  // No-op stub: PTESolverRhoU has per-material temperatures, not a single
  // equilibrium temperature, so pressure-sign-mismatch jump does not apply.
  PORTABLE_INLINE_FUNCTION
  Real GetPhysicalT() const { return 0.0; }

  // PTESolverRhoU has per-material temperatures (not a single equilibrium T),
  // so the pressure-sign-mismatch jump is not applicable.  Return false to
  // disable it without special-casing the solver loop.
  PORTABLE_INLINE_FUNCTION
  bool TryPressureJump(const Real /*T_physical*/, const MixParams & /*params*/) {
    return false;
  }

  // No-op debug stubs (only PTESolverRhoT has real prints)
  PORTABLE_INLINE_FUNCTION
  void DebugPrintState(std::size_t, Real, bool, bool) const {}
  PORTABLE_INLINE_FUNCTION
  void DebugPrintStep(Real) const {}
  PORTABLE_INLINE_FUNCTION
  bool IsAtTFloor() const { return false; }

 private:
  Real *dpdv, *dtdv, *dpde, *dtde, *vtemp, *utemp;
};

// ======================================================================
// Solver loop
// ======================================================================
template <class System>
PORTABLE_INLINE_FUNCTION SolverStatus PTESolver(System &s) {
  SolverStatus status;
  Real &err = status.residual;
  bool &converged = status.converged;
  bool close_enough = false;

  // initialize the system, fill in residual, and get its norm
  err = s.Init();

  // Pull out params
  const MixParams &params = s.GetParams();

  converged = false;
  const std::size_t pte_max_iter = s.Nmat() * params.pte_max_iter_per_mat;
  auto &niter = s.Niter();
  auto &small_step_iters = status.small_step_iters;
  for (niter = 0; niter < pte_max_iter; ++niter) {
    status.max_niter = std::max(status.max_niter, niter);

    // Check for convergence
    // TODO(JMM): I wish I could use std::tie or structured binding,
    // but our clang-based toolchain does not like it
    auto check = s.CheckPTE();
    converged = check.first;
    close_enough = check.second;
    if (converged) {
      s.DebugPrintState(niter, err, converged, close_enough);
      break;
    }

    // compute the Jacobian
    s.Jacobian();

    // solve for the Newton step
    bool success = s.Solve();
    s.DebugPrintState(niter, err, converged, close_enough);
    if (!success) {
      // do something to crash out?  Tell folks what happened?
      // printf("crashing out at iteration: %ld\n", niter);
      converged = false;
      break;
    }

    // possibly scale the update to stay within reasonable bounds
    Real scale = std::min(1.0, s.ScaleDx());
    s.DebugPrintStep(scale);
    PORTABLE_REQUIRE(scale <= 1.0, "PTE Solver is attempting to increase the step size");

    // If scale is very small, we may be iterating around the regime
    // of validity and not making progress. Try to catch this, fail
    // out early, and report this as a failure, so fallbacks can take
    // over in a host code. We check that this has happened for a few
    // iterations, in case we can escape.
    if (std::abs(scale) < params.pte_small_step_thresh) small_step_iters++;
    if (small_step_iters >= params.pte_small_step_tries) {
#ifdef PTE_DEBUG_TRACE
      std::printf(
          "[PTESolver] small_step_iters=%zu >= pte_small_step_tries=%zu at iter %zu\n",
          small_step_iters, params.pte_small_step_tries, niter);
      std::printf(
          "[PTESolver] pressure_jump_enabled=%d  attempts=%zu  max_attempts=%zu\n",
          (int)params.pte_pressure_jump_enabled, status.pressure_jump_attempts,
          params.pte_pressure_jump_max_attempts);
#endif
      // Attempt pressure-sign-mismatch jump before failing out.
      // When a material is trapped in a deep negative-pressure region
      // (cold-curve binding well) while others are at positive pressure,
      // bisect to find the P=0 crossing and jump the stuck material there.
      // Skip if temperature is already at the EOS floor — pressure jumps
      // cannot help when the solver needs to lower T but can't.
      if (params.pte_pressure_jump_enabled &&
          status.pressure_jump_attempts < params.pte_pressure_jump_max_attempts &&
          !s.IsAtTFloor()) {
        Real T_phys = s.GetPhysicalT();
#ifdef PTE_DEBUG_TRACE
        std::printf("[PTESolver] Attempting pressure jump with T_physical=%.6e\n",
                    T_phys);
#endif
        bool jumped = s.TryPressureJump(T_phys, params);
        status.pressure_jump_attempts++;
#ifdef PTE_DEBUG_TRACE
        std::printf("[PTESolver] TryPressureJump returned jumped=%d\n", (int)jumped);
#endif
        if (jumped) {
          small_step_iters = 0;
          s.Fixup();
          err = s.TestUpdate(0.0, true);
#ifdef PTE_DEBUG_TRACE
          std::printf("[PTESolver] Post-jump err=%.6e\n", err);
#endif
          continue;
        }
      }
#ifdef PTE_DEBUG_TRACE
      std::printf("[PTESolver] Small-step exit at iter %zu (at_T_floor=%d)\n",
                  niter, (int)s.IsAtTFloor());
#endif
      converged = false;
      break;
    }

    // Line search
    Real gradfdx = -2.0 * scale * err;
    scale = 1.0; // New scale for line search
    Real err_old = err;
    // Test the update and reset the cache the current state
    err = s.TestUpdate(scale, true /* cache_state */);
    if (err > err_old + params.line_search_alpha * gradfdx) {
      // backtrack to middle of step
      scale = 0.5;
      Real err_mid = s.TestUpdate(scale);
      if (err_mid < err && err_mid < err_old) {
        // We know the half step is better than both the full step and the
        // prior result, so try a pseudo parabolic fit to the error and find its
        // minimum. The `scale` value is bound between 0.75 and 0.25.
        scale = 0.75 + 0.5 * robust::ratio(err_mid - err, err - 2.0 * err_mid + err_old);
      }
      for (std::size_t line_iter = 0; line_iter < params.line_search_max_iter;
           line_iter++) {
        err = s.TestUpdate(scale);
        if (err < err_old + params.line_search_alpha * scale * gradfdx) break;
        // shrink the step if the error isn't reduced enough
        scale *= params.line_search_fac;
        status.max_line_niter = std::max(line_iter, status.max_line_niter);
      }
    }

    // apply fixes post update, e.g. renormalize volume fractions to deal with round-off
    s.Fixup();

    // Track stagnation: consecutive iterations with negligible error reduction.
    // When the solver is stuck in a linear drift (e.g. one material deep in a
    // negative-pressure binding well while others are at positive pressure),
    // the error ratio err/err_old is very close to 1.0 for many iterations.
    if (err > params.pte_stagnation_thresh * err_old) {
      status.stagnation_iters++;
    } else {
      status.stagnation_iters = 0;
    }

    // If stagnating and NOT close_enough, attempt a pressure-sign-mismatch
    // jump.  This catches the failure mode where ScaleDx() never returns a
    // tiny value (so small_step_iters is never incremented) but the solver
    // is drifting linearly with no prospect of convergence.
    if (status.stagnation_iters >= params.pte_stagnation_tries && !close_enough) {
      if (params.pte_pressure_jump_enabled &&
          status.pressure_jump_attempts < params.pte_pressure_jump_max_attempts &&
          !s.IsAtTFloor()) {
#ifdef PTE_DEBUG_TRACE
        std::printf("[PTESolver] Stagnation detected: %zu iters with err/err_old > %.4f "
                    "at iter %zu (err=%.6e)\n",
                    status.stagnation_iters, params.pte_stagnation_thresh, niter, err);
#endif
        Real T_phys = s.GetPhysicalT();
#ifdef PTE_DEBUG_TRACE
        std::printf("[PTESolver] Attempting pressure jump with T_physical=%.6e\n",
                    T_phys);
#endif
        bool jumped = s.TryPressureJump(T_phys, params);
        status.pressure_jump_attempts++;
#ifdef PTE_DEBUG_TRACE
        std::printf("[PTESolver] TryPressureJump returned jumped=%d\n", (int)jumped);
#endif
        if (jumped) {
          status.stagnation_iters = 0;
          s.Fixup();
          err = s.TestUpdate(0.0, true);
#ifdef PTE_DEBUG_TRACE
          std::printf("[PTESolver] Post-jump err=%.6e\n", err);
#endif
          continue;
        }
        // Jump failed — fall through to fail below
      }
      // Pressure jumps exhausted, unavailable, or skipped (T floor).
#ifdef PTE_DEBUG_TRACE
      std::printf("[PTESolver] Stagnation exit at iter %zu (at_T_floor=%d)\n",
                  niter, (int)s.IsAtTFloor());
#endif
      converged = false;
      break;
    }

    // check for the case where the solver is no longer moving towards
    // a descent direction sufficiently rapidly, but we believe we
    // have converged to an acceptable level This can also happen if
    // the residual is so small we are at the limits of floating point
    // precision.
    if (((err > params.pte_slow_convergence_thresh * err_old) ||
         niter == pte_max_iter - 1) &&
        close_enough) {
      converged = true;
      status.slow_convergence_detected = true;
      break;
    }
  }
  // undo any scaling that was applied internally for the solver
  s.Finalize();
  return status;
}

} // namespace singularity

#endif // _SINGULARITY_EOS_CLOSURE_MIXED_CELL_MODELS_
