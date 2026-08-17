//------------------------------------------------------------------------------
// PTE as a concave maximisation in the conjugate variables (beta, gamma) = (1/T, P/T).
//------------------------------------------------------------------------------
// THE METHOD.  The cyclic solvers iterate on (P, T) with the residual pair
//
//     R = ( sum_m Y_m e_m(P,T) - e0 ,  sum_m Y_m tau_m(P,T) - tau0 ),
//
// whose Jacobian in (P, T) is NOT symmetric and is not the derivative of anything, so there is no
// merit function to line-search on -- exactly the objection recorded in mixed_cell_models.hpp
// ("the cyclic step is not a descent direction on the residual norm, so the generic Newton +
// line-search driver would fight it").  Changing variables removes it.  With beta = 1/T,
// gamma = P/T and the Planck potential Xi = G/T = beta*e + gamma*tau - s,
//
//     dXi = e dbeta + tau dgamma,
//
// so Xi is a potential whose gradient is (e, tau).  Three things follow:
//
//   1. MIXING IS EXACTLY ADDITIVE.  Xi_mix = sum_m Y_m Xi_m at common (beta, gamma), because P
//      and T are shared by construction.  The inter-material coupling the cyclic method iterates
//      on is gone, not approximated.
//   2. THE RESIDUAL IS A GRADIENT.  With J = Xi_mix - beta*e0 - gamma*tau0, grad J = R exactly.
//      The two equations are the stationarity condition of one scalar, the Legendre dual of
//      maximising entropy at fixed (tau0, e0).
//   3. THE JACOBIAN IS A SYMMETRIC HESSIAN.  Xi_bg = Xi_gb IS the Maxwell relation
//      (de/dP)_T = -T (dtau/dT)_P - P (dtau/dP)_T, and Hess Xi is negative definite exactly where
//      the material is Menikoff-Plohr stable (K_T > 0, c_v > 0) -- the condition the offline
//      free-energy fit already enforces.  A sum of negative-definite matrices is negative
//      definite, so the mixture Hessian is nonsingular for any nmat and any mass fractions.
//
// Symmetry is what makes the Newton direction d = -H^-1 R a descent direction on ||R||^2:
// grad(merit).d = -R^T W^2 R < 0 needs H^T H^-1 = I, i.e. H symmetric.
//
// THE CYCLIC METHOD IS THE SAME PROGRAM.  PTESolveCyclicRhoE's two bracketed sub-solves are exact
// block-coordinate maximisations of this same J: the isochoric solve is dJ/dgamma = 0 and the
// isoenergetic one is dJ/dbeta = 0.  Each is bracketable because each residual is strictly
// monotone in its own coordinate -- d2J/dgamma2 < 0 <=> K_T > 0, and d2J/dbeta2 < 0 <=> c_v > 0.
// "The bracket IS the robustness" is coordinatewise concavity.  So the two are not rivals but the
// textbook pair: coordinate ascent is globally convergent and derivative-free, Newton is locally
// quadratic, and a step of either may be taken at any time without leaving the theory.
//
// This solver therefore takes Newton steps and falls back to one coordinate step whenever Newton
// fails to CONTRACT the residual.  The switch is a measurement, never a degeneracy sensor.
//
// ⚠️ TWO RULES, BOTH MEASURED, NEITHER OBVIOUS -- do not "simplify" either away:
//
//   a. The coordinate step is accepted UNCONDITIONALLY.  It always increases J, but it may raise
//      ||R|| on the way, and gating it on the residual discards precisely the property being
//      borrowed.  Measured offline on the RUN102 interface cell (Cu+DD+TT, rho=0.2483, T=19.47),
//      64 seeds over the box: 1.6% converged with a ||R|| gate, 65.6% without.
//   b. Newton must CONTRACT (merit < 0.5 * merit), not merely descend, while a bracket is
//      available.  The alternative is an exact maximisation in one coordinate, so a Newton step
//      that only inches the residual down is not worth taking; near the solution Newton contracts
//      quadratically and passes trivially, on a plateau it crawls and the safeguard takes over.
//      Same cell: 65.6% -> 100%.
//
// MEASURED against PTESolveCyclicRhoE offline (V22 tables, EOS evaluations per solve, 64 seeds):
//
//     Al+Cu rho=3.0 T=3000, cold seeds : cyclic_rhoe 100% @ 1561 | dual 100% @ 130   (12x)
//     RUN102 plateau,       cold seeds : cyclic_rhoe 100% @ 11590 | dual 100% @ 2206 (5.3x)
//     RUN102 plateau,       warm seeds : cyclic_rhoe 100% @ 10937 | dual 100% @ 2113 (5.2x)
//
// ⚠️ IT NEEDS CONSISTENT DERIVATIVES, AND THAT IS THE WHOLE GATE.  The Hessian is assembled from
// DensityEnergyDerivativesFromPressureTemperature.  Reconstructing those partials by differencing
// rho and e instead violates the Maxwell relation by a few percent, and symmetrising a
// 3%-asymmetric Jacobian destroys the descent guarantee: the same offline harness scored 59%
// against 98% purely on that difference.  TableDependsPT and TableDependsRhoT both return the
// stored fields, every one descended from the single fitted free energy, so this holds on the
// shipped tables -- but a backend that derives them some other way must be checked.  `status`
// carries the measured asymmetry so the caller can see it rather than assume it.
//
// WHAT IT DOES NOT FIX.  det Hess Xi ~ 1/(c_v K_T) still diverges on a two-phase plateau, and
// Xi(beta, gamma) is not even single-valued inside the dome -- a whole tie line collapses to one
// (beta, gamma) point.  The safeguard routes around that; it does not repair it, and recovering
// the tie line's quality itself still needs a primal unknown (see pte_cyclic_rhoe_lever.hpp).
//
// Units are the caller's, and must be self-consistent (FLASH passes cgs): P in barye, T in K,
// rho in g/cm^3, e in erg/g.
//------------------------------------------------------------------------------

#ifndef _SINGULARITY_EOS_CLOSURE_PTE_DUAL_
#define _SINGULARITY_EOS_CLOSURE_PTE_DUAL_

#include <cmath>
#include <limits>

#include <ports-of-call/portability.hpp>
#include <singularity-eos/base/robust_utils.hpp>
#include <singularity-eos/closure/mixed_cell_models.hpp> // MixParams, SolverStatus
#include <singularity-eos/closure/pte_cyclic_rhoe.hpp>   // MixtureTauE, LogBracketRoot

namespace singularity {

namespace pte_dual_impl {

using cyclic_rhoe_impl::LogBracketRoot;
using cyclic_rhoe_impl::MixtureTauE;

// Smallest |eigenvalue| the Hessian may keep, as a fraction of its spectral radius.  RELATIVE,
// because the two diagonal entries carry different units and differ by many orders of magnitude:
// an absolute floor (an `H + eps*I` ridge) would be a different physical statement on each axis.
constexpr Real kEigenvalueFloorFraction = 1.0e-10;

// Contraction a Newton step must achieve to be preferred over an exact coordinate maximisation.
constexpr Real kNewtonContraction = 0.5;

// Step-length guards.  A Newton step may not push beta or gamma below this fraction of its
// current value (both are strictly positive), nor multiply either by more than kMaxStepRatio.
// One decade per iteration crosses a ~20-decade seed box in ~20 steps while never leaving the
// region the local Hessian describes.
constexpr Real kPositivityRetention = 0.1;
constexpr Real kMaxStepRatio = 10.0;

constexpr int kMaxBacktracks = 30;

// The mixture gradient and Hessian of Xi at one (P, T).
struct DualState {
  Real residual_energy; // dJ/dbeta  = sum Y_m e_m   - e0
  Real residual_tau;    // dJ/dgamma = sum Y_m tau_m - tau0
  Real xi_bb;
  Real xi_bg; // symmetrised from the two routes that compute it
  Real xi_gg;
  Real asymmetry; // relative disagreement between those routes; 0 for a consistent EOS
  // The raw mixture partials in (P,T).  Kept because the safeguard needs the SAME two tangent
  // slopes PTESolveCyclicRhoE builds, and here they come free with the Hessian -- that solver
  // spends 8 extra mixture evaluations on central differences to estimate them.
  Real dtau_dP, dtau_dT, de_dP, de_dT;
  bool valid;
};

// Assemble grad J and Hess Xi from ONE derivative evaluation per material.
//
// The chain rule from (P, T) to (beta, gamma) is fixed by T = 1/beta, P = gamma/beta:
//     dX/dgamma|_beta = T (dX/dP)_T
//     dX/dbeta|_gamma = -P T (dX/dP)_T - T^2 (dX/dT)_P
template <typename EOSIndexer, typename RealIndexer, typename LambdaIndexer>
PORTABLE_INLINE_FUNCTION DualState MixtureDual(const std::size_t nmat, EOSIndexer &eos,
                                               const RealIndexer &Ym, LambdaIndexer &lambda,
                                               const Real tau_target, const Real sie_tot,
                                               const Real P, const Real T, Real *rho_out,
                                               Real *sie_out) {
  DualState s{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, false};
  Real tau_mix = 0.0, e_mix = 0.0;
  Real dtau_dP = 0.0, dtau_dT = 0.0, de_dP = 0.0, de_dT = 0.0;
  for (std::size_t m = 0; m < nmat; ++m) {
    Real rho_m, sie_m, drho_dP, drho_dT, de_dP_m, de_dT_m;
    eos[m].DensityEnergyDerivativesFromPressureTemperature(P, T, lambda[m], rho_m, sie_m,
                                                           drho_dP, drho_dT, de_dP_m, de_dT_m);
    if (!(rho_m > 0.0) || !std::isfinite(sie_m) || !std::isfinite(drho_dP) ||
        !std::isfinite(drho_dT) || !std::isfinite(de_dP_m) || !std::isfinite(de_dT_m)) {
      return s;
    }
    // tau = 1/rho, so dtau/dX = -(drho/dX)/rho^2.
    const Real inv_rho2 = robust::ratio(1.0, rho_m * rho_m);
    // Hand the state back: the caller's finalize would otherwise re-query the table for numbers
    // this call already has, which on a warm-converged cell DOUBLES the solve's table work.
    if (rho_out != nullptr) rho_out[m] = rho_m;
    if (sie_out != nullptr) sie_out[m] = sie_m;
    tau_mix += Ym[m] * robust::ratio(1.0, rho_m);
    e_mix += Ym[m] * sie_m;
    dtau_dP -= Ym[m] * drho_dP * inv_rho2;
    dtau_dT -= Ym[m] * drho_dT * inv_rho2;
    de_dP += Ym[m] * de_dP_m;
    de_dT += Ym[m] * de_dT_m;
  }

  const Real xi_gg = T * dtau_dP;
  const Real xi_bb = -P * T * de_dP - T * T * de_dT;
  // The cross term both ways, so the disagreement is measurable rather than silently averaged.
  const Real xi_bg_from_e = T * de_dP;
  const Real xi_bg_from_tau = -P * T * dtau_dP - T * T * dtau_dT;
  // Scale the disagreement by the Cauchy-Schwarz bound on an off-diagonal entry of a definite
  // matrix, NOT by the cross term itself: the true cross term VANISHES for an ideal gas, where
  // xi_bg_from_tau is a cancellation of two equal large numbers, and dividing by that residue
  // reports a perfectly consistent EOS as 100% asymmetric.
  const Real scale = std::sqrt(std::abs(xi_bb * xi_gg));
  s.dtau_dP = dtau_dP;
  s.dtau_dT = dtau_dT;
  s.de_dP = de_dP;
  s.de_dT = de_dT;
  s.residual_energy = e_mix - sie_tot;
  s.residual_tau = tau_mix - tau_target;
  s.xi_bb = xi_bb;
  s.xi_gg = xi_gg;
  s.xi_bg = 0.5 * (xi_bg_from_e + xi_bg_from_tau);
  s.asymmetry = (scale > 0.0) ? std::abs(xi_bg_from_e - xi_bg_from_tau) / scale : 0.0;
  s.valid = std::isfinite(s.residual_energy) && std::isfinite(s.residual_tau) &&
            std::isfinite(xi_bb) && std::isfinite(xi_gg) && std::isfinite(s.xi_bg);
  return s;
}

// Solve Hess Xi . d = -grad J for the step in (beta, gamma), clamping a near-singular Hessian.
//
// ⚠️⚠️ NON-DIMENSIONALISED FIRST, AND THAT IS LOAD-BEARING.  beta = 1/T and gamma = P/T carry
// DIFFERENT DIMENSIONS, so their second derivatives scale differently under a change of units.
// MEASURED on one Al+Cu liner cell (rho 9.396, T 5384 K): Xi_bb/Xi_gg is 7e3 in the canonical
// matdata set and **7e23 in cgs**, which is what FLASH passes.  An eigenvalue floor taken as a
// fraction of the spectral radius then sits ABOVE the genuine small eigenvalue -- 1.16e+04
// against ~1e-10 -- so a LEGITIMATE direction is clamped to something fourteen orders too large,
// the Newton step in it is crushed to nothing, it fails to contract, and the safeguard fires on
// essentially every iteration.  Same cell, same tables, only the units differing:
//
//     canonical :  3 iterations,    8 evaluations, 0 safeguard steps
//     cgs       : 42 iterations, 8326 evaluations, 41 safeguard steps  (and a sibling cell FAILED)
//
// Preconditioning by D = diag(beta, gamma) -- equivalently stepping in (ln beta, ln gamma) --
// makes both variables O(1) and the Hessian O(1), so the method is UNIT-INVARIANT, which a
// physics solver must be.  With it the two unit systems agree to the digit.
//
// PTESolveCyclicRhoE is immune to all of this because its tangents are RATIOS of partials, where
// the dimensional factors cancel.  That is why this bug was invisible until the dual arm ran.
//
// A symmetric 2x2 diagonalises in closed form, so the clamp acts on EIGENVALUES rather than on
// the diagonal.  That distinction still matters near a two-phase plateau: the offending direction
// there is the Clausius-Clapeyron eigenvector, not an axis, and an `H + eps*I` ridge would perturb
// both directions by an amount that means something different in each one's units.
PORTABLE_INLINE_FUNCTION bool NewtonDirection(const DualState &s, const Real beta,
                                              const Real gamma, Real &d_beta, Real &d_gamma) {
  // D = diag(beta, gamma): H' = D H D and grad' = D grad, so the step is d = D d'.
  const Real a = s.xi_bb * beta * beta;
  const Real b = s.xi_bg * beta * gamma;
  const Real c = s.xi_gg * gamma * gamma;
  const Real mean = 0.5 * (a + c);
  const Real radius = std::sqrt(0.25 * (a - c) * (a - c) + b * b);
  const Real lam_hi = mean + radius;
  const Real lam_lo = mean - radius;
  const Real floor = kEigenvalueFloorFraction * std::max(std::abs(lam_hi), std::abs(lam_lo));
  if (!(floor > 0.0)) return false;

  Real vx, vy;
  if (b == 0.0) {
    if (a >= c) { vx = 1.0; vy = 0.0; } else { vx = 0.0; vy = 1.0; }
  } else {
    vx = lam_hi - c;
    vy = b;
    const Real norm = std::sqrt(vx * vx + vy * vy);
    if (!(norm > 0.0)) return false;
    vx /= norm;
    vy /= norm;
  }

  d_beta = 0.0;
  d_gamma = 0.0;
  const Real grad[2] = {s.residual_energy * beta, s.residual_tau * gamma};
  const Real vec[2][2] = {{vx, vy}, {-vy, vx}};
  const Real lam[2] = {lam_hi, lam_lo};
  for (int k = 0; k < 2; ++k) {
    // Negative definite is the physical case; a non-negative eigenvalue means the mixture is
    // locally unstable at this iterate, so reflect rather than divide by it.
    const Real clamped = -std::max(std::abs(lam[k]), floor);
    const Real projection = vec[k][0] * grad[0] + vec[k][1] * grad[1];
    const Real coefficient = -projection / clamped;
    d_beta += coefficient * vec[k][0];
    d_gamma += coefficient * vec[k][1];
  }
  // Back out of the scaled space.
  d_beta *= beta;
  d_gamma *= gamma;
  return std::isfinite(d_beta) && std::isfinite(d_gamma);
}

} // namespace pte_dual_impl

// Solve the ion PTE system for the equilibrated (P, T) of a mixture given (rho, e), by
// safeguarded Newton on (beta, gamma) = (1/T, P/T).
//
// The argument contract is PTESolveCyclicRhoE's, so the two are drop-in alternatives at the call
// site.  On success every output array holds the equilibrium state and `Tguess` the equilibrium
// T; on failure the arrays are left as the caller passed them.  `press` is an in/out warm-start
// seed: a positive entry is used as the first iterate, otherwise the first isochoric bracket
// places it.
//
// Needs DensityEnergyDerivativesFromPressureTemperature in addition to
// DensityEnergyFromPressureTemperature, MinimumPressure, MaximumPressureAtTemperature and
// MinimumTemperature.
template <typename EOSIndexer, typename RealIndexer, typename LambdaIndexer>
PORTABLE_INLINE_FUNCTION SolverStatus
PTESolveDual(const std::size_t nmat, EOSIndexer &&eos, const Real vfrac_tot, const Real sie_tot,
             RealIndexer &&rho, RealIndexer &&vfrac, RealIndexer &&sie, RealIndexer &&temp,
             RealIndexer &&press, LambdaIndexer &&lambda, Real &Tguess, const MixParams &params) {
  using namespace pte_dual_impl;
  SolverStatus status;
  status.converged = false;
  status.residual = std::numeric_limits<Real>::infinity();

  // Positive-form input contract, so a NaN fails it.  Report a failure instead of asserting: a
  // guard cell's volume fraction is not evolved state and AMR prolongation can leave it zero, and
  // one bad cell must not abort the job from inside the EOS.
  if (nmat < 1 || !std::isfinite(sie_tot) || !(Tguess > 0.0) || !(vfrac_tot > 0.0)) return status;
  Real rho_bulk = 0.0;
  for (std::size_t m = 0; m < nmat; ++m) {
    if (!(vfrac[m] > 0.0) || !(rho[m] > 0.0)) return status;
    rho_bulk += vfrac[m] * rho[m];
  }
  if (!(rho_bulk > 0.0)) return status;

  constexpr std::size_t kMaxMat = 16;
  if (nmat > kMaxMat) return status;
  Real Ym[kMaxMat];
  for (std::size_t m = 0; m < nmat; ++m) Ym[m] = vfrac[m] * rho[m] / rho_bulk;
  const Real tau_target = robust::ratio(vfrac_tot, rho_bulk);
  if (!(tau_target > 0.0)) return status;

  // Aggregate EOS box, and the bracket floors, exactly as PTESolveCyclicRhoE derives them -- see
  // the long note there for why the floors must never become the clamp bounds.
  Real p_lo = eos[0].MinimumPressure();
  Real p_hi = eos[0].MaximumPressureAtTemperature(Tguess);
  Real t_lo = eos[0].MinimumTemperature();
  for (std::size_t m = 1; m < nmat; ++m) {
    p_lo = std::max(p_lo, eos[m].MinimumPressure());
    p_hi = std::min(p_hi, eos[m].MaximumPressureAtTemperature(Tguess));
    t_lo = std::min(t_lo, eos[m].MinimumTemperature());
  }
  const Real t_hi = params.temperature_limit;
  if (!(p_hi > 0.0) || !(t_hi > t_lo)) return status;
  const Real p_lo_bracket = std::max(p_lo, 1.0);
  const Real t_lo_bracket = std::max(t_lo, 1.0e-8);
  // The Newton iterate is clamped to a strictly positive box: gamma = P/T is a conjugate
  // variable, not a coordinate that may pass through zero.
  const Real p_lo_clamp = std::max(p_lo, 1.0e-300);
  const Real t_lo_clamp = std::max(t_lo, 1.0e-300);

  auto mix = [&](const Real P, const Real T, Real &tau, Real &energy) {
    MixtureTauE(nmat, eos, Ym, lambda, P, T, tau, energy);
  };
  // Half the squared scaled residual norm: the SUM of squares, not the max of the two.  The max
  // is not differentiable, so an Armijo test on it can fail for every step length even along a
  // genuine descent direction, and the solve then stalls with a finite residual.  The max-norm
  // belongs in the stopping test, where it matches the cyclic arms.
  auto merit_of = [&](const DualState &s) {
    if (!s.valid) return std::numeric_limits<Real>::infinity();
    const Real rt = s.residual_tau / tau_target;
    const Real re = s.residual_energy / (1.0 + std::abs(sie_tot));
    return 0.5 * (rt * rt + re * re);
  };
  auto max_norm_of = [&](const DualState &s) {
    if (!s.valid) return std::numeric_limits<Real>::infinity();
    return std::max(std::abs(s.residual_tau) / tau_target,
                    std::abs(s.residual_energy) / (1.0 + std::abs(sie_tot)));
  };

  Real T = std::min(std::max(Tguess, t_lo_clamp), t_hi);
  Real P = 0.0;
  for (std::size_t m = 0; m < nmat; ++m) {
    if (press[m] > 0.0) { P = press[m]; break; }
  }
  if (!(P > 0.0)) {
    const auto tau_resid = [&](const Real p) {
      Real tau, energy;
      mix(p, T, tau, energy);
      return tau - tau_target;
    };
    if (!LogBracketRoot(p_lo_bracket, p_hi, tau_resid, P)) {
      std::size_t idom = 0;
      for (std::size_t m = 1; m < nmat; ++m)
        if (Ym[m] > Ym[idom]) idom = m;
      P = eos[idom].PressureFromDensityTemperature(rho[idom], T);
    }
  }
  P = std::min(std::max(P, p_lo_clamp), p_hi);

  const Real tol =
      std::max(std::min(params.pte_rel_tolerance_v, params.pte_rel_tolerance_e), 1.0e-14);
  const std::size_t max_iter =
      (params.pte_max_iter_per_mat > 0) ? params.pte_max_iter_per_mat * nmat : 100;

  Real beta = 1.0 / T;
  Real gamma = P / T;
  // Per-material state carried out of the last accepted evaluation, so the finalize block never
  // re-queries the table.
  Real rho_keep[kMaxMat], sie_keep[kMaxMat], rho_try[kMaxMat], sie_try[kMaxMat];

  // CHEAP PROBE FIRST.  In FLASH's bulk regime the warm start from the previous step is already
  // the answer -- the cyclic arm measures 0.24 iterations per call there -- so most solves need
  // no Hessian at all.  Assembling one before testing convergence made every such cell pay the
  // derivative accessor's premium over the plain one for nothing.  A converged probe exits here
  // having spent exactly `nmat` plain lookups, against the cyclic arm's 2*nmat (its own probe
  // plus its finalize).
  {
    Real tau_probe = 0.0, e_probe = 0.0;
    for (std::size_t m = 0; m < nmat; ++m) {
      Real rho_m, sie_m;
      eos[m].DensityEnergyFromPressureTemperature(P, T, lambda[m], rho_m, sie_m);
      if (!(rho_m > 0.0) || !std::isfinite(sie_m)) {
        tau_probe = std::numeric_limits<Real>::quiet_NaN();
        break;
      }
      rho_keep[m] = rho_m;
      sie_keep[m] = sie_m;
      tau_probe += Ym[m] * robust::ratio(1.0, rho_m);
      e_probe += Ym[m] * sie_m;
    }
    if (std::isfinite(tau_probe)) {
      const Real rt = (tau_probe - tau_target) / tau_target;
      const Real re = (e_probe - sie_tot) / (1.0 + std::abs(sie_tot));
      if (std::max(std::abs(rt), std::abs(re)) <= tol) {
        for (std::size_t m = 0; m < nmat; ++m) {
          rho[m] = rho_keep[m];
          sie[m] = sie_keep[m];
          temp[m] = T;
          press[m] = P;
          vfrac[m] = Ym[m] * rho_bulk / rho_keep[m];
        }
        Tguess = T;
        status.converged = true;
        status.residual = std::max(std::abs(rt), std::abs(re));
        status.max_niter = 0;
        return status;
      }
    }
  }

  DualState state =
      MixtureDual(nmat, eos, Ym, lambda, tau_target, sie_tot, P, T, rho_keep, sie_keep);
  Real merit = merit_of(state);
  Real max_asymmetry = state.valid ? state.asymmetry : 0.0;
  std::size_t iters = 0;
  std::size_t safeguard_steps = 0;

  while (iters < max_iter && max_norm_of(state) > tol) {
    ++iters;
    Real d_beta, d_gamma;
    if (!state.valid || !NewtonDirection(state, beta, gamma, d_beta, d_gamma)) break;

    // Multiplicative trust region, applied in BOTH directions.  Capping only the downward move
    // (to keep the quadrant) leaves the upward one free, and an unbounded Newton step from a cold
    // seed lands decades outside the table where the model that produced it means nothing.
    Real alpha = 1.0;
    const Real value[2] = {beta, gamma};
    const Real delta[2] = {d_beta, d_gamma};
    for (int k = 0; k < 2; ++k) {
      if (delta[k] < 0.0) {
        alpha = std::min(alpha, (1.0 - kPositivityRetention) * value[k] / -delta[k]);
      } else if (delta[k] > 0.0) {
        alpha = std::min(alpha, (kMaxStepRatio - 1.0) * value[k] / delta[k]);
      }
    }

    bool accepted = false;
    for (int back = 0; back < kMaxBacktracks; ++back) {
      const Real tb = beta + alpha * d_beta;
      const Real tg = gamma + alpha * d_gamma;
      if (!(tb > 0.0) || !(tg > 0.0)) { alpha *= 0.5; continue; }
      const Real trial_T = std::min(std::max(1.0 / tb, t_lo_clamp), t_hi);
      const Real trial_P = std::min(std::max(tg / tb, p_lo_clamp), p_hi);
      const DualState trial = MixtureDual(nmat, eos, Ym, lambda, tau_target, sie_tot, trial_P,
                                          trial_T, rho_try, sie_try);
      const Real trial_merit = merit_of(trial);
      // Demand a real CONTRACTION, not merely Armijo descent: the alternative is an exact
      // maximisation in one coordinate, so a Newton step that only inches is not worth taking.
      if (std::isfinite(trial_merit) && trial_merit < kNewtonContraction * merit) {
        // Re-derive the conjugate pair from the CLAMPED state, so the iterate the solver carries
        // is the one the EOS was actually evaluated at.
        beta = 1.0 / trial_T;
        gamma = trial_P * beta;
        state = trial;
        merit = trial_merit;
        max_asymmetry = std::max(max_asymmetry, trial.asymmetry);
        for (std::size_t m = 0; m < nmat; ++m) {
          rho_keep[m] = rho_try[m];
          sie_keep[m] = sie_try[m];
        }
        accepted = true;
        break;
      }
      alpha *= 0.5;
    }
    if (accepted) continue;

    // Safeguard: one block-coordinate maximisation of the same J, by bracketed bisection.  Both
    // roots exist and are unique because each residual is strictly monotone in its own coordinate
    // (K_T > 0 and c_v > 0).  Aim the sub-solves at the residual still to be removed, as
    // PTESolveCyclicRhoE does; an exact coordinate root is waste while the other is far off.
    const Real sub_tol = std::min(std::max(1.0e-3 * std::sqrt(2.0 * merit), 1.0e-12), 1.0e-3);
    Real step_P = gamma / beta;
    Real step_T = 1.0 / beta;
    const auto tau_resid = [&](const Real p) {
      Real tau, energy;
      mix(p, step_T, tau, energy);
      return tau - tau_target;
    };
    Real root;
    if (LogBracketRoot(p_lo_bracket, p_hi, tau_resid, root, sub_tol)) step_P = root;
    const auto e_resid = [&](const Real t) {
      Real tau, energy;
      mix(step_P, t, tau, energy);
      return energy - sie_tot;
    };
    if (LogBracketRoot(t_lo_bracket, t_hi, e_resid, root, sub_tol)) step_T = root;
    step_P = std::min(std::max(step_P, p_lo_clamp), p_hi);
    step_T = std::min(std::max(step_T, t_lo_clamp), t_hi);

    // ---- Tangent-intersection acceleration: PTESolveCyclicRhoE's step 3 ----
    //
    // ⚠️ WITHOUT THIS THE SAFEGUARD IS STRICTLY WEAKER THAN THE METHOD IT BORROWS FROM, and the
    // whole solver loses its worst-case guarantee.  The cyclic step is two bracketed sub-solves
    // PLUS this intersection; taking only the sub-solves converges markedly slower on exactly the
    // cells that need the safeguard.  MEASURED in 1D maglif with it omitted: 14.96 mean
    // iterations per PTE call against the cyclic arm's 3.04, i.e. the dual was worse than plain
    // cyclic on any cell where Newton was not carrying the solve -- while also paying a rejected
    // Newton trial each iteration.  With it, the dual's worst case IS the cyclic step and Newton
    // can only add.
    //
    // Both slopes are RATIOS of mixture partials, which is what survives a two-phase plateau:
    // d_P tau and d_T tau both diverge there but their ratio tends to the Clausius-Clapeyron
    // slope dP_sat/dT.  The reciprocal form for the isoenergetic tangent keeps the ideal-gas
    // limit (d_P e = 0, a vertical line) exact instead of a division by zero.
    //
    // This costs 2 derivative evaluations per material where PTESolveCyclicRhoE spends 8 plain
    // ones on central differences -- the partials are already assembled here.
    {
      const DualState at_p =
          MixtureDual(nmat, eos, Ym, lambda, tau_target, sie_tot, step_P, 1.0 / beta, nullptr,
                      nullptr);
      const DualState at_t =
          MixtureDual(nmat, eos, Ym, lambda, tau_target, sie_tot, gamma / beta, step_T, nullptr,
                      nullptr);
      if (at_p.valid && at_t.valid) {
        const Real m_rho = (at_p.dtau_dP != 0.0) ? -at_p.dtau_dT / at_p.dtau_dP : 0.0;
        const Real inv_m_e = (at_t.de_dT != 0.0) ? -at_t.de_dP / at_t.de_dT : 0.0;
        const Real ab = m_rho * inv_m_e;
        if (std::isfinite(ab) && std::abs(1.0 - ab) > 1.0e-300) {
          const Real T_now = 1.0 / beta, P_now = gamma / beta;
          const Real t_int = (step_T + inv_m_e * (step_P - P_now) - ab * T_now) / (1.0 - ab);
          const Real p_int = step_P + m_rho * (t_int - T_now);
          const Real p_c = std::min(std::max(p_int, p_lo_clamp), p_hi);
          const Real t_c = std::min(std::max(t_int, t_lo_clamp), t_hi);
          if (std::isfinite(p_c) && std::isfinite(t_c)) {
            Real tau_i, e_i;
            mix(p_c, t_c, tau_i, e_i);
            if (std::isfinite(tau_i)) {
              const Real r_int = std::max(std::abs(tau_i - tau_target) / tau_target,
                                          std::abs(e_i - sie_tot) / (1.0 + std::abs(sie_tot)));
              Real tau_s, e_s;
              mix(step_P, step_T, tau_s, e_s);
              const Real r_sub = std::isfinite(tau_s)
                                     ? std::max(std::abs(tau_s - tau_target) / tau_target,
                                                std::abs(e_s - sie_tot) / (1.0 + std::abs(sie_tot)))
                                     : std::numeric_limits<Real>::infinity();
              // Take the intersection only when it beats the plain sub-solve pair, exactly as
              // the cyclic arm does; otherwise keep (p_tilde, t_tilde).
              if (r_int < r_sub) {
                step_P = p_c;
                step_T = t_c;
              }
            }
          }
        }
      }
    }

    // Accept it UNCONDITIONALLY when it is finite and it moved.  It always increases J, but it
    // may raise ||R|| on the way, and gating it on the residual throws away exactly the property
    // being borrowed (measured: 1.6% of seeds converged with such a gate, 65.6% without).
    const bool moved = (step_P != gamma / beta) || (step_T != 1.0 / beta);
    const DualState stepped = MixtureDual(nmat, eos, Ym, lambda, tau_target, sie_tot, step_P,
                                          step_T, rho_try, sie_try);
    const Real stepped_merit = merit_of(stepped);
    if (!moved || !std::isfinite(stepped_merit)) break;
    beta = 1.0 / step_T;
    gamma = step_P * beta;
    state = stepped;
    merit = stepped_merit;
    max_asymmetry = std::max(max_asymmetry, stepped.asymmetry);
    for (std::size_t m = 0; m < nmat; ++m) {
      rho_keep[m] = rho_try[m];
      sie_keep[m] = sie_try[m];
    }
    ++safeguard_steps;
  }

  T = 1.0 / beta;
  P = gamma * T;
  const Real residual = max_norm_of(state);
  status.residual = residual;
  status.max_niter = iters;
  status.safeguard_iters = safeguard_steps;
  status.converged = std::isfinite(residual) && residual <= tol;
  if (status.converged) {
    // From the CACHE: `state` was produced by an evaluation at exactly this (P, T), so re-querying
    // would buy nothing but another nmat lookups.
    for (std::size_t m = 0; m < nmat; ++m) {
      const Real rho_m = rho_keep[m], sie_m = sie_keep[m];
      if (!(rho_m > 0.0) || !std::isfinite(sie_m)) {
        status.converged = false;
        return status;
      }
      rho[m] = rho_m;
      sie[m] = sie_m;
      temp[m] = T;
      press[m] = P;
      // Sums to vfrac_tot when the volume closes, which `converged` guarantees.
      vfrac[m] = Ym[m] * rho_bulk / rho_m;
    }
    Tguess = T;
  }
  return status;
}

} // namespace singularity

#endif // _SINGULARITY_EOS_CLOSURE_PTE_DUAL_
