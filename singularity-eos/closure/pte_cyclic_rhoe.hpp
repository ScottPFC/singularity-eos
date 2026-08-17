//------------------------------------------------------------------------------
// Cyclic PTE solve in the (rho, e) form -- standalone, seed-independent.
//------------------------------------------------------------------------------
// Deliberately NOT built on PTESolverBase/PTESolverPT like PTESolverPTCyclic is.  It needs only
// the (P,T) -> (rho, e) primitive and mass fractions, so it carries no scratch array, no scaled
// (Pequil/Tequil) state, no Residual()/CheckPTE() bookkeeping and no Init().  That is a
// measured choice, not a stylistic one: an inheriting version of this same algorithm cost 7x the
// base cyclic per cell and lost cells the base kept, because every iteration paid for machinery
// the method does not use.  Keep it minimal; add scaffolding only when a measurement asks for it.
//
// THE METHOD.  Same cyclic structure as Clayton-McConnell-Solomon Alg 5.2 (PTESolverPTCyclic),
// with two changes, both aimed at a tabular EOS whose box spans ~18 decades in P:
//
//   1. The sub-steps are BRACKETED 1-D solves in LOG space rather than single Newton steps -- an
//      isochoric solve for P~ with tau(P~, T^n) = tau0 and an isoenergetic solve for T~ with
//      e(P^n, T~) = e0.  An iterate cannot leave the table box, and the tolerance is relative.
//      Log space is load-bearing: an absolute bracket tolerance taken from the box width is
//      larger than the root itself in the dilute corner (measured on a DD/TT/Cu interface cell,
//      (Phi-Plo)*1e-14 = 2.5e-6 GPa against a P* of 4.0e-4, which stalls the mixture volume
//      residual at 2.8e-4 while (P,T) looks converged).  Exhaustive bisection rather than a
//      seeded Newton off the analytic derivative: Newton is ~3 evaluations against ~52 and the
//      saving is real, but it collapses the stale-seed basin (bench_pte_solve.cpp on Al+Cu at
//      staleness 10: 17070/20000 -> 209/20000).  The bracket IS the robustness.
//   2. The second tangent is the ISOENERGETIC curve (dP/dT)_e = -(d_T e)/(d_P e) at (P^n, T~)
//      instead of the isentrope, so with (rho, e) as the input pair each sub-solve owns exactly
//      one conserved quantity.  Stored as the reciprocal dT/dP = -(d_P e)/(d_T e) so the
//      ideal-gas limit (d_P e = 0, a vertical isoenergetic line T = T~) is exact rather than a
//      division by zero.
//
// Both tangent slopes are RATIOS of mixture partials, which is what survives a two-phase
// plateau: d_P tau and d_T tau both diverge there but their ratio stays finite and tends to the
// Clausius-Clapeyron slope dP_sat/dT, where a 2-D Newton determinant (rho c_v K_T) vanishes.
//
// FAILURE POLICY.  It stops on the first iterate that does not reduce the residual and reports
// converged = false, rather than grinding to the iteration cap.  A caller with a fallback ladder
// (FLASH's eos_singularity has three rungs) is far better served by a fast negative, and the
// rungs are cheap; a cell that cannot converge otherwise burns the full cap at ~1.4x a base
// cyclic iteration each.  Measured end to end in FLASH (RUN143 vs RUN134, identical tables and
// initial condition, plain grid4 fuel with its two-phase tie-line intact): 0.114 s/step against
// 0.556, with the SAME physics -- fusion yield within 0.02%, bang time equal to 1e-4 ns, chi and
// step count identical.  Offline (pfc `matdata invert pte-stress`, Python twin
// pfc/sim/matdata/_pte.py::_cyclic_rhoe_step): 256/256 seeds over the box on three mixture cells
// against the base cyclic's 36/256, 83/256 and 0/256.
//
// Units are the caller's, and must be self-consistent (FLASH passes cgs): P in barye, T in K,
// rho in g/cm^3, e in erg/g.
//------------------------------------------------------------------------------

#ifndef _SINGULARITY_EOS_CLOSURE_PTE_CYCLIC_RHOE_
#define _SINGULARITY_EOS_CLOSURE_PTE_CYCLIC_RHOE_

#include <cmath>
#include <limits>

#include <ports-of-call/portability.hpp>
#include <ports-of-call/portable_errors.hpp>
#include <singularity-eos/base/robust_utils.hpp>
#include <singularity-eos/closure/mixed_cell_models.hpp> // MixParams, SolverStatus

namespace singularity {

namespace cyclic_rhoe_impl {

// Mixture specific volume and energy at (P, T), mass-fraction weighted.  Non-finite or
// non-positive per-material density poisons both, so the caller can test one value.
template <typename EOSIndexer, typename RealIndexer, typename LambdaIndexer>
PORTABLE_INLINE_FUNCTION void MixtureTauE(const std::size_t nmat, EOSIndexer &eos,
                                          const RealIndexer &Ym, LambdaIndexer &lambda,
                                          const Real P, const Real T, Real &tau, Real &energy) {
  tau = 0.0;
  energy = 0.0;
  for (std::size_t m = 0; m < nmat; ++m) {
    Real rho_m, sie_m;
    eos[m].DensityEnergyFromPressureTemperature(P, T, lambda[m], rho_m, sie_m);
    if (!(rho_m > 0.0) || !std::isfinite(sie_m)) {
      tau = std::numeric_limits<Real>::quiet_NaN();
      energy = std::numeric_limits<Real>::quiet_NaN();
      return;
    }
    tau += Ym[m] * robust::ratio(1.0, rho_m);
    energy += Ym[m] * sie_m;
  }
}

// Bracketed root of a monotone residual on [lo, hi] by bisection in log space.  False when the
// endpoints do not straddle zero, so the caller keeps its current iterate rather than stepping
// outside the table.
//
// `rel_tol` is the bracket width ratio to stop at.  Resolving to machine precision is waste: the
// tangent intersection in step 3 supersedes this answer, so the sub-solve only has to aim it.
// Bisection halves the log bracket per evaluation, so each decade of tolerance is ~3.3 evaluations
// and the brackets are essentially the entire cost of the method -- measured on the RUN102
// interface cell in the Python twin, tying the tolerance to the residual cut EOS evaluations per
// solve 3023 -> 1591 (-47%) at unchanged 100% seed robustness.
template <typename Residual>
PORTABLE_INLINE_FUNCTION bool LogBracketRoot(const Real lo_in, const Real hi_in,
                                             const Residual &resid, Real &root,
                                             const Real rel_tol = 1.0e-14) {
  Real lo = lo_in, hi = hi_in;
  if (!(hi > lo && lo > 0.0)) return false;
  Real f_lo = resid(lo);
  const Real f_hi = resid(hi);
  if (!std::isfinite(f_lo) || !std::isfinite(f_hi)) return false;
  if (f_lo == 0.0) { root = lo; return true; }
  if (f_hi == 0.0) { root = hi; return true; }
  if (f_lo * f_hi > 0.0) return false;
  for (int it = 0; it < 200; ++it) {
    const Real mid = std::sqrt(lo * hi);
    const Real f_mid = resid(mid);
    if (!std::isfinite(f_mid)) return false;
    if (f_mid == 0.0) { root = mid; return true; }
    if ((f_mid > 0.0) == (f_lo > 0.0)) { lo = mid; f_lo = f_mid; } else { hi = mid; }
    if (hi / lo < 1.0 + rel_tol) break;
  }
  root = std::sqrt(lo * hi);
  return true;
}

} // namespace cyclic_rhoe_impl

// Solve the ion PTE system for the equilibrated (P, T) of a mixture given (rho, e).
//
// On success every output array holds the equilibrium state and `Tguess` the equilibrium T; on
// failure the arrays are left as the caller passed them (the caller's fallback ladder owns the
// cell).  `press` is an in/out warm-start seed: a positive entry is used as the first iterate,
// otherwise the first isochoric bracket places it.
//
// Args:
//   nmat: number of active materials.
//   eos: per-material EOS indexer; needs DensityEnergyFromPressureTemperature, MinimumPressure,
//     MaximumPressureAtTemperature and MinimumTemperature.
//   vfrac_tot: total volume the cell holds (tau = vfrac_tot / rho_bulk).
//   sie_tot: mixture specific internal energy target.
//   rho, vfrac: in as the caller's guess (they set rho_bulk and the mass fractions), out as the
//     equilibrium values.
//   sie, temp, press: out (press also in, as the seed).
//   Tguess: in/out temperature guess / equilibrium temperature.
//   params: MixParams; only the volume/energy relative tolerances and the iteration cap are used.
template <typename EOSIndexer, typename RealIndexer, typename LambdaIndexer>
PORTABLE_INLINE_FUNCTION SolverStatus
PTESolveCyclicRhoE(const std::size_t nmat, EOSIndexer &&eos, const Real vfrac_tot,
                   const Real sie_tot, RealIndexer &&rho, RealIndexer &&vfrac,
                   RealIndexer &&sie, RealIndexer &&temp, RealIndexer &&press,
                   LambdaIndexer &&lambda, Real &Tguess, const MixParams &params) {
  using namespace cyclic_rhoe_impl;
  SolverStatus status;
  status.converged = false;
  status.residual = std::numeric_limits<Real>::infinity();

  // Positive-form input contract, so a NaN fails it.  Report a failure instead of asserting: a
  // guard cell's volume fraction is not evolved state and AMR prolongation can leave it zero,
  // and one bad cell must not abort the job from inside the EOS.
  if (nmat < 1 || !std::isfinite(sie_tot) || !(Tguess > 0.0) || !(vfrac_tot > 0.0)) {
    return status;
  }
  Real rho_bulk = 0.0;
  for (std::size_t m = 0; m < nmat; ++m) {
    if (!(vfrac[m] > 0.0) || !(rho[m] > 0.0)) return status;
    rho_bulk += vfrac[m] * rho[m];
  }
  if (!(rho_bulk > 0.0)) return status;

  // Mass fractions, stack-allocated: nmat is a species count, so a fixed small bound avoids the
  // scratch-array plumbing the inheriting solvers need.
  constexpr std::size_t kMaxMat = 16;
  if (nmat > kMaxMat) return status;
  Real Ym[kMaxMat];
  for (std::size_t m = 0; m < nmat; ++m) Ym[m] = vfrac[m] * rho[m] / rho_bulk;
  const Real tau_target = robust::ratio(vfrac_tot, rho_bulk);

  // Aggregate EOS box.  A log bracket cannot start at zero and TableDependsRhoT reports
  // MinimumPressure() == 0 exactly (deliberately, so a clamp cannot drive a condensed material
  // into tension), so floor the low ends at a negligible fraction of the box.
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
  // A log bracket cannot start at zero, so floor both low ends -- but with ABSOLUTE floors, not
  // fractions of the box.  `temperature_limit` is 1e15 K, so a `t_hi * 1e-12` floor would put the
  // temperature bracket's low end at 1000 K and silently exclude every cryogenic solution: the
  // bracket then shows no sign change, step 2 does nothing on every cell, and the solve limps
  // (measured: RUN146 0.36-0.83 s/step against RUN143's 0.114 for the same algorithm).  The same
  // trap on the pressure axis is why `MinimumPressure()`, which TableDependsRhoT reports as
  // exactly 0, cannot be used as the bracket's low end either.
  // A log bracket cannot start at zero, and both axes report a zero-ish low end: TableDependsRhoT
  // gives MinimumPressure() == 0 exactly (deliberately, so a clamp cannot drive a condensed
  // material into tension) and a temperature_offset table's grid reaches T = 0.  So the BRACKETS
  // get floored -- but ONLY the brackets.  The floor must never become the CLAMP bound: doing that
  // puts the solution outside the search domain rather than merely outside the bracket, and then
  // nothing recovers it (measured: with the temperature floor above the solution, the solve
  // converges on 0 of 20000 bench cells whether or not the Newton fallback is enabled).  Keeping
  // them separate is why the Python twin (pfc _pte.py::_cyclic_rhoe_step) survived the same
  // mis-floored bracket that cost the C++ 0.890 s/step against 0.117.
  //
  // 1 barye (1e-10 GPa) is the validated pressure floor: below any equilibrium pressure a mixture
  // reaches, and it keeps the bracket ~22 decades rather than the ~48 a token epsilon would give.
  const Real p_lo_bracket = std::max(p_lo, 1.0);
  const Real t_lo_bracket = std::max(t_lo, 1.0e-8);

  auto mix = [&](const Real P, const Real T, Real &tau, Real &energy) {
    MixtureTauE(nmat, eos, Ym, lambda, P, T, tau, energy);
  };
  auto residual_of = [&](const Real P, const Real T) {
    Real tau, energy;
    mix(P, T, tau, energy);
    if (!std::isfinite(tau)) return std::numeric_limits<Real>::infinity();
    return std::max(std::abs(tau - tau_target) / tau_target,
                    std::abs(energy - sie_tot) / (1.0 + std::abs(sie_tot)));
  };

  Real T = std::min(std::max(Tguess, t_lo), t_hi);
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
  P = std::min(std::max(P, p_lo), p_hi);

  const Real tol =
      std::max(std::min(params.pte_rel_tolerance_v, params.pte_rel_tolerance_e), 1.0e-14);
  const std::size_t max_iter =
      (params.pte_max_iter_per_mat > 0) ? params.pte_max_iter_per_mat * nmat : 100;

  Real residual = residual_of(P, T);
  std::size_t iters = 0;
  for (; iters < max_iter && residual > tol; ++iters) {
    // Aim the sub-solves at the residual still to be removed, not at machine precision.  Floored
    // so the last iteration still resolves finely enough to converge, capped so the first one
    // cannot bracket uselessly coarsely.
    Real tau_now, e_now;
    mix(P, T, tau_now, e_now);
    const Real tau_err = std::abs(tau_now - tau_target) / tau_target;
    const Real e_err = std::abs(e_now - sie_tot) / (1.0 + std::abs(sie_tot));
    // The FLOOR is the dominant cost knob, not the 1e-3 factor.  LogBracketRoot halves the LOG
    // bracket per evaluation, so a bracket spanning D decades costs log2(ln(10^D)/ln(1+tol))
    // evaluations: over the ~22-decade pressure bracket that is ~16 at 1e-3, ~32 at 1e-8 and ~52
    // at 1e-14.  Cost therefore GROWS as the outer solve converges and the floor takes over --
    // which is backwards, since the comment on LogBracketRoot is right that step 3's tangent
    // supersedes this answer and the sub-solve only has to AIM it.
    //
    // MEASURED, AND THE FLOOR IS NOT THE LEVER (bench_pte_solve, CyclicRhoEStep, 20000 cells):
    //   floor  1e-14 -> 250.7 roots/cell, 48.8 us, 19988 converged   (current)
    //          1e-10 -> 250.7             48.9     19988
    //          1e-8  -> 247.1             48.2     19988
    //          1e-6  -> 229.9             45.1     19987
    // Only -8% at 1e-6.  The floor binds only where 1e-3*err < floor, i.e. err < 1e-11, which is
    // the LAST outer iteration -- and the outer loop runs ~1.5 iterations, so it barely fires.
    // The cost sits at the COARSE end (sub_tol capped at 1e-3, ~16 evaluations per bracket) and
    // is therefore set by bisection's one-bit-per-evaluation RATE, not by the tolerance.  Making
    // it cheaper means a superlinear bracketed method (Illinois/Ridders) at the SAME stopping
    // width -- which is the change ca038da4b reverted, on the grounds that the sub-solve's
    // coarseness was acting as DAMPING on the outer iteration.  Measure outer iterations and the
    // converged count, not just wall time, before believing any replacement.
    //
    // Kept as a build knob because raising it is the one direction that is safe by construction
    // (coarser, never sharper); the default is unchanged.
#ifndef SG_PTE_RHOE_SUBTOL_FLOOR
#define SG_PTE_RHOE_SUBTOL_FLOOR 1.0e-14
#endif
    const Real sub_tol = std::min(
        std::max(1.0e-3 * std::max(tau_err, e_err), Real(SG_PTE_RHOE_SUBTOL_FLOOR)), 1.0e-3);

    // Step 1: isochoric sub-solve for P~ at fixed T^n.
    Real p_tilde = P;
    const auto tau_resid = [&](const Real p) {
      Real tau, energy;
      mix(p, T, tau, energy);
      return tau - tau_target;
    };
    bool p_bracketed = LogBracketRoot(p_lo_bracket, p_hi, tau_resid, p_tilde, sub_tol);

    // Step 2: isoenergetic sub-solve for T~ at fixed P^n.
    Real t_tilde = T;
    const auto e_resid = [&](const Real t) {
      Real tau, energy;
      mix(P, t, tau, energy);
      return energy - sie_tot;
    };
    bool t_bracketed = LogBracketRoot(t_lo_bracket, t_hi, e_resid, t_tilde, sub_tol);

    // A bracket that finds no sign change must DEGRADE, not vanish.  Leaving the sub-step as a
    // no-op is what turned a mis-floored bracket into a 7x slowdown (FLASH: 0.890 s/step against
    // 0.117 for the same algorithm) -- the tangent alone cannot carry the iteration.  One clamped
    // Newton step off a finite-difference slope costs two extra evaluations on the failure path
    // only, and it is what the Python twin (pfc _pte.py::_cyclic_rhoe_step) already did, which is
    // why the same mistake was invisible there.
    const Real hf = 1.0e-4;
    if (!p_bracketed) {
      const Real f0 = tau_resid(P);
      const Real fp = tau_resid(P * (1.0 + hf));
      const Real slope = (fp - f0) / (hf * P);
      if (slope != 0.0 && std::isfinite(slope) && std::isfinite(f0)) {
        p_tilde = std::min(std::max(P - f0 / slope, p_lo), p_hi);
      }
    }
    if (!t_bracketed) {
      const Real g0 = e_resid(T);
      const Real gp = e_resid(T * (1.0 + hf));
      const Real slope = (gp - g0) / (hf * T);
      if (slope != 0.0 && std::isfinite(slope) && std::isfinite(g0)) {
        t_tilde = std::min(std::max(T - g0 / slope, t_lo), t_hi);
      }
    }

    // Step 3: tangent slopes by central differences in log space (the tables are log-gridded,
    // so a relative step is the natural one) and the closed-form intersection.  Eight extra
    // lookups against the ~100 the two brackets cost, so the analytic-derivative accessor is
    // not worth the coupling here.
    const Real h = 1.0e-4;
    Real tau_pp, tau_pm, tau_tp, tau_tm, e_unused;
    mix(p_tilde * (1.0 + h), T, tau_pp, e_unused);
    mix(p_tilde * (1.0 - h), T, tau_pm, e_unused);
    mix(p_tilde, T * (1.0 + h), tau_tp, e_unused);
    mix(p_tilde, T * (1.0 - h), tau_tm, e_unused);
    Real e_pp, e_pm, e_tp, e_tm, tau_unused;
    mix(P * (1.0 + h), t_tilde, tau_unused, e_pp);
    mix(P * (1.0 - h), t_tilde, tau_unused, e_pm);
    mix(P, t_tilde * (1.0 + h), tau_unused, e_tp);
    mix(P, t_tilde * (1.0 - h), tau_unused, e_tm);

    const Real dtau_dp = (tau_pp - tau_pm) / (2.0 * h * p_tilde);
    const Real dtau_dt = (tau_tp - tau_tm) / (2.0 * h * T);
    const Real de_dp = (e_pp - e_pm) / (2.0 * h * P);
    const Real de_dt = (e_tp - e_tm) / (2.0 * h * t_tilde);

    Real p_next = p_tilde, t_next = t_tilde;
    const Real m_rho = (dtau_dp != 0.0) ? -dtau_dt / dtau_dp : 0.0;      // (dP/dT)_tau
    const Real inv_m_e = (de_dt != 0.0) ? -de_dp / de_dt : 0.0;          // (dT/dP)_e
    const Real ab = m_rho * inv_m_e;
    if (std::isfinite(ab) && std::abs(1.0 - ab) > 1.0e-300) {
      // Line A: P = P~ + m_rho (T - T^n);  line B: T = T~ + (1/m_e)(P - P^n).
      const Real t_int = (t_tilde + inv_m_e * (p_tilde - P) - ab * T) / (1.0 - ab);
      const Real p_int = p_tilde + m_rho * (t_int - T);
      if (std::isfinite(t_int) && std::isfinite(p_int) && p_int >= p_lo && p_int <= p_hi &&
          t_int >= t_lo && t_int <= t_hi) {
        p_next = p_int;
        t_next = t_int;
      }
    }
    const Real p_new = std::min(std::max(p_next, p_lo), p_hi);
    const Real t_new = std::min(std::max(t_next, t_lo), t_hi);
    const Real residual_new = residual_of(p_new, t_new);
    // Accept only a non-improving iterate; the sub-solves alone are a contraction, so rejecting
    // a bad tangent cannot stall the way an unguarded Newton step can.  If neither the tangent
    // nor the sub-solve pair improves, stop and let the caller's ladder take the cell.
    if (!(residual_new < residual)) {
      const Real r_tilde = residual_of(p_tilde, t_tilde);
      if (r_tilde < residual) {
        P = p_tilde;
        T = t_tilde;
        residual = r_tilde;
        continue;
      }
      P = p_new;
      T = t_new;
      residual = residual_new;
      ++iters;
      break;
    }
    P = p_new;
    T = t_new;
    residual = residual_new;
  }

  // ---- LAST-RESORT RESCUE: MINIMISE the residual where the brackets cannot ROOT it ----
  //
  // Reached ONLY after the loop above has stopped improving without converging, so a converging
  // cell never evaluates any of this and its answer is bit-identical.
  //
  // WHY A MINIMISER AND NOT A BETTER ROOT FIND.  Both sub-solves are BRACKETED, and a bracket
  // needs a SIGN CHANGE.  On a cold mixed cell the mixture energy along the tau-closing curve
  // can be NON-MONOTONE in T, dipping to a minimum that lies a hair BELOW the target -- measured
  // on a RUN252 Cu/DD/TT cell (rho 0.50138, sie_tot -1.9466e9 erg/g, table frame) the minimum
  // sits 8.56e3 erg/g below target, 4.4e-6 relative:
  //
  //     T (K)      0.01        0.5         1.0          2.0         5.0
  //     sie     -1.9398e9  -1.9440e9  -1.9466e9    -1.9173e9   -1.7741e9
  //     gap     +6.82e+06  +2.60e+06  -8.56e+03    +2.93e+07   +1.73e+08
  //
  // So the energy residual is very nearly TANGENT to zero, with two roots almost coincident, and
  // there is no sign change for LogBracketRoot to find.  The no-bracket Newton fallback then
  // measures a ~0 slope at that minimum and does not move: MEASURED, sweeping the seed over
  // 1..2000 K, T came back EQUAL to its seed in all ten cases with the residual stuck at 3.4e-3.
  //
  // A minimiser needs only unimodality, not a sign change, so the tangency cannot defeat it.
  // MEASURED on that state, identical on the V14 and V17 table sets: residual 2.23e-16 against a
  // 1e-6 bar, tau to 2.2e-16 and sie to 0.0 relative, at T = 0.998 K.
  //
  // NOTE the state has TWO genuine PTE roots -- PTESolverPTCyclic reaches T = 0.9979 from a 1 K
  // seed and T = 1.3906 from 75 K, both to ~1e-13.  This lands on the lower one.  That
  // non-uniqueness is a property of the state, not of the method.
#ifndef SG_PTE_RHOE_MINIMISER_RESCUE
#define SG_PTE_RHOE_MINIMISER_RESCUE 1
#endif
#ifndef SG_PTE_RHOE_RESCUE_ITERS
#define SG_PTE_RHOE_RESCUE_ITERS 60
#endif
  if (SG_PTE_RHOE_MINIMISER_RESCUE && !(std::isfinite(residual) && residual <= tol)) {
    // tau is monotone in P, so unlike the energy bracket this one cannot fail to find its root.
    const auto close_tau = [&](const Real T_at) {
      Real lo = p_lo_bracket, hi = p_hi, tau, energy;
      for (int i = 0; i < 64; ++i) {
        const Real pm = std::sqrt(lo * hi);
        mix(pm, T_at, tau, energy);
        if (!std::isfinite(tau)) {
          hi = pm;
          continue;
        }
        if (tau > tau_target) lo = pm;
        else hi = pm;
        if (hi <= lo * (1.0 + 1.0e-13)) break;
      }
      return std::sqrt(lo * hi);
    };
    // Reuse the SAME residual the loop above judges itself by, so "improved" means one thing.
    const auto f_of_t = [&](const Real T_at) { return residual_of(close_tau(T_at), T_at); };

    const Real gr = 0.6180339887498949; // (sqrt(5) - 1) / 2
    Real a = std::log(t_lo_bracket), b = std::log(t_hi);
    Real c = b - gr * (b - a), d = a + gr * (b - a);
    Real fc = f_of_t(std::exp(c)), fd = f_of_t(std::exp(d));
    for (int i = 0; i < SG_PTE_RHOE_RESCUE_ITERS && (b - a) > 1.0e-12; ++i) {
      if (fc < fd) {
        b = d;
        d = c;
        fd = fc;
        c = b - gr * (b - a);
        fc = f_of_t(std::exp(c));
      } else {
        a = c;
        c = d;
        fc = fd;
        d = a + gr * (b - a);
        fd = f_of_t(std::exp(d));
      }
    }
    const Real t_star = std::exp(0.5 * (a + b));
    const Real p_star = close_tau(t_star);
    const Real r_star = residual_of(p_star, t_star);
    // Accept only an improvement: the rescue must never leave a cell worse than the loop did.
    if (std::isfinite(r_star) && r_star < residual) {
      P = p_star;
      T = t_star;
      residual = r_star;
    }
  }

  status.residual = residual;
  status.max_niter = iters;
  status.converged = std::isfinite(residual) && residual <= tol;
  if (status.converged) {
    for (std::size_t m = 0; m < nmat; ++m) {
      Real rho_m, sie_m;
      eos[m].DensityEnergyFromPressureTemperature(P, T, lambda[m], rho_m, sie_m);
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

#endif // _SINGULARITY_EOS_CLOSURE_PTE_CYCLIC_RHOE_
