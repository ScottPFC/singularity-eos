//------------------------------------------------------------------------------
// The dual PTE solve with an explicit kink step: the lever rule as the subgradient condition.
//------------------------------------------------------------------------------
// PTESolveDual maximises the concave J(beta, gamma) = sum Y_m Xi_m - beta e0 - gamma tau0, with
// Xi = G/T.  On a hulled two-phase material G = min(G_dense, G_light), so Xi is the MINIMUM of two
// smooth concave branch potentials and has a kink on the coexistence curve
//
//     Gamma = { Xi_light = Xi_dense } = { gamma = beta P_sat(1/beta) }.
//
// Its superdifferential there is conv{grad Xi_dense, grad Xi_light}, i.e. the segment
// { (1-x)(e_d, tau_d) + x (e_l, tau_l) : x in [0,1] }, so 0 in dJ IS the lever-rule PTE system
// with the vapour quality x as the convex weight.  Two consequences are used here:
//
//   1. Clausius-Clapeyron is the kink's tangent.  Gamma is a level set of Xi_l - Xi_d, whose
//      gradient is the latent vector g = (de, dtau), so the tangent t has g.t = 0, i.e.
//      dgamma/dbeta = -de/dtau, which is dP_sat/dT = (de + P dtau)/(T dtau) in (beta, gamma).
//   2. The reduced dual psi(beta) = max_gamma J(beta, gamma) is smooth and concave on the kink.
//      The inner maximisation at fixed T is the monotone isochoric volume solve: it either finds a
//      smooth root, or its bracket collapses onto P_sat, where the gamma component of the
//      subgradient must vanish -- which fixes x from the volume equation alone.  psi' is then the
//      remaining energy residual and psi'' = t^T H(x) t < 0, the curvature of J ALONG the kink,
//      finite where the unreduced det H is unbounded.
//
// PHASE 1 is PTESolveDual's own seed, warm-start probe and Newton iteration, operation for
// operation, so a cell the dual closes by Newton alone gets the dual's answer bit for bit.  The
// first Newton step that fails to contract hands over, where PTESolveDual would take a coordinate
// step, to PHASE 2: Newton on psi, safeguarded by a bracket on the monotone psi' (bisection
// whenever a step fails to halve |psi'|, because psi' is only C0 in slope where the isochore
// enters a dome).  On the kink each inner solve is a bisection onto the GROUP's own volume jump,
// predicted by Clapeyron.  The kink is detected by the inner bracket collapsing with both ends
// outside tolerance, which is a measurement, not a flatness sensor.
//
// psi is defined only where the isochore has a volume root inside the pressure box.  From a far
// seed it may have none at the seed temperature; the volume residual at the box ends then says
// which way the domain lies (too expanded at every P: cool; too compressed: heat), and phase 2
// steps T geometrically that way until the isochore exists.  A cell phase 2 still cannot close is
// handed to PTESolveDual from the caller's untouched inputs, so the kink solver converges on
// every cell the dual does.  The one exception is a certified decline: the isochore AT a
// temperature bound, twice, with the energy residual pointing further out, which proves no
// state in the box meets the target.
//
// THE MIXTURE MODULUS ON A KINK.  A group member's returned density is its lever density, inside
// the two-phase gap, where a (rho,T) table's bulk modulus is ~0 on a flattened tie-line and
// negative on a raw loop.  The solver therefore reports, per group member, Wood's frozen-quality
// modulus 1/K = alpha_l/K_S,l + alpha_d/K_S,d over the two coexisting endpoint states, with each
// endpoint's K_S = K_T c_p/c_v taken from the SAME (P,T) partials the solve used, and the
// matching frozen-quality c_v.  Frozen quality omits the latent-heat relaxation, so this is the
// stiffer (upper) two-phase sound speed.  See DualKinkDiagnostics.
//
// Offline twin: pfc/sim/matdata/_pte_dual_kink.py.  Units are the caller's (FLASH passes cgs).
//------------------------------------------------------------------------------

#ifndef _SINGULARITY_EOS_CLOSURE_PTE_DUAL_KINK_
#define _SINGULARITY_EOS_CLOSURE_PTE_DUAL_KINK_

#include <algorithm>
#include <cmath>
#include <limits>

#include <ports-of-call/portability.hpp>
#include <singularity-eos/base/robust_utils.hpp>
#include <singularity-eos/closure/mixed_cell_models.hpp>
#include <singularity-eos/closure/pte_cyclic_rhoe.hpp>
#include <singularity-eos/closure/pte_dual.hpp>

namespace singularity {

namespace pte_dual_kink_impl {

using cyclic_rhoe_impl::LogBracketRoot;
using cyclic_rhoe_impl::LogBracketRootCore;
using pte_dual_impl::DualState;
using pte_dual_impl::MixtureDual;
using pte_dual_impl::NewtonDirection;

constexpr std::size_t kMaxMat = 16;
// Bracket width ratio at which an inner solve stops.  Tight, because on the kink the endpoint
// states are read AT the bracket ends, so the width is an error in the answer, not in a step.
constexpr Real kInnerRelTol = 1.0e-14;
// A material "jumped" across a collapsed bracket when its relative volume change exceeds this
// multiple of the bracket's own log width.
constexpr Real kJumpRatioMin = 20.0;
// Two jumped materials pool into one lever unit when their relative spans agree this closely
// (isotope-scaled DD/TT tilt to one shelf and step together).
constexpr Real kPoolRelTol = 1.0e-6;
constexpr Real kPredictorMinWidth = 1.0e-9;
// Phase-2 temperature step bound while one side of the psi' bracket is still unknown.
constexpr Real kMaxTRatio = 4.0;

// One material at one (P, T): the state and its (P,T) partials.
struct Point {
  Real tau, e, tau_P, tau_T, e_P, e_T;
  bool ok;
};

template <typename EOS_T, typename Lambda>
PORTABLE_INLINE_FUNCTION Point EvalPoint(EOS_T &eos, const Real P, const Real T, Lambda &&lambda) {
  Real rho, sie, drho_dP, drho_dT, de_dP, de_dT;
  eos.DensityEnergyDerivativesFromPressureTemperature(P, T, lambda, rho, sie, drho_dP, drho_dT,
                                                      de_dP, de_dT);
  Point p;
  p.ok = rho > 0.0 && std::isfinite(sie) && std::isfinite(drho_dP) && std::isfinite(drho_dT) &&
         std::isfinite(de_dP) && std::isfinite(de_dT);
  const Real inv_rho2 = robust::ratio(1.0, rho * rho);
  p.tau = robust::ratio(1.0, rho);
  p.e = sie;
  p.tau_P = -drho_dP * inv_rho2;
  p.tau_T = -drho_dT * inv_rho2;
  p.e_P = de_dP;
  p.e_T = de_dT;
  return p;
}

// Isochoric specific heat of one single-phase state from its (P,T) partials:
// c_v = (de/dT)_tau = e_T + e_P (dP/dT)_tau, with (dP/dT)_tau = -tau_T/tau_P.
PORTABLE_INLINE_FUNCTION Real PointSpecificHeat(const Point &p) {
  if (!(p.tau_P < 0.0)) return 0.0;
  const Real cv = p.e_T - p.e_P * p.tau_T / p.tau_P;
  return (std::isfinite(cv) && cv > 0.0) ? cv : 0.0;
}

// Adiabatic bulk modulus of one single-phase state from its (P,T) partials:
// K_S = K_T c_p/c_v with K_T = -tau/tau_P and c_p = (dh/dT)_P = e_T + P tau_T.
// 0 when the state is not mechanically and thermally stable (K_T, c_v or c_p not positive).
PORTABLE_INLINE_FUNCTION Real PointBulkModulus(const Point &p, const Real P) {
  if (!(p.tau_P < 0.0)) return 0.0;
  const Real k_t = -p.tau / p.tau_P;
  const Real cv = PointSpecificHeat(p);
  const Real cp = p.e_T + P * p.tau_T;
  if (!(k_t > 0.0) || !(cv > 0.0) || !(cp > 0.0)) return 0.0;
  const Real k_s = k_t * cp / cv;
  return std::isfinite(k_s) ? k_s : 0.0;
}

// Wood's frozen-quality modulus of a two-phase member at light-branch mass fraction x:
// the branches share the pressure, so their compliances add in volume proportion.
PORTABLE_INLINE_FUNCTION Real WoodBulkModulus(const Point &light, const Point &dense, const Real P,
                                              const Real x) {
  const Real k_l = PointBulkModulus(light, P);
  const Real k_d = PointBulkModulus(dense, P);
  if (!(k_l > 0.0) || !(k_d > 0.0)) return 0.0;
  const Real tau_m = (1.0 - x) * dense.tau + x * light.tau;
  const Real alpha_l = robust::ratio(x * light.tau, tau_m);
  const Real alpha_d = robust::ratio((1.0 - x) * dense.tau, tau_m);
  const Real compliance = robust::ratio(alpha_l, k_l) + robust::ratio(alpha_d, k_d);
  if (!(compliance > 0.0)) return 0.0;
  const Real k = robust::ratio(1.0, compliance);
  return std::isfinite(k) ? k : 0.0;
}

// Frozen-quality specific heat: the mass-weighted endpoint c_v.  The lever energy already
// carries the latent heat through x, so the gap-density table value would count it twice.
PORTABLE_INLINE_FUNCTION Real FrozenSpecificHeat(const Point &light, const Point &dense,
                                                 const Real x) {
  const Real cv_l = PointSpecificHeat(light), cv_d = PointSpecificHeat(dense);
  if (!(cv_l > 0.0) || !(cv_d > 0.0)) return 0.0;
  const Real cv = (1.0 - x) * cv_d + x * cv_l;
  return std::isfinite(cv) ? cv : 0.0;
}

// The inner maximiser gamma*(beta) at one temperature, with the reduced gradient and curvature.
struct Isochore {
  Real P, T, tau, e, slope; // slope = d(e - e0)/dT along the isochore (on the kink: along Gamma)
  bool kink;
  Real x, p_sat_slope;
  bool in_group[kMaxMat];
  Point light[kMaxMat], dense[kMaxMat], other[kMaxMat];
  bool valid;
};

PORTABLE_INLINE_FUNCTION void SmoothIsochore(const std::size_t nmat, const Real *Ym,
                                             const Point *pts, const Real P, const Real T,
                                             Isochore &out) {
  Real tau = 0, e = 0, tP = 0, tT = 0, eP = 0, eT = 0;
  for (std::size_t m = 0; m < nmat; ++m) {
    tau += Ym[m] * pts[m].tau;
    e += Ym[m] * pts[m].e;
    tP += Ym[m] * pts[m].tau_P;
    tT += Ym[m] * pts[m].tau_T;
    eP += Ym[m] * pts[m].e_P;
    eT += Ym[m] * pts[m].e_T;
    out.other[m] = pts[m];
    out.in_group[m] = false;
  }
  out.P = P;
  out.T = T;
  out.tau = tau;
  out.e = e;
  out.slope = (tP != 0.0) ? eT - eP * tT / tP : eT;
  out.kink = false;
  out.x = std::numeric_limits<Real>::quiet_NaN();
  out.p_sat_slope = std::numeric_limits<Real>::quiet_NaN();
  out.valid = std::isfinite(tau) && std::isfinite(e);
}

// The reduced state on the kink: x from the (affine) volume equation, and the slope
// A_e - (D_e/D_tau) A_tau, where D = sum_G Y dX is the latent vector and A the motion of every
// state along Gamma at fixed x, d/dT = d_T + P_sat' d_P with Clapeyron's P_sat'.
PORTABLE_INLINE_FUNCTION bool KinkIsochore(const std::size_t nmat, const Real *Ym,
                                           const Real tau0, const Real P, const Real T,
                                           Isochore &k) {
  Real d_tau = 0, d_e = 0, tau_dense = 0, tau_rest = 0;
  for (std::size_t m = 0; m < nmat; ++m) {
    if (k.in_group[m]) {
      d_tau += Ym[m] * (k.light[m].tau - k.dense[m].tau);
      d_e += Ym[m] * (k.light[m].e - k.dense[m].e);
      tau_dense += Ym[m] * k.dense[m].tau;
    } else {
      tau_rest += Ym[m] * k.other[m].tau;
    }
  }
  if (!(d_tau > 0.0)) return false;
  const Real x = (tau0 - tau_rest - tau_dense) / d_tau;
  if (!(x >= 0.0 && x <= 1.0)) return false;
  const Real ps = (d_e + P * d_tau) / (T * d_tau);
  Real a_tau = 0, a_e = 0, e = 0;
  for (std::size_t m = 0; m < nmat; ++m) {
    if (k.in_group[m]) {
      const Point &l = k.light[m], &d = k.dense[m];
      a_tau += Ym[m] * ((1 - x) * (d.tau_T + d.tau_P * ps) + x * (l.tau_T + l.tau_P * ps));
      a_e += Ym[m] * ((1 - x) * (d.e_T + d.e_P * ps) + x * (l.e_T + l.e_P * ps));
      e += Ym[m] * ((1 - x) * d.e + x * l.e);
    } else {
      const Point &o = k.other[m];
      a_tau += Ym[m] * (o.tau_T + o.tau_P * ps);
      a_e += Ym[m] * (o.e_T + o.e_P * ps);
      e += Ym[m] * o.e;
    }
  }
  k.P = P;
  k.T = T;
  k.tau = tau0;
  k.e = e;
  k.slope = a_e - d_e / d_tau * a_tau;
  k.kink = true;
  k.x = x;
  k.p_sat_slope = ps;
  k.valid = std::isfinite(e) && std::isfinite(k.slope);
  return k.valid;
}

// Per-material attribution of a collapsed bracket; fills in_group with the largest pooled group.
PORTABLE_INLINE_FUNCTION bool AttributeGroup(const std::size_t nmat, const Point *lo,
                                             const Point *hi, const Real log_width,
                                             bool *in_group) {
  Real span[kMaxMat];
  Real best = 0.0;
  for (std::size_t m = 0; m < nmat; ++m) {
    const Real mag = std::max(std::abs(lo[m].tau), std::abs(hi[m].tau));
    span[m] = (mag > 0.0) ? std::abs(lo[m].tau - hi[m].tau) / mag : 0.0;
    const bool jumped = (log_width > 0.0) ? span[m] / log_width > kJumpRatioMin : span[m] > 0.0;
    if (!jumped) span[m] = 0.0;
    best = std::max(best, span[m]);
  }
  if (!(best > 0.0)) return false;
  for (std::size_t m = 0; m < nmat; ++m)
    in_group[m] = span[m] > 0.0 && std::abs(span[m] - best) <= kPoolRelTol * best;
  return true;
}

} // namespace pte_dual_kink_impl

// Per-material two-phase state of a converged kink solve.
//
// `bmod` and `cv` are what a caller needs for a correct mixture sound speed and heat capacity: a
// group member's returned density is its lever density, inside the gap, so neither can be read
// back from a (rho,T) table at that density.  The rest says which materials the kink pinned, at
// what quality, between which coexisting densities.
//
// Every array is [nmat] and independently optional; a null pointer is not written.  All are
// zeroed on entry, so a material the kink did not pin -- and every material of a solve that did
// not end on a kink or did not converge -- reads back as not pinned.
struct DualKinkDiagnostics {
  Real *quality = nullptr;   // group quality x (light-branch MASS fraction)
  Real *rho_light = nullptr; // coexisting light-branch density
  Real *rho_dense = nullptr; // coexisting dense-branch density
  Real *bmod = nullptr;      // Wood frozen-quality adiabatic bulk modulus; 0 = unavailable
  Real *cv = nullptr;        // frozen-quality specific heat; 0 = unavailable
  int *pinned = nullptr;     // 1 iff the kink, not the (P,T) table, set this material's state
  int n_pinned = 0;          // OUT: how many materials the kink pinned
};

// Solve the ion PTE system for (P, T) given (rho, e): PTESolveDual's Newton while it contracts,
// then Newton on the reduced dual psi with the kink step.  Argument contract is PTESolveDual's.
// On a converged kink, the two-phase group's per-material outputs are the lever states
// (one quality for the whole group), so vfrac closes exactly.  Status fields:
// `safeguard_iters` counts reduced (psi) iterations, 0 when Newton alone closed the cell;
// `pressure_jump_attempts` is 1 when the returned state is on a kink; and `stagnation_iters` is
// 1 when phase 2 could not close the cell and the status is PTESolveDual's.
template <typename EOSIndexer, typename RealIndexer, typename LambdaIndexer>
PORTABLE_INLINE_FUNCTION SolverStatus
PTESolveDualKink(const std::size_t nmat, EOSIndexer &&eos, const Real vfrac_tot,
                 const Real sie_tot, RealIndexer &&rho, RealIndexer &&vfrac, RealIndexer &&sie,
                 RealIndexer &&temp, RealIndexer &&press, LambdaIndexer &&lambda, Real &Tguess,
                 const MixParams &params, DualKinkDiagnostics *diag = nullptr) {
  using namespace pte_dual_kink_impl;
  SolverStatus status;
  status.converged = false;
  status.residual = std::numeric_limits<Real>::infinity();

  if (diag != nullptr) {
    diag->n_pinned = 0;
    for (std::size_t m = 0; m < nmat && m < kMaxMat; ++m) {
      if (diag->quality != nullptr) diag->quality[m] = 0.0;
      if (diag->rho_light != nullptr) diag->rho_light[m] = 0.0;
      if (diag->rho_dense != nullptr) diag->rho_dense[m] = 0.0;
      if (diag->bmod != nullptr) diag->bmod[m] = 0.0;
      if (diag->cv != nullptr) diag->cv[m] = 0.0;
      if (diag->pinned != nullptr) diag->pinned[m] = 0;
    }
  }

  // ---- phase 1: PTESolveDual's setup, probe and Newton iteration, unchanged ----------------
  if (nmat < 1 || !std::isfinite(sie_tot) || !(Tguess > 0.0) || !(vfrac_tot > 0.0)) return status;
  Real rho_bulk = 0.0;
  for (std::size_t m = 0; m < nmat; ++m) {
    if (!(vfrac[m] > 0.0) || !(rho[m] > 0.0)) return status;
    rho_bulk += vfrac[m] * rho[m];
  }
  if (!(rho_bulk > 0.0)) return status;
  if (nmat > kMaxMat) return status;
  Real Ym[kMaxMat];
  for (std::size_t m = 0; m < nmat; ++m) Ym[m] = vfrac[m] * rho[m] / rho_bulk;
  const Real tau0 = robust::ratio(vfrac_tot, rho_bulk);
  if (!(tau0 > 0.0)) return status;

  Real p_lo = eos[0].MinimumPressure();
  Real p_hi = eos[0].MaximumPressureAtTemperature(Tguess);
  Real t_lo = eos[0].MinimumTemperature();
  for (std::size_t m = 1; m < nmat; ++m) {
    p_lo = std::max(p_lo, eos[m].MinimumPressure());
    p_hi = std::min(p_hi, eos[m].MaximumPressureAtTemperature(Tguess));
    t_lo = std::max(t_lo, eos[m].MinimumTemperature());
  }
  const Real t_hi = params.temperature_limit;
  if (!(p_hi > 0.0) || !(t_hi > t_lo)) return status;
  const Real p_lo_bracket = std::max(p_lo, 1.0);
  const Real t_lo_bracket = std::max(t_lo, 1.0e-8);
  const Real p_lo_clamp = std::max(p_lo, 1.0e-300);
  const Real t_lo_clamp = t_lo_bracket;
  const Real e_scale = 1.0 + std::abs(sie_tot);

  auto merit_of = [&](const DualState &s) {
    if (!s.valid) return std::numeric_limits<Real>::infinity();
    const Real rt = s.residual_tau / tau0;
    const Real re = s.residual_energy / (1.0 + std::abs(sie_tot));
    return 0.5 * (rt * rt + re * re);
  };
  auto max_norm_of = [&](const DualState &s) {
    if (!s.valid) return std::numeric_limits<Real>::infinity();
    return std::max(std::abs(s.residual_tau) / tau0,
                    std::abs(s.residual_energy) / (1.0 + std::abs(sie_tot)));
  };

  Real T = std::min(std::max(Tguess, t_lo_clamp), t_hi);
  Real P = 0.0;
  for (std::size_t m = 0; m < nmat; ++m) {
    if (press[m] > 0.0) {
      P = press[m];
      break;
    }
  }
  if (!(P > 0.0)) {
    const auto tau_resid = [&](const Real p) {
      Real tau, energy;
      cyclic_rhoe_impl::MixtureTauE(nmat, eos, Ym, lambda, p, T, tau, energy);
      return tau - tau0;
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
  const Real scale = tol * tau0;

  Real beta = 1.0 / T;
  Real gamma = P / T;
  Real rho_keep[kMaxMat], sie_keep[kMaxMat], rho_try[kMaxMat], sie_try[kMaxMat];

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
      const Real rt = (tau_probe - tau0) / tau0;
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

  DualState state = MixtureDual(nmat, eos, Ym, lambda, tau0, sie_tot, P, T, rho_keep, sie_keep);
  Real merit = merit_of(state);
  std::size_t iters = 0;
  while (iters < max_iter && max_norm_of(state) > tol) {
    ++iters;
    Real d_beta, d_gamma;
    if (!state.valid || !NewtonDirection(state, beta, gamma, d_beta, d_gamma)) break;
    Real alpha = 1.0;
    const Real value[2] = {beta, gamma};
    const Real delta[2] = {d_beta, d_gamma};
    for (int k = 0; k < 2; ++k) {
      if (delta[k] < 0.0) {
        alpha = std::min(alpha, (1.0 - pte_dual_impl::kPositivityRetention) * value[k] / -delta[k]);
      } else if (delta[k] > 0.0) {
        alpha = std::min(alpha, (pte_dual_impl::kMaxStepRatio - 1.0) * value[k] / delta[k]);
      }
    }
    bool accepted = false;
    for (int back = 0; back < pte_dual_impl::kMaxBacktracks; ++back) {
      const Real tb = beta + alpha * d_beta;
      const Real tg = gamma + alpha * d_gamma;
      if (!(tb > 0.0) || !(tg > 0.0)) {
        alpha *= 0.5;
        continue;
      }
      const Real trial_T = std::min(std::max(1.0 / tb, t_lo_clamp), t_hi);
      const Real trial_P = std::min(std::max(tg / tb, p_lo_clamp), p_hi);
      const DualState trial =
          MixtureDual(nmat, eos, Ym, lambda, tau0, sie_tot, trial_P, trial_T, rho_try, sie_try);
      const Real trial_merit = merit_of(trial);
      if (std::isfinite(trial_merit) && trial_merit < pte_dual_impl::kNewtonContraction * merit) {
        beta = 1.0 / trial_T;
        gamma = trial_P * beta;
        state = trial;
        merit = trial_merit;
        for (std::size_t m = 0; m < nmat; ++m) {
          rho_keep[m] = rho_try[m];
          sie_keep[m] = sie_try[m];
        }
        accepted = true;
        break;
      }
      alpha *= 0.5;
    }
    if (!accepted) break;
  }

  T = 1.0 / beta;
  P = gamma * T;
  {
    const Real residual = max_norm_of(state);
    if (std::isfinite(residual) && residual <= tol) {
      status.residual = residual;
      status.max_niter = iters;
      for (std::size_t m = 0; m < nmat; ++m) {
        if (!(rho_keep[m] > 0.0) || !std::isfinite(sie_keep[m])) return status;
      }
      for (std::size_t m = 0; m < nmat; ++m) {
        rho[m] = rho_keep[m];
        sie[m] = sie_keep[m];
        temp[m] = T;
        press[m] = P;
        vfrac[m] = Ym[m] * rho_bulk / rho_keep[m];
      }
      Tguess = T;
      status.converged = true;
      return status;
    }
  }

  // ---- phase 2: Newton on the reduced dual psi(beta), with the kink step ------------------
  auto tau_of = [&](const std::size_t m, const Real Pq, const Real Tq) {
    Real r, e;
    eos[m].DensityEnergyFromPressureTemperature(Pq, Tq, lambda[m], r, e);
    return robust::ratio(1.0, r);
  };

  Isochore cur, cand;
  cur.valid = false;
  Point lo_pts[kMaxMat], hi_pts[kMaxMat];

  // Full inner maximisation at T: a bracketed volume solve, warm-started around `guess`.
  auto full_isochore = [&](const Real Tq, const Real guess, Isochore &out) {
    const auto r = [&](const Real p) {
      Real t = 0;
      for (std::size_t m = 0; m < nmat; ++m) t += Ym[m] * tau_of(m, p, Tq);
      return t - tau0;
    };
    const Real widths[3] = {1.0e-3, 1.0e-1, 3.0};
    for (int w = 0; w <= 3; ++w) {
      Real lo = p_lo_bracket, hi = p_hi;
      if (w < 3) {
        if (!(guess > p_lo_bracket && guess < p_hi)) continue;
        lo = std::max(p_lo_bracket, guess * std::exp(-widths[w]));
        hi = std::min(p_hi, guess * std::exp(widths[w]));
      }
      Real root, blo, bhi, flo, fhi;
      if (!LogBracketRootCore(lo, hi, r, kInnerRelTol, root, blo, bhi, flo, fhi)) continue;
      if (std::min(std::abs(flo), std::abs(fhi)) > scale) {
        bool ok = true;
        for (std::size_t m = 0; m < nmat; ++m) {
          lo_pts[m] = EvalPoint(eos[m], blo, Tq, lambda[m]);
          hi_pts[m] = EvalPoint(eos[m], bhi, Tq, lambda[m]);
          ok = ok && lo_pts[m].ok && hi_pts[m].ok;
        }
        if (ok && AttributeGroup(nmat, lo_pts, hi_pts, std::log(bhi / blo), out.in_group)) {
          for (std::size_t m = 0; m < nmat; ++m) {
            out.light[m] = lo_pts[m];
            out.dense[m] = hi_pts[m];
            out.other[m] = lo_pts[m];
          }
          if (KinkIsochore(nmat, Ym, tau0, std::sqrt(blo * bhi), Tq, out)) return true;
        }
      }
      for (std::size_t m = 0; m < nmat; ++m) {
        lo_pts[m] = EvalPoint(eos[m], root, Tq, lambda[m]);
        if (!lo_pts[m].ok) return false;
      }
      SmoothIsochore(nmat, Ym, lo_pts, root, Tq, out);
      return out.valid;
    }
    return false;
  };

  // Which way the isochore's domain lies from Tq when it has no volume root in the box: -1 cool
  // (the mixture is too expanded even at the top of the box), +1 heat (too compressed even at
  // the bottom), 0 when the ends straddle tau0 or are not finite.
  auto isochore_direction = [&](const Real Tq) {
    Real tau_lo = 0, tau_hi = 0;
    for (std::size_t m = 0; m < nmat; ++m) {
      tau_lo += Ym[m] * tau_of(m, p_lo_bracket, Tq);
      tau_hi += Ym[m] * tau_of(m, p_hi, Tq);
    }
    if (!std::isfinite(tau_lo) || !std::isfinite(tau_hi)) return 0;
    if (tau_lo > tau0 && tau_hi > tau0) return -1;
    if (tau_lo < tau0 && tau_hi < tau0) return 1;
    return 0;
  };

  // Follow a known kink to a new T by bisecting onto the GROUP's volume jump around the
  // Clapeyron prediction; the target is the group volume the mixture needs there.
  auto track_kink = [&](const Real Tq, const Isochore &prev, Isochore &out) {
    const Real pred = prev.P + prev.p_sat_slope * (Tq - prev.T);
    if (!(pred > 0.0)) return false;
    Real target = tau0;
    for (std::size_t m = 0; m < nmat; ++m)
      if (!prev.in_group[m]) target -= Ym[m] * tau_of(m, pred, Tq);
    const auto r = [&](const Real p) {
      Real t = 0;
      for (std::size_t m = 0; m < nmat; ++m)
        if (prev.in_group[m]) t += Ym[m] * tau_of(m, p, Tq);
      return t - target;
    };
    Real width = std::max(2.0 * std::abs(std::log(pred / prev.P)), kPredictorMinWidth);
    Real root, blo, bhi, flo, fhi;
    while (true) {
      const Real lo = std::max(p_lo_bracket, pred * std::exp(-width));
      const Real hi = std::min(p_hi, pred * std::exp(width));
      if (LogBracketRootCore(lo, hi, r, kInnerRelTol, root, blo, bhi, flo, fhi)) break;
      if (lo == p_lo_bracket && hi == p_hi) return false;
      width *= 16.0;
    }
    const Real lw = std::log(bhi / blo);
    for (std::size_t m = 0; m < nmat; ++m) {
      out.in_group[m] = prev.in_group[m];
      if (!prev.in_group[m]) continue;
      out.light[m] = EvalPoint(eos[m], blo, Tq, lambda[m]);
      out.dense[m] = EvalPoint(eos[m], bhi, Tq, lambda[m]);
      if (!out.light[m].ok || !out.dense[m].ok) return false;
      const Real mag = std::max(std::abs(out.light[m].tau), std::abs(out.dense[m].tau));
      const Real span = std::abs(out.light[m].tau - out.dense[m].tau) / mag;
      if (!(lw > 0.0 ? span / lw > kJumpRatioMin : span > 0.0)) return false;
    }
    const Real p_sat = std::sqrt(blo * bhi);
    for (std::size_t m = 0; m < nmat; ++m) {
      if (prev.in_group[m]) continue;
      out.other[m] = EvalPoint(eos[m], p_sat, Tq, lambda[m]);
      if (!out.other[m].ok) return false;
    }
    return KinkIsochore(nmat, Ym, tau0, p_sat, Tq, out);
  };

  Real t_below = t_lo_clamp, t_above = t_hi;
  bool known_below = false, known_above = false;
  Real previous_f = std::numeric_limits<Real>::infinity();
  int pinned = 0;
  bool done = false;
  bool certified_decline = false;
  std::size_t phase2_iters = 0;
  while (iters < max_iter) {
    ++iters;
    ++phase2_iters;
    bool got = false;
    if (cur.valid && cur.kink) got = track_kink(T, cur, cand);
    if (!got) got = full_isochore(T, cur.valid ? cur.P : P, cand);
    if (!got) {
      const int dir = isochore_direction(T);
      const Real lo_b = known_below ? t_below : t_lo_clamp;
      const Real hi_b = known_above ? t_above : t_hi;
      const Real t_next =
          (dir > 0) ? std::min(T * kMaxTRatio, hi_b) : std::max(T / kMaxTRatio, lo_b);
      if (dir == 0 || !(t_next != T)) break;
      T = t_next;
      continue;
    }
    cur = cand;
    const Real f = cur.e - sie_tot;
    if (std::abs(f) < tol * e_scale && std::abs(cur.tau - tau0) < scale) {
      done = true;
      break;
    }
    if (f > 0.0) {
      t_above = T;
      known_above = true;
    } else {
      t_below = T;
      known_below = true;
    }
    // Pinned at a temperature bound with the residual pushing further out: the target is not
    // reachable inside the box (the recession direction of the concave program).
    const bool at_floor = T <= t_lo_clamp * (1.0 + 1.0e-12) && f > 0.0;
    const bool at_ceil = T >= t_hi * (1.0 - 1.0e-12) && f < 0.0;
    pinned = (at_floor || at_ceil) ? pinned + 1 : 0;
    if (pinned >= 2) {
      certified_decline = true;
      break;
    }
    // A bracket at machine width is the resolution floor the kink endpoints set.
    if (known_below && known_above && t_above <= t_below * (1.0 + 4.0e-16)) break;
    Real t_next = (cur.slope > 0.0 && std::isfinite(cur.slope))
                      ? T - f / cur.slope
                      : std::numeric_limits<Real>::quiet_NaN();
    const Real lo_b = known_below ? t_below : std::max(t_lo_clamp, T / kMaxTRatio);
    const Real hi_b = known_above ? t_above : std::min(t_hi, T * kMaxTRatio);
    const bool stagnated = std::abs(f) > 0.5 * previous_f && known_below && known_above;
    previous_f = std::abs(f);
    if (stagnated || !(std::isfinite(t_next) && t_next > lo_b && t_next < hi_b)) {
      if (known_below && known_above)
        t_next = std::sqrt(t_below * t_above);
      else
        t_next = (f < 0.0) ? hi_b : lo_b;
    }
    T = std::min(std::max(t_next, t_lo_clamp), t_hi);
  }

  status.max_niter = iters;
  status.safeguard_iters = phase2_iters;
  if (cur.valid) {
    status.pressure_jump_attempts = cur.kink ? 1 : 0;
    status.residual =
        std::max(std::abs(cur.tau - tau0) / tau0, std::abs(cur.e - sie_tot) / e_scale);
  }
  if (done) {
    for (std::size_t m = 0; m < nmat; ++m) {
      Real tau_m = cur.other[m].tau;
      if (cur.kink && cur.in_group[m])
        tau_m = (1.0 - cur.x) * cur.dense[m].tau + cur.x * cur.light[m].tau;
      if (!(tau_m > 0.0)) done = false;
    }
  }
  if (!done) {
    if (certified_decline) return status;
    // Nothing has been written back, so the dual starts from exactly the caller's inputs.
    SolverStatus fallback = PTESolveDual(nmat, eos, vfrac_tot, sie_tot, rho, vfrac, sie, temp,
                                         press, lambda, Tguess, params);
    fallback.stagnation_iters = 1;
    return fallback;
  }
  status.converged = true;
  for (std::size_t m = 0; m < nmat; ++m) {
    Real tau_m = cur.other[m].tau, e_m = cur.other[m].e;
    if (cur.kink && cur.in_group[m]) {
      const Point &l = cur.light[m], &d = cur.dense[m];
      tau_m = (1.0 - cur.x) * d.tau + cur.x * l.tau;
      e_m = (1.0 - cur.x) * d.e + cur.x * l.e;
      if (diag != nullptr) {
        ++diag->n_pinned;
        if (diag->pinned != nullptr) diag->pinned[m] = 1;
        if (diag->quality != nullptr) diag->quality[m] = cur.x;
        if (diag->rho_light != nullptr) diag->rho_light[m] = robust::ratio(1.0, l.tau);
        if (diag->rho_dense != nullptr) diag->rho_dense[m] = robust::ratio(1.0, d.tau);
        if (diag->bmod != nullptr) diag->bmod[m] = WoodBulkModulus(l, d, cur.P, cur.x);
        if (diag->cv != nullptr) diag->cv[m] = FrozenSpecificHeat(l, d, cur.x);
      }
    }
    rho[m] = robust::ratio(1.0, tau_m);
    sie[m] = e_m;
    temp[m] = cur.T;
    press[m] = cur.P;
    vfrac[m] = Ym[m] * rho_bulk * tau_m;
  }
  Tguess = cur.T;
  return status;
}

} // namespace singularity

#endif // _SINGULARITY_EOS_CLOSURE_PTE_DUAL_KINK_
