//------------------------------------------------------------------------------
// © 2026. Pacific Fusion. Lever-rule two-phase closure for the (rho,e) cyclic PTE solve.
//
// PTESolveCyclicRhoE brackets its sub-solves, which cannot run away -- but a bracket also cannot
// tell a root from a DISCONTINUITY.  On a two-phase tie-line a material's tau(P,T) steps between
// the coexisting branches, so tau_mix(P) jumps as P crosses P_sat(T): the bisection collapses onto
// that step and returns the jump's LOCATION with the residual still finite at both ends.  The solve
// does not diverge, it STALLS.  Measured on the RUN102 interface cell (Cu+DD+TT, rho=0.2483,
// T=19.47) with the Python twin: bracket width 5.1e-15 relative, endpoint residuals +4.59e-2 /
// -1.99e-1, so the volume residual changes by 0.245 m3/Mg -- 6% of tau_0 -- across that interval.
// That is the RUN102 off-manifold signature (exact common P, volume closure short) and very likely
// why RUN134 was SLOW (0.549 s/step) rather than aborting.
//
// The fix gives the plateau-sitting material the degree of freedom the tie-line actually has: its
// quality.  tau_m and e_m become the SAME convex combination of the same two endpoints
//
//    tau_m(x) = (1-x) tau_dense + x tau_light,   e_m(x) = (1-x) e_dense + x e_light
//
// (one weight for both -- Clayton Remark 3.2: an e built independently of tau breaks the system),
// d(tau_m)/dx = tau_light - tau_dense is O(1) where d(tau_m)/dP was unusable, and tau_m stops
// depending on P at all, so the singular row LEAVES the system rather than being softened.
//
// This is a port of the proven Python reference, pfc/sim/matdata/_pte_lever.py, restricted to what
// the singularity EOS interface can do.  The Python has three tie-line trackers; two of them walk
// the table's own density grid, which EOS does not expose, so only the density-jump bisection is
// ported.  Gated before porting: with the Python restricted to that one tracker
// (lever_tracker="jump"), MonoTie L-V converges in 6 iterations and MonoTie melt in 15, both with
// the volume residual at round-off.
//
// A SEPARATE entry point from PTESolveCyclicRhoE on purpose: that solver is validated (RUN143,
// 0.114 s/step) and must not regress while this is proven.
//------------------------------------------------------------------------------

#ifndef _SINGULARITY_EOS_CLOSURE_PTE_CYCLIC_RHOE_LEVER_
#define _SINGULARITY_EOS_CLOSURE_PTE_CYCLIC_RHOE_LEVER_

#include <cmath>
#include <limits>
#ifdef SG_LEVER_GATE_DEBUG
#include <cstdio>
#endif

#include <ports-of-call/portability.hpp>
#include <singularity-eos/base/robust_utils.hpp>
#include <singularity-eos/closure/mixed_cell_models.hpp>
#include <singularity-eos/closure/pte_cyclic_rhoe.hpp>

namespace singularity {

namespace cyclic_rhoe_lever_impl {

using namespace cyclic_rhoe_impl;

// Least (relative span)/(bracket log width) that counts as a branch step rather than smooth
// variation.  The scale is physical, not tuned: |dln tau/dln P| = 1 for an ideal gas and smaller for
// anything condensed, so a material moving 20x more than an ideal gas would across the same
// pressure interval is on no single-phase branch.  Detection re-brackets to machine width first,
// which puts the measured margin at ~1e13 on the RUN102 cell -- nowhere near this bound.  It only
// has to exclude the steep dilute branches.
constexpr Real kJumpRatioMin = 20.0;

// One material's two-phase band at one temperature.  Densities and energies are the coexisting
// endpoint states; p_sat is the common pressure they share.
struct TieLine {
  Real temperature = 0.0;
  Real p_sat = 0.0;
  Real rho_light = 0.0;
  Real rho_dense = 0.0;
  Real e_light = 0.0;
  Real e_dense = 0.0;
  bool valid = false;
  // True when rho_light sits at the table's MINIMUM DENSITY, i.e. the band runs off the low-density
  // end of the grid and there is no light-side single-phase branch tabulated.  That is the NORMAL
  // situation for a metal at low T -- Al's saturated vapour density at 25 K is ~1e-100 g/cc against
  // a grid floor of 2.7e-6 -- so the tie-line correctly spans the whole tabulated dilute range.
  // WoodBulkModulus needs it because it cannot read a light-branch modulus that is not there.
  bool light_at_grid_edge = false;

  PORTABLE_INLINE_FUNCTION Real DeltaTau() const {
    return robust::ratio(1.0, rho_light) - robust::ratio(1.0, rho_dense);
  }
  PORTABLE_INLINE_FUNCTION Real DeltaEnergy() const { return e_light - e_dense; }
  PORTABLE_INLINE_FUNCTION Real DensityMid() const { return std::sqrt(rho_light * rho_dense); }

  // tau and e at a quality, by the SAME lever weight -- see the file header on Remark 3.2.
  PORTABLE_INLINE_FUNCTION void State(const Real quality, Real &tau, Real &energy) const {
    const Real x = std::min(std::max(quality, 0.0), 1.0);
    tau = (1.0 - x) * robust::ratio(1.0, rho_dense) + x * robust::ratio(1.0, rho_light);
    energy = (1.0 - x) * e_dense + x * e_light;
  }

  // The branches' VOLUME fractions within this material.  The quality is a MASS fraction -- the
  // lever weights tau and e by mass -- so converting needs the branch specific volumes.  These sum
  // to 1 by construction.
  PORTABLE_INLINE_FUNCTION void VolumeFractions(const Real quality, Real &alpha_light,
                                                Real &alpha_dense) const {
    const Real x = std::min(std::max(quality, 0.0), 1.0);
    Real tau_m, e_m;
    State(x, tau_m, e_m);
    alpha_light = robust::ratio(x * robust::ratio(1.0, rho_light), tau_m);
    alpha_dense = robust::ratio((1.0 - x) * robust::ratio(1.0, rho_dense), tau_m);
  }

  // Wood's two-phase bulk modulus.  The branches share the pressure, so their COMPLIANCES add in
  // volume proportion: 1/K = alpha_light/K_light + alpha_dense/K_dense.
  //
  // This is the one quantity the caller CANNOT get from the table once a material is pinned, and it
  // is not a refinement -- it is a correctness requirement.  A pinned material's returned density is
  // the LEVER density, which by construction sits inside the two-phase gap, and there
  // BulkModulusFromDensityTemperature is meaningless: ~0 on a flattened tie-line, negative on a raw
  // loop.  A mixture sound speed built from it is zero or imaginary.  Evaluated on the ENDPOINTS
  // instead, which are single-phase states on their own stable branches.
  //
  // Frozen quality at equilibrium pressure -- the standard Wood construction.  It deliberately omits
  // the latent-heat term a fully relaxing (equilibrium-x) sound speed carries, so it is the stiffer
  // of the two and consistent with FrozenSpecificHeat below.  Returns 0 when either endpoint
  // modulus is unusable, which the caller must treat as "no value", not as zero stiffness.
  template <typename EOS, typename Lambda>
  PORTABLE_INLINE_FUNCTION Real WoodBulkModulus(const EOS &eos, Lambda &&lambda,
                                                const Real quality) const {
    if (!valid) return 0.0;
    Real k_light = eos.BulkModulusFromDensityTemperature(rho_light, temperature, lambda);
    const Real k_dense = eos.BulkModulusFromDensityTemperature(rho_dense, temperature, lambda);
    // When the band runs to the grid edge there is no tabulated light branch: every point on that
    // side is still ON the flat chord, so the table returns K_T = 0 and Wood would decline -- for
    // missing data, not for physics.  But the dilute end of these tables is one we INSERTED: the
    // builder replaces F below `extend_low_rho` (0.05 g/cc) with an IDEAL-GAS isotherm, and for an
    // ideal gas K_T = P exactly, with the coexisting vapour sitting at P = p_sat.  So supply the
    // analytic value rather than reading a meaningless number off the chord.  Requires p_sat > 0:
    // these tables carry genuinely NEGATIVE tie-line pressures at low T (Cu min P = -3.965e5), and
    // a negative "modulus" is not a vapour, it is tension with no light branch at all.
    if (!(k_light > 0.0) && light_at_grid_edge && p_sat > 0.0) k_light = p_sat;
    if (!(k_light > 0.0) || !(k_dense > 0.0)) return 0.0;
    Real alpha_light, alpha_dense;
    VolumeFractions(quality, alpha_light, alpha_dense);
    const Real compliance =
        robust::ratio(alpha_light, k_light) + robust::ratio(alpha_dense, k_dense);
    if (!(compliance > 0.0)) return 0.0;
    const Real k_wood = robust::ratio(1.0, compliance);
    return std::isfinite(k_wood) ? k_wood : 0.0;
  }

  // Frozen-quality specific heat: the mass-weighted endpoint c_v, for the same reason the modulus is
  // taken on the endpoints.  At the lever density, crossing T moves the state through the band's
  // energy jump, so the table's (rho,T) dE/dT there is inflated by latent heat that the mixture
  // energy already carries through the lever weight -- double counting it.  Returns 0 when unusable.
  template <typename EOS, typename Lambda>
  PORTABLE_INLINE_FUNCTION Real FrozenSpecificHeat(const EOS &eos, Lambda &&lambda,
                                                   const Real quality) const {
    if (!valid) return 0.0;
    const Real x = std::min(std::max(quality, 0.0), 1.0);
    const Real cv_light = eos.SpecificHeatFromDensityTemperature(rho_light, temperature, lambda);
    const Real cv_dense = eos.SpecificHeatFromDensityTemperature(rho_dense, temperature, lambda);
    if (!(cv_light > 0.0) || !(cv_dense > 0.0)) return 0.0;
    const Real cv = (1.0 - x) * cv_dense + x * cv_light;
    return std::isfinite(cv) ? cv : 0.0;
  }
};

// A bracketed sub-solve's final state.  The endpoint residuals are the whole point: a CONTINUOUS
// residual drives min(|f_lo|, |f_hi|) to round-off as the bracket collapses, while a coexistence
// jump leaves it finite, and that is what separates a root from a stall.
struct BracketReport {
  Real root = 0.0;
  Real lo = 0.0;
  Real hi = 0.0;
  Real f_lo = 0.0;
  Real f_hi = 0.0;
  bool ok = false;

  PORTABLE_INLINE_FUNCTION Real LogWidth() const {
    return (hi > lo && lo > 0.0) ? std::log(hi / lo) : 0.0;
  }
  PORTABLE_INLINE_FUNCTION bool Stalled(const Real scale) const {
    return ok && std::min(std::abs(f_lo), std::abs(f_hi)) > scale;
  }
};

// LogBracketRoot, reporting the final bracket instead of discarding it.  Same bisection, same
// tolerance contract; see pte_cyclic_rhoe.hpp for why bisection and not an interpolating finder.
template <typename Residual>
PORTABLE_INLINE_FUNCTION BracketReport LogBracketReport(const Real lo_in, const Real hi_in,
                                                        const Residual &resid,
                                                        const Real rel_tol = 1.0e-14) {
  BracketReport out;
  out.lo = lo_in;
  out.hi = hi_in;
  if (!(hi_in > lo_in && lo_in > 0.0)) return out;
  Real lo = lo_in, hi = hi_in;
  Real f_lo = resid(lo);
  Real f_hi = resid(hi);
  out.f_lo = f_lo;
  out.f_hi = f_hi;
  if (!std::isfinite(f_lo) || !std::isfinite(f_hi)) return out;
  if (f_lo == 0.0) { out.root = lo; out.ok = true; return out; }
  if (f_hi == 0.0) { out.root = hi; out.ok = true; return out; }
  if (f_lo * f_hi > 0.0) return out;
  for (int it = 0; it < 200; ++it) {
    const Real mid = std::sqrt(lo * hi);
    const Real f_mid = resid(mid);
    if (!std::isfinite(f_mid)) return out;
    if (f_mid == 0.0) { out.root = mid; out.lo = lo; out.hi = hi; out.ok = true; return out; }
    if ((f_mid > 0.0) == (f_lo > 0.0)) { lo = mid; f_lo = f_mid; } else { hi = mid; f_hi = f_mid; }
    if (hi / lo < 1.0 + rel_tol) break;
  }
  out.root = std::sqrt(lo * hi);
  out.lo = lo;
  out.hi = hi;
  out.f_lo = f_lo;
  out.f_hi = f_hi;
  out.ok = true;
  return out;
}

// How far a material moved across a bracket, relative to what the bracket's own width can explain.
// A RELATIVE span, not a log ratio, so it also works on a specific energy, which is routinely
// negative; for positive values it equals |dln(value)| to first order.
PORTABLE_INLINE_FUNCTION Real JumpRatio(const Real v_lo, const Real v_hi, const Real log_width) {
  const Real magnitude = std::max(std::abs(v_lo), std::abs(v_hi));
  if (!(magnitude > 0.0)) return 0.0;
  const Real span = std::abs(v_lo - v_hi) / magnitude;
  if (!std::isfinite(span)) return 0.0;
  if (log_width > 0.0) return span / log_width;
  // A bracket collapsed to zero width in floating point must still be attributable.
  return (span > 0.0) ? std::numeric_limits<Real>::infinity() : 0.0;
}

// One material's band at temperature T, by BISECTING ONTO the density jump.
//
// tau(P,T) at fixed T is monotone decreasing in P and JUMPS across the transition, so for a
// tau_inside in the gap the residual tau(P,T) - tau_inside straddles zero without ever vanishing:
// the bisection drives its bracket onto the transition pressure and the two bracket ends ARE the
// coexisting states.  No flatness test, no grid, no external saturation curve -- which is what lets
// this port at all, and it is also the only tracker that finds a melt band, whose |dlnP/dlnrho|
// only dips to 0.41 (measured, MonoTie Al 93722 at 1100 K).
//
// rel_tol is tight because this tolerance propagates into the ANSWER, not just into a step: the
// endpoint states are read AT the bracket ends, so a loose P_sat shifts the endpoint energies and
// floors the mixture's energy residual (measured: 1e-8 here floored the melt solve at a relative
// energy residual of 3.3e-10, a state it had otherwise recovered exactly).
template <typename EOS, typename Lambda>
PORTABLE_INLINE_FUNCTION TieLine TieLineByJump(const EOS &eos, Lambda &&lambda, const Real T,
                                               const Real tau_inside, const Real p_lo,
                                               const Real p_hi, const Real rel_tol = 1.0e-13) {
  TieLine tie;
  tie.temperature = T;
  const auto resid = [&](const Real p) {
    Real rho, sie;
    eos.DensityEnergyFromPressureTemperature(p, T, lambda, rho, sie);
    if (!(rho > 0.0)) return std::numeric_limits<Real>::quiet_NaN();
    return robust::ratio(1.0, rho) - tau_inside;
  };
  const BracketReport report = LogBracketReport(p_lo, p_hi, resid, rel_tol);
  if (!report.ok || !(report.hi > report.lo && report.lo > 0.0)) return tie;

  Real rho_light, sie_light, rho_dense, sie_dense;
  eos.DensityEnergyFromPressureTemperature(report.lo, T, lambda, rho_light, sie_light);
  eos.DensityEnergyFromPressureTemperature(report.hi, T, lambda, rho_dense, sie_dense);
  if (!(rho_light > 0.0) || !(rho_dense > rho_light)) return tie;
  // A smooth root also leaves the ends slightly apart, so requiring merely rho_dense > rho_light
  // accepts single-phase states as degenerate bands (measured: delta_tau = 2e-10 on MonoTie Al at
  // 1300 K), which the lever sweep would then divide by.  Demand a gap the collapsed bracket cannot
  // explain -- the same test that attributes a stall in the first place.
  if (JumpRatio(robust::ratio(1.0, rho_light), robust::ratio(1.0, rho_dense), report.LogWidth()) <=
      kJumpRatioMin) {
    return tie;
  }
  tie.p_sat = report.root;
  tie.rho_light = rho_light;
  tie.rho_dense = rho_dense;
  tie.e_light = sie_light;
  tie.e_dense = sie_dense;
  tie.valid = true;
  return tie;
}

// Widest run of CONSTANT pressure in density containing `tau_inside`, at fixed T -- the true
// binodal on a FLATTENED table.
//
// TieLineByJump brackets in PRESSURE, looking for the density step that a raw van der Waals loop
// shows as P crosses P_sat.  A hull-stabilized table has no such step BY CONSTRUCTION: P is
// exactly constant across the whole plateau, so tau(P,T) is degenerate there and the bisection
// stops wherever round-off puts it.  Measured on shipped TT 5268 at 19.9 K, where the true plateau
// runs rho 3.65e-2 .. 3.21e-1 (ratio 8.8), the jump tracker returned 2.66e-1 .. 2.87e-1 (ratio
// 1.08) -- 7.2% of the band and entirely INSIDE it.  Both "endpoints" then have K_T at round-off
// (8.3e-5 and 6.6e-5 barye against 1.0e9 on the real branch), so WoodBulkModulus, which is only as
// good as the endpoints it is handed, returned 7.2e-5 and the two-phase sound speed stayed zero.
//
// So search the axis the plateau is FLAT IN.  Walk rho outward from a point known to be inside,
// geometrically, while P stays at P_sat, then bisect each edge.  This is what the Python
// reference's two unported trackers do by walking the table's own density grid; it needs no grid,
// only PressureFromDensityTemperature, which the EOS interface does expose.
template <typename EOS, typename Lambda>
PORTABLE_INLINE_FUNCTION TieLine TieLineByPlateau(const EOS &eos, Lambda &&lambda, const Real T,
                                                  const Real tau_inside,
                                                  const Real rel_tol = 1.0e-9) {
  TieLine tie;
  tie.temperature = T;
  const Real rho_in = robust::ratio(1.0, tau_inside);
  const Real rho_min = eos.MinimumDensity();
  const Real rho_max = eos.MaximumDensity();
  if (!(rho_in > rho_min) || !(rho_in < rho_max)) return tie;

  const Real p_sat = eos.PressureFromDensityTemperature(rho_in, T, lambda);
  if (!std::isfinite(p_sat)) return tie;
  // Absolute floor as well as relative: a tie-line pressure may legitimately be ~0 (or negative)
  // on these tables, where a purely relative test would never call anything flat.
  const Real p_scale = std::max(std::abs(p_sat), 1.0);
  const auto flat = [&](const Real r) {
    const Real p = eos.PressureFromDensityTemperature(r, T, lambda);
    return std::isfinite(p) && std::abs(p - p_sat) <= rel_tol * p_scale;
  };
  if (!flat(rho_in)) return tie; // not on a plateau at all

  // One edge: march by `step` (>1 outward in density, <1 inward) to the first point off the
  // plateau, then bisect in log rho.  Returns the last point still ON it.
  const auto edge = [&](const Real step, const Real bound) {
    Real on = rho_in;
    Real off = 0.0;
    bool found = false;
    for (int i = 0; i < 200; ++i) {
      Real next = on * step;
      if ((step > 1.0 && next >= bound) || (step < 1.0 && next <= bound)) {
        next = bound;
        if (flat(next)) return next; // plateau runs to the table edge
        off = next;
        found = true;
        break;
      }
      if (!flat(next)) {
        off = next;
        found = true;
        break;
      }
      on = next;
    }
    if (!found) return on;
    for (int i = 0; i < 60; ++i) {
      const Real mid = std::sqrt(on * off);
      if (!(mid > 0.0) || mid == on || mid == off) break;
      if (flat(mid)) on = mid;
      else off = mid;
      if (std::abs(std::log(off / on)) < 1.0e-12) break;
    }
    return on;
  };

  const Real rho_light = edge(1.0 / 1.25, rho_min);
  const Real rho_dense = edge(1.25, rho_max);
  if (!(rho_light > 0.0) || !(rho_dense > rho_light)) return tie;
  // Reject a plateau no wider than the flatness test itself can resolve -- otherwise a smooth,
  // very stiff branch reads as a degenerate band and the lever sweep divides by its delta_tau.
  if (rho_dense / rho_light < 1.0 + 1.0e-6) return tie;
  // A band with NO LATENT HEAT is not coexistence, and the lever cannot use it: the whole point
  // of pinning is that the quality moves the energy, via DeltaEnergy = e_light - e_dense. On the
  // cold isotherms the tables' flat segments run to the density grid FLOOR rather than to a
  // vapour binodal (Cu's true vapour density at 72 K is ~1e-213 g/cc), so they are stability
  // repairs, not tie-lines, and their tabulated e is flat -- DeltaEnergy is exactly 0. Pinning
  // there would leave the energy equation with no free variable.
  {
    const Real e_light_probe = eos.InternalEnergyFromDensityTemperature(rho_light, T, lambda);
    const Real e_dense_probe = eos.InternalEnergyFromDensityTemperature(rho_dense, T, lambda);
    const Real e_scale = std::max(std::abs(e_light_probe), std::abs(e_dense_probe));
    if (!(std::abs(e_light_probe - e_dense_probe) > 1.0e-12 * std::max(e_scale, 1.0))) return tie;
  }

  tie.p_sat = p_sat;
  tie.rho_light = rho_light;
  tie.rho_dense = rho_dense;
  tie.light_at_grid_edge = (rho_light <= rho_min * (1.0 + 1.0e-9));
  tie.e_light = eos.InternalEnergyFromDensityTemperature(rho_light, T, lambda);
  tie.e_dense = eos.InternalEnergyFromDensityTemperature(rho_dense, T, lambda);
  if (!std::isfinite(tie.e_light) || !std::isfinite(tie.e_dense)) return tie;
  tie.valid = true;
  return tie;
}

// Mixture tau/e with the pinned members served by their lever state instead of the table.
template <typename EOSIndexer, typename RealIndexer, typename LambdaIndexer>
PORTABLE_INLINE_FUNCTION void MixtureTauELever(const std::size_t nmat, EOSIndexer &eos,
                                               const RealIndexer &Ym, LambdaIndexer &lambda,
                                               const bool *pinned, const TieLine *ties,
                                               const Real quality, const Real P, const Real T,
                                               Real &tau, Real &energy) {
  tau = 0.0;
  energy = 0.0;
  for (std::size_t m = 0; m < nmat; ++m) {
    Real tau_m, e_m;
    if (pinned[m] && ties[m].valid) {
      ties[m].State(quality, tau_m, e_m);
    } else {
      Real rho_m, sie_m;
      eos[m].DensityEnergyFromPressureTemperature(P, T, lambda[m], rho_m, sie_m);
      if (!(rho_m > 0.0) || !std::isfinite(sie_m)) {
        tau = std::numeric_limits<Real>::quiet_NaN();
        energy = std::numeric_limits<Real>::quiet_NaN();
        return;
      }
      tau_m = robust::ratio(1.0, rho_m);
      e_m = sie_m;
    }
    tau += Ym[m] * tau_m;
    energy += Ym[m] * e_m;
  }
}

} // namespace cyclic_rhoe_lever_impl

// Per-material lever state, reported out for two distinct reasons.
//
// `bmod` and `cv` are REQUIRED for a correct answer: a pinned material's returned density is inside
// the two-phase gap, so the caller cannot re-derive them from the table at that density (see
// TieLine::WoodBulkModulus).  The rest is diagnostic -- which materials pinned, at what quality, and
// between which coexisting densities -- which is what makes a run interpretable at all: without it
// there is no way to tell whether the lever ever engaged.
//
// Every array is [nmat] and independently optional; a null pointer is not written.  The solver zeroes
// whatever it is given on ENTRY, so an unpinned material -- and every material of a solve that
// declines -- reads back as not pinned rather than as stale state from a previous cell.
struct LeverDiagnostics {
  Real *quality = nullptr;   // shared group quality x (light-branch MASS fraction)
  Real *rho_light = nullptr; // coexisting light-branch density
  Real *rho_dense = nullptr; // coexisting dense-branch density
  Real *bmod = nullptr;      // Wood two-phase bulk modulus; 0 = unavailable
  Real *cv = nullptr;        // frozen-quality specific heat; 0 = unavailable
  int *pinned = nullptr;     // 1 iff the LEVER, not the table, set this material's state
  int n_pinned = 0;          // OUT: how many materials the lever pinned
};

// Solve the ion PTE system for (P, T) given (rho, e), giving a plateau-sitting material its
// quality as a free variable.  Same contract as PTESolveCyclicRhoE -- see that header -- with one
// difference that matters: for a PINNED material the returned rho and sie are the LEVER state, not
// a (P,T) root, because on a tie-line the (P,T) root is precisely the ambiguous quantity.  vfrac
// then closes to vfrac_tot by construction rather than by luck.
//
// A separate entry point from PTESolveCyclicRhoE, and self-contained: the ordinary step is repeated
// here rather than shared, so the validated solver (RUN143, 0.114 s/step) cannot regress while this
// is proven.  Fold them together once it is.
template <typename EOSIndexer, typename RealIndexer, typename LambdaIndexer>
PORTABLE_INLINE_FUNCTION SolverStatus
PTESolveCyclicRhoELever(const std::size_t nmat, EOSIndexer &&eos, const Real vfrac_tot,
                        const Real sie_tot, RealIndexer &&rho, RealIndexer &&vfrac,
                        RealIndexer &&sie, RealIndexer &&temp, RealIndexer &&press,
                        LambdaIndexer &&lambda, Real &Tguess, const MixParams &params,
                        LeverDiagnostics *diag = nullptr) {
  using namespace cyclic_rhoe_lever_impl;
  SolverStatus status;
  status.converged = false;
  status.residual = std::numeric_limits<Real>::infinity();

  // Called before any early return AND on every failure exit.  A partially-filled diag would be
  // worse than none: the caller's fallback ladder can rescue a declined cell by other means, and
  // would then apply a lever modulus belonging to densities this solve never accepted.
  const auto clear_diag = [&]() {
    if (diag == nullptr) return;
    diag->n_pinned = 0;
    for (std::size_t m = 0; m < nmat; ++m) {
      if (diag->quality != nullptr) diag->quality[m] = 0.0;
      if (diag->rho_light != nullptr) diag->rho_light[m] = 0.0;
      if (diag->rho_dense != nullptr) diag->rho_dense[m] = 0.0;
      if (diag->bmod != nullptr) diag->bmod[m] = 0.0;
      if (diag->cv != nullptr) diag->cv[m] = 0.0;
      if (diag->pinned != nullptr) diag->pinned[m] = 0;
    }
  };
  clear_diag();

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
  // Every pressure that can reach a log MUST be strictly positive, so this floored bound -- not
  // the raw p_lo -- is what all the clamps below use.  p_lo is max(MinimumPressure()), which is
  // ZERO for TableDependsRhoT.  The hull-stabilized tables carry genuinely NEGATIVE tie-line
  // pressures (Cu min P = -3.965e5, Al -2.964e5 barye), so a pinned material's p_sat clamped to
  // p_lo lands on exactly 0 and the next log() aborts:
  //     ### ERROR  Condition: x > 0   "log divergent for x <= 0"   fast-math/logs.hpp:253
  // That killed RUN160/161/162 at the identical step (17526) -- the step the liner first went
  // two-phase, i.e. the instant the lever first pinned.  1 barye = 1e-6 bar is physically
  // negligible next to any tie-line pressure of interest.
  const Real p_lo_bracket = std::max(p_lo, 1.0);
  const Real t_lo_bracket = std::max(t_lo, 1.0e-8);

  // Lever state.  `pinned` is per material because a pooled group shares one quality: the
  // isotope-transformed DD/TT pair steps together (measured on the RUN102 cell, both jumping by
  // 6.2399e-2, the same value to five digits), and two separate saturation rows would be
  // near-parallel.  Members keep their OWN endpoints; only `quality` is shared.
  bool pinned[kMaxMat] = {};
  TieLine ties[kMaxMat];
  Real seed_rho[kMaxMat] = {};
  Real quality = 0.5;
  bool any_pinned = false;
  bool demoted = false;
  std::size_t dominant = 0;
  int stall_streak = 0;
  int clamped_streak = 0;

  // Bands at a temperature, recomputed rather than stored: the T sub-solve queries many
  // temperatures and a stale band is the usual reason tracking fails.
  const auto bands_at = [&](const Real T, TieLine *out) {
    for (std::size_t m = 0; m < nmat; ++m) {
      out[m] = TieLine();
      if (!pinned[m]) continue;
      const Real tau_inside = robust::ratio(1.0, seed_rho[m]);
      // Plateau tracker FIRST: it is the one that works on a flattened table, where P is exactly
      // constant across the band and there is no pressure step for the jump tracker to bisect.
      // Fall back to the jump tracker for a RAW van der Waals loop, where P is NOT flat across
      // the band (so the plateau search finds nothing) but the density root does step.
      out[m] = TieLineByPlateau(eos[m], lambda[m], T, tau_inside);
      if (!out[m].valid) {
        out[m] = TieLineByJump(eos[m], lambda[m], T, tau_inside, p_lo_bracket, p_hi);
      }
    }
  };
  const auto mix_plain = [&](const Real P, const Real T, Real &tau, Real &e) {
    MixtureTauE(nmat, eos, Ym, lambda, P, T, tau, e);
  };
  const auto mix_at = [&](const Real P, const Real T, Real &tau, Real &e) {
    TieLine local[kMaxMat];
    bands_at(T, local);
    MixtureTauELever(nmat, eos, Ym, lambda, pinned, local, quality, P, T, tau, e);
  };
  const auto residual_of = [&](const Real P, const Real T) {
    Real tau, e;
    if (any_pinned) mix_at(P, T, tau, e); else mix_plain(P, T, tau, e);
    if (!std::isfinite(tau)) return std::numeric_limits<Real>::infinity();
    return std::max(std::abs(tau - tau_target) / tau_target,
                    std::abs(e - sie_tot) / (1.0 + std::abs(sie_tot)));
  };

  Real T = std::min(std::max(Tguess, t_lo), t_hi);
  Real P = 0.0;
  for (std::size_t m = 0; m < nmat; ++m) {
    if (press[m] > 0.0) { P = press[m]; break; }
  }
  if (!(P > 0.0)) {
    const auto tau_resid = [&](const Real p) {
      Real tau, e;
      mix_plain(p, T, tau, e);
      return tau - tau_target;
    };
    if (!LogBracketRoot(p_lo_bracket, p_hi, tau_resid, P)) {
      std::size_t idom = 0;
      for (std::size_t m = 1; m < nmat; ++m)
        if (Ym[m] > Ym[idom]) idom = m;
      P = eos[idom].PressureFromDensityTemperature(rho[idom], T);
    }
  }
  P = std::min(std::max(P, p_lo_bracket), p_hi);

  const Real tol =
      std::max(std::min(params.pte_rel_tolerance_v, params.pte_rel_tolerance_e), 1.0e-14);
  const std::size_t max_iter =
      (params.pte_max_iter_per_mat > 0) ? params.pte_max_iter_per_mat * nmat : 100;
  const Real scale_tau = tol * tau_target;

  Real residual = residual_of(P, T);
  std::size_t iters = 0;
  for (; iters < max_iter && residual > tol; ++iters) {
    TieLine now[kMaxMat];
    if (any_pinned) bands_at(T, now);
    const bool have_band = any_pinned && now[dominant].valid;

    if (have_band) {
      // ---- Pinned: P is slaved to P_sat(T), so the singular row is gone from the system. ----
      const auto sat_pressure = [&](const Real t) {
        TieLine local[kMaxMat];
        bands_at(t, local);
        return local[dominant].valid ? std::min(std::max(local[dominant].p_sat, p_lo_bracket), p_hi) : P;
      };
      const auto mix_on_sat = [&](const Real t, Real &tau, Real &e) {
        TieLine local[kMaxMat];
        bands_at(t, local);
        const Real p_t =
            local[dominant].valid ? std::min(std::max(local[dominant].p_sat, p_lo_bracket), p_hi) : P;
        MixtureTauELever(nmat, eos, Ym, lambda, pinned, local, quality, p_t, t, tau, e);
      };
      const auto lever_slopes = [&](const Real t, Real &d_tau, Real &d_e) {
        TieLine local[kMaxMat];
        bands_at(t, local);
        d_tau = 0.0;
        d_e = 0.0;
        for (std::size_t m = 0; m < nmat; ++m) {
          if (pinned[m] && local[m].valid) {
            d_tau += Ym[m] * local[m].DeltaTau();
            d_e += Ym[m] * local[m].DeltaEnergy();
          }
        }
      };

      // Coupled 2x2 Newton in (T, x) along the saturation curve.  Required, not an optimisation:
      // moving x moves the mixture energy in proportion to the LATENT HEAT, so on a melt band
      // (delta_e 0.218 against delta_tau 0.015, measured) a sequential T-then-x sweep oscillates.
      // The determinant is the thermal direction crossed with the latent one, bounded away from
      // zero exactly where the (P,T) Newton determinant rho c_v K_T vanishes.  Derivatives by
      // differencing the MIXTURE: the analytical assembly needs d(tau)/dP AT a binodal, where
      // P_rho is small, and measured worse (melt 12 -> 18 iterations, raw diverged).
      Real t_next = T;
      bool coupled_ok = false;
      Real d_tau_dx, d_e_dx;
      lever_slopes(T, d_tau_dx, d_e_dx);
      if (d_tau_dx > 0.0) {
        const Real h = 1.0e-3 * T;
        Real tau_lo, e_lo, tau_hi, e_hi, tau_here, e_here;
        mix_on_sat(T - h, tau_lo, e_lo);
        mix_on_sat(T + h, tau_hi, e_hi);
        mix_on_sat(T, tau_here, e_here);
        const Real d_tau_dt = (tau_hi - tau_lo) / (2.0 * h);
        const Real d_e_dt = (e_hi - e_lo) / (2.0 * h);
        const Real det = d_tau_dt * d_e_dx - d_tau_dx * d_e_dt;
        if (std::isfinite(det) && det != 0.0 && std::isfinite(tau_here) && std::isfinite(e_here)) {
          const Real r_tau = tau_target - tau_here;
          const Real r_e = sie_tot - e_here;
          const Real dt = (r_tau * d_e_dx - d_tau_dx * r_e) / det;
          const Real dx = (d_tau_dt * r_e - r_tau * d_e_dt) / det;
          if (std::isfinite(dt) && std::isfinite(dx)) {
            t_next = std::min(std::max(T + dt, t_lo), t_hi);
            const Real target_x = quality + dx;
            if (target_x < 0.0 || target_x > 1.0) ++clamped_streak; else clamped_streak = 0;
            quality = std::min(std::max(target_x, 0.0), 1.0);
            coupled_ok = true;
          }
        }
      }
      if (!coupled_ok) {
        // Fall back to a bracketed T solve on energy with P slaved -- the sub-solve keeps working
        // where the linearisation does not.
        const auto e_resid = [&](const Real t) {
          Real tau, e;
          mix_on_sat(t, tau, e);
          return e - sie_tot;
        };
        Real root;
        if (LogBracketRoot(t_lo_bracket, t_hi, e_resid, root, 1.0e-12)) t_next = root;
      }

      P = sat_pressure(t_next);
      T = t_next;
      // Close volume exactly at the new (P, T): the residual is affine in x there, so this is one
      // exact correction, not an iteration.
      Real d_tau_new, d_e_new;
      lever_slopes(T, d_tau_new, d_e_new);
      if (d_tau_new > 0.0) {
        Real tau_now, e_now;
        mix_at(P, T, tau_now, e_now);
        if (std::isfinite(tau_now)) {
          const Real target_x = quality + (tau_target - tau_now) / d_tau_new;
          if (target_x < 0.0 || target_x > 1.0) ++clamped_streak; else clamped_streak = 0;
          quality = std::min(std::max(target_x, 0.0), 1.0);
        }
      }
      // Only a CLAMPED quality demotes -- that is the genuine "this material is single-phase"
      // signal.  A tracking miss must never demote: it says nothing about the physics, and
      // unpinning drops the only mechanism that can close volume (measured in the Python twin:
      // three misses after a large T step demoted at iteration 10 and the solve then oscillated at
      // rel_tau ~3e-3 for 90 more iterations instead of converging).
      if (clamped_streak >= 3) {
        for (std::size_t m = 0; m < nmat; ++m) pinned[m] = false;
        any_pinned = false;
        demoted = true;
        clamped_streak = 0;
      }
      residual = residual_of(P, T);
      continue;
    }

    if (any_pinned) {
      // Pinned but the band is not visible at this T -- re-seed from the current density and take
      // the ordinary step.  No demotion; see above.
      for (std::size_t m = 0; m < nmat; ++m) {
        if (!pinned[m]) continue;
        Real rho_m, sie_m;
        eos[m].DensityEnergyFromPressureTemperature(P, T, lambda[m], rho_m, sie_m);
        if (rho_m > 0.0) seed_rho[m] = rho_m;
      }
    }

    // ---- Unpinned (or band-less): the ordinary (rho,e) cyclic step, with its brackets reported.
    Real tau_now, e_now;
    mix_plain(P, T, tau_now, e_now);
    const Real tau_err = std::abs(tau_now - tau_target) / tau_target;
    const Real e_err = std::abs(e_now - sie_tot) / (1.0 + std::abs(sie_tot));
    const Real sub_tol = std::min(std::max(1.0e-3 * std::max(tau_err, e_err), 1.0e-14), 1.0e-3);

    const auto tau_resid = [&](const Real p) {
      Real tau, e;
      mix_plain(p, T, tau, e);
      return tau - tau_target;
    };
    const auto e_resid = [&](const Real t) {
      Real tau, e;
      mix_plain(P, t, tau, e);
      return e - sie_tot;
    };
    const BracketReport vol = LogBracketReport(p_lo_bracket, p_hi, tau_resid, sub_tol);
    const BracketReport ene = LogBracketReport(t_lo_bracket, t_hi, e_resid, sub_tol);
    Real p_tilde = vol.ok ? vol.root : P;
    Real t_tilde = ene.ok ? ene.root : T;

    const Real hf = 1.0e-4;
    if (!vol.ok) {
      const Real f0 = tau_resid(P);
      const Real slope = (tau_resid(P * (1.0 + hf)) - f0) / (hf * P);
      if (slope != 0.0 && std::isfinite(slope) && std::isfinite(f0))
        p_tilde = std::min(std::max(P - f0 / slope, p_lo_bracket), p_hi);
    }
    if (!ene.ok) {
      const Real g0 = e_resid(T);
      const Real slope = (e_resid(T * (1.0 + hf)) - g0) / (hf * T);
      if (slope != 0.0 && std::isfinite(slope) && std::isfinite(g0))
        t_tilde = std::min(std::max(T - g0 / slope, t_lo), t_hi);
    }

    // Promotion.  The stall test runs on a RE-TIGHTENED bracket, and it has to: the sub-tolerance
    // above is adaptive, so early on every bracket is loose and every residual looks large, which
    // would promote in any solve at all.  Tightening separates the cases -- a continuous residual
    // drives min(|f_lo|,|f_hi|) to round-off, a coexistence jump leaves it finite -- and restores
    // the attribution margin (measured: jump ratio ~60 at sub_tol 1e-3 against ~1e13 collapsed).
#ifdef SG_LEVER_GATE_DEBUG
    // Which of the six serial conditions blocks promotion?  A bare "n_pinned = 0" cannot say, and
    // guessing has cost several 20-minute FLASH runs.  Offline driver only -- never defined in the
    // FLASH build.
    std::printf("[gate] it=%zu res=%.4e demoted=%d any_pinned=%d vol.ok=%d vol.Stalled=%d "
                "streak=%d scale_tau=%.4e vol.lo=%.6e vol.hi=%.6e\n",
                static_cast<std::size_t>(iters), residual, demoted ? 1 : 0, any_pinned ? 1 : 0,
                vol.ok ? 1 : 0, (vol.ok && vol.Stalled(scale_tau)) ? 1 : 0, stall_streak,
                scale_tau, vol.lo, vol.hi);
#endif
    if (!demoted && !any_pinned && vol.ok && vol.Stalled(scale_tau)) {
      const BracketReport tight = LogBracketReport(vol.lo, vol.hi, tau_resid, 1.0e-14);
#ifdef SG_LEVER_GATE_DEBUG
      {
        Real e_dbg = e_now;
        if (ene.ok) { Real t_dbg; mix_plain(P, t_tilde, t_dbg, e_dbg); }
        std::printf("[gate]   tight.ok=%d tight.Stalled=%d e_rel(at T)=%.4e e_rel(at t_tilde)=%.4e\n",
                    tight.ok ? 1 : 0, (tight.ok && tight.Stalled(scale_tau)) ? 1 : 0,
                    std::abs(e_now - sie_tot) / (1.0 + std::abs(sie_tot)),
                    std::abs(e_dbg - sie_tot) / (1.0 + std::abs(sie_tot)));
      }
#endif
      if (tight.ok && tight.Stalled(scale_tau)) {
        ++stall_streak;
        const Real e_rel = std::abs(e_now - sie_tot) / (1.0 + std::abs(sie_tot));
        if (stall_streak >= 2 && e_rel <= 1.0e-2) {
          const Real width = tight.LogWidth();
          Real span[kMaxMat] = {};
          Real best = 0.0;
          for (std::size_t m = 0; m < nmat; ++m) {
            Real r_lo, e_l, r_hi, e_h;
            eos[m].DensityEnergyFromPressureTemperature(tight.lo, T, lambda[m], r_lo, e_l);
            eos[m].DensityEnergyFromPressureTemperature(tight.hi, T, lambda[m], r_hi, e_h);
            if (!(r_lo > 0.0) || !(r_hi > 0.0)) continue;
            const Real tau_l = robust::ratio(1.0, r_lo);
            const Real tau_h = robust::ratio(1.0, r_hi);
            if (JumpRatio(tau_l, tau_h, width) > kJumpRatioMin) {
              const Real magnitude = std::max(std::abs(tau_l), std::abs(tau_h));
              span[m] = (magnitude > 0.0) ? std::abs(tau_l - tau_h) / magnitude : 0.0;
              best = std::max(best, span[m]);
            }
          }
          if (best > 0.0) {
            // Pool the members whose span matches the largest: same transition stage, one shared
            // quality, each keeping its own endpoints.
            for (std::size_t m = 0; m < nmat; ++m) {
              if (span[m] > 0.0 && std::abs(span[m] - best) <= 1.0e-6 * best) {
                pinned[m] = true;
                Real r_lo, e_l, r_hi, e_h;
                eos[m].DensityEnergyFromPressureTemperature(tight.lo, T, lambda[m], r_lo, e_l);
                eos[m].DensityEnergyFromPressureTemperature(tight.hi, T, lambda[m], r_hi, e_h);
                seed_rho[m] = std::sqrt(r_lo * r_hi); // mid-band density
                any_pinned = true;
              }
            }
            if (any_pinned) {
              dominant = 0;
              Real y_best = -1.0;
              for (std::size_t m = 0; m < nmat; ++m) {
                if (pinned[m] && Ym[m] > y_best) { y_best = Ym[m]; dominant = m; }
              }
              quality = 0.5;
              residual = residual_of(P, T);
              continue;
            }
          }
        }
      } else {
        stall_streak = 0;
      }
    } else if (!vol.ok || !vol.Stalled(scale_tau)) {
      stall_streak = 0;
    }

    // Step 3: tangent intersection, exactly as PTESolveCyclicRhoE.
    const Real h = 1.0e-4;
    Real tau_pp, tau_pm, tau_tp, tau_tm, e_unused;
    mix_plain(p_tilde * (1.0 + h), T, tau_pp, e_unused);
    mix_plain(p_tilde * (1.0 - h), T, tau_pm, e_unused);
    mix_plain(p_tilde, T * (1.0 + h), tau_tp, e_unused);
    mix_plain(p_tilde, T * (1.0 - h), tau_tm, e_unused);
    Real e_pp, e_pm, e_tp, e_tm, tau_unused;
    mix_plain(P * (1.0 + h), t_tilde, tau_unused, e_pp);
    mix_plain(P * (1.0 - h), t_tilde, tau_unused, e_pm);
    mix_plain(P, t_tilde * (1.0 + h), tau_unused, e_tp);
    mix_plain(P, t_tilde * (1.0 - h), tau_unused, e_tm);
    const Real dtau_dp = (tau_pp - tau_pm) / (2.0 * h * p_tilde);
    const Real dtau_dt = (tau_tp - tau_tm) / (2.0 * h * T);
    const Real de_dp = (e_pp - e_pm) / (2.0 * h * P);
    const Real de_dt = (e_tp - e_tm) / (2.0 * h * t_tilde);

    Real p_next = p_tilde, t_next2 = t_tilde;
    const Real m_rho = (dtau_dp != 0.0) ? -dtau_dt / dtau_dp : 0.0;
    const Real inv_m_e = (de_dt != 0.0) ? -de_dp / de_dt : 0.0;
    const Real ab = m_rho * inv_m_e;
    if (std::isfinite(ab) && std::abs(1.0 - ab) > 1.0e-300) {
      const Real t_int = (t_tilde + inv_m_e * (p_tilde - P) - ab * T) / (1.0 - ab);
      const Real p_int = p_tilde + m_rho * (t_int - T);
      if (std::isfinite(t_int) && std::isfinite(p_int) && p_int >= p_lo && p_int <= p_hi &&
          t_int >= t_lo && t_int <= t_hi) {
        p_next = p_int;
        t_next2 = t_int;
      }
    }
    const Real p_new = std::min(std::max(p_next, p_lo_bracket), p_hi);
    const Real t_new = std::min(std::max(t_next2, t_lo), t_hi);
    const Real residual_new = residual_of(p_new, t_new);
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

  status.residual = residual;
  status.max_niter = iters;
  status.converged = std::isfinite(residual) && residual <= tol;
  if (status.converged) {
    TieLine final_bands[kMaxMat];
    if (any_pinned) bands_at(T, final_bands);
    for (std::size_t m = 0; m < nmat; ++m) {
      Real tau_m, e_m;
      if (any_pinned && pinned[m] && final_bands[m].valid) {
        // The LEVER state, not a (P,T) root: on a tie-line the root is the ambiguous quantity, and
        // writing it back would contradict the solve.  vfrac then closes to vfrac_tot exactly.
        final_bands[m].State(quality, tau_m, e_m);
        if (diag != nullptr) {
          ++diag->n_pinned;
          if (diag->pinned != nullptr) diag->pinned[m] = 1;
          if (diag->quality != nullptr) diag->quality[m] = quality;
          if (diag->rho_light != nullptr) diag->rho_light[m] = final_bands[m].rho_light;
          if (diag->rho_dense != nullptr) diag->rho_dense[m] = final_bands[m].rho_dense;
          if (diag->bmod != nullptr)
            diag->bmod[m] = final_bands[m].WoodBulkModulus(eos[m], lambda[m], quality);
          if (diag->cv != nullptr)
            diag->cv[m] = final_bands[m].FrozenSpecificHeat(eos[m], lambda[m], quality);
        }
      } else {
        Real rho_m, sie_m;
        eos[m].DensityEnergyFromPressureTemperature(P, T, lambda[m], rho_m, sie_m);
        if (!(rho_m > 0.0) || !std::isfinite(sie_m)) {
          status.converged = false;
          clear_diag();
          return status;
        }
        tau_m = robust::ratio(1.0, rho_m);
        e_m = sie_m;
      }
      if (!(tau_m > 0.0) || !std::isfinite(e_m)) {
        status.converged = false;
        clear_diag();
        return status;
      }
      rho[m] = robust::ratio(1.0, tau_m);
      sie[m] = e_m;
      temp[m] = T;
      press[m] = P;
      vfrac[m] = Ym[m] * rho_bulk * tau_m;
    }
    Tguess = T;
  }
  return status;
}

} // namespace singularity


#endif // _SINGULARITY_EOS_CLOSURE_PTE_CYCLIC_RHOE_LEVER_
