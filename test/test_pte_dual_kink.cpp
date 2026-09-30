//------------------------------------------------------------------------------
// PTESolveDualKink against closed forms.
//
// A hulled van der Waals fluid (exact Maxwell construction: flat tie-line, tau(P,T) jumps at
// P_sat) mixed with an ideal gas.  Each cell is built from a chosen (T*, x*), so the PTE answer
// is known exactly: the solver must return T*, the vdW member's lever quality x*, Wood's
// frozen-quality modulus over the two saturated states, and the constant vdW c_v.  A second
// block checks that on single-phase cells Newton alone closes, the kink solver's output is
// bitwise PTESolveDual's.  A third, in a narrow pressure box, takes far temperature seeds whose
// isochore has no volume root in the box: phase 2 must walk T to where it has one, and no cell
// the dual closes may be lost.
//
// Standalone (no Catch2): prints one line per check, exits non-zero on any failure.
//------------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <singularity-eos/closure/mixed_cell_models.hpp>
#include <singularity-eos/closure/pte_dual.hpp>
#include <singularity-eos/closure/pte_dual_kink.hpp>

using singularity::DualKinkDiagnostics;
using singularity::MixParams;
using singularity::PTESolveDual;
using singularity::PTESolveDualKink;
using singularity::SolverStatus;
using Real = double;

namespace {

constexpr Real kGPa = 1.0e10, kMJkg = 1.0e10;
// Pressure box every material reports; the far-seed block narrows it.
Real g_p_min = 1.0, g_p_max = 10 * kGPa;

struct Sat {
  Real T = -1, p = 0, tl = 0, tv = 0, el = 0, ev = 0;
  bool ok = false;
};

// Van der Waals fluid with water-like critical point, cgs.  (P,T) queries return the stable
// branch, so tau jumps from tl to tv at P_sat.
struct Vdw {
  Real R, a, b, cv, Tc;
  mutable Sat cache;
  Vdw() {
    R = 4.615e-4 * kMJkg;
    const Real t_c = 647.0, p_c = 0.0221 * kGPa;
    b = R * t_c / (8 * p_c);
    a = 27 * b * b * p_c;
    cv = 3.25 * R;
    Tc = 8 * a / (27 * R * b);
  }
  Real p(Real t, Real T) const { return R * T / (t - b) - a / (t * t); }
  Real pt(Real t, Real T) const { return -R * T / ((t - b) * (t - b)) + 2 * a / (t * t * t); }
  Real root(Real P, Real T, Real lo, Real hi) const {
    for (int i = 0; i < 400 && hi / lo > 1 + 4e-16; ++i) {
      const Real m = std::sqrt(lo * hi);
      (p(m, T) > P ? lo : hi) = m;
    }
    Real t = std::sqrt(lo * hi);
    for (int k = 0; k < 3; ++k) {
      const Real s = pt(t, T);
      if (s != 0) {
        const Real tn = t - (p(t, T) - P) / s;
        if (tn > lo * (1 - 1e-12) && tn < hi * (1 + 1e-12)) t = tn;
      }
    }
    return t;
  }
  Real hsp(Real t, Real T) const { return R * T * t * t * t - 2 * a * (t - b) * (t - b); }
  Real spinodal(Real lo, Real hi, Real T) const {
    const Real flo = hsp(lo, T);
    for (int i = 0; i < 300 && hi / lo > 1 + 1e-15; ++i) {
      const Real m = std::sqrt(lo * hi);
      if ((hsp(m, T) > 0) == (flo > 0)) lo = m;
      else hi = m;
    }
    return std::sqrt(lo * hi);
  }
  const Sat &sat(Real T) const {
    if (cache.T == T) return cache;
    cache = Sat();
    cache.T = T;
    if (T >= 0.9999 * Tc) return cache;
    const Real s1 = spinodal(b * (1 + 1e-12), 3 * b, T), s2 = spinodal(3 * b, 1e12 * b, T);
    const Real pmin = p(s1, T), pmax = p(s2, T), span = pmax - pmin;
    Real lo = pmin <= 0 ? pmax * 1e-30 : pmin + 1e-9 * span, hi = pmax - 1e-9 * span;
    auto area = [&](Real P, Real &tl, Real &tv) {
      tl = root(P, T, b * (1 + 1e-15), s1);
      Real top = s2;
      while (p(top, T) > P) top *= 2;
      tv = root(P, T, s2, top);
      return R * T * std::log((tv - b) / (tl - b)) + a / tv - a / tl - P * (tv - tl);
    };
    Real tl, tv;
    for (int i = 0; i < 200 && hi / lo > 1 + 1e-15; ++i) {
      const Real m = std::sqrt(lo * hi);
      (area(m, tl, tv) > 0 ? lo : hi) = m;
    }
    area(std::sqrt(lo * hi), tl, tv);
    for (int k = 0; k < 8; ++k) { // Newton polish on (equal P, equal area)
      const Real pl = p(tl, T), pv = p(tv, T), sl = pt(tl, T), sv = pt(tv, T);
      const Real g1 = pl - pv;
      const Real g2 = R * T * std::log((tv - b) / (tl - b)) + a / tv - a / tl - pl * (tv - tl);
      const Real j11 = sl, j12 = -sv, j21 = -sl * (tv - tl), j22 = pv - pl;
      const Real det = j11 * j22 - j12 * j21;
      if (det == 0) break;
      tl -= (g1 * j22 - j12 * g2) / det;
      tv -= (j11 * g2 - j21 * g1) / det;
    }
    cache.p = p(tl, T);
    cache.tl = tl;
    cache.tv = tv;
    cache.el = cv * T - a / tl;
    cache.ev = cv * T - a / tv;
    cache.ok = true;
    return cache;
  }
  void PT(Real P, Real T, Real &tau, Real &e, Real &tP, Real &tT, Real &eP, Real &eT) const {
    const Sat &s = sat(T);
    if (!s.ok) {
      Real top = 3 * b;
      while (p(top, T) > P) top *= 2;
      tau = root(P, T, b * (1 + 1e-15), top);
    } else if (P >= s.p) {
      tau = root(P, T, b * (1 + 1e-15), s.tl);
    } else {
      Real top = s.tv;
      while (p(top, T) > P) top *= 2;
      tau = root(P, T, s.tv, top);
    }
    const Real Pt = pt(tau, T), PT_ = R / (tau - b), et = a / (tau * tau);
    e = cv * T - a / tau;
    tP = 1 / Pt;
    tT = -PT_ / Pt;
    eP = et / Pt;
    eT = cv - et * PT_ / Pt;
  }
  // Closed-form adiabatic modulus of a single-phase state: K_S = K_T + T tau (dP/dT)^2 / c_v.
  Real KS(Real tau, Real T) const {
    const Real pT = R / (tau - b);
    return -tau * pt(tau, T) + T * tau * pT * pT / cv;
  }
};

struct Ideal { // tau = R T / P, e = cv T
  Real R = 4.157e-3 * kMJkg, cv = 2.5 * 4.157e-3 * kMJkg;
  void PT(Real P, Real T, Real &tau, Real &e, Real &tP, Real &tT, Real &eP, Real &eT) const {
    tau = R * T / P;
    e = cv * T;
    tP = -tau / P;
    tT = R / P;
    eP = 0;
    eT = cv;
  }
};

struct Stiff { // stiffened gas, water parameters
  Real cv = 1.6e-3 * kMJkg, cp = 4.4 * 1.6e-3 * kMJkg, pinf = 0.6 * kGPa;
  void PT(Real P, Real T, Real &tau, Real &e, Real &tP, Real &tT, Real &eP, Real &eT) const {
    const Real r = cp - cv, g = cp / cv, d = P + pinf;
    tau = r * T / d;
    e = cv * T * (P + g * pinf) / d;
    tP = -r * T / (d * d);
    tT = r / d;
    eP = cv * T * (d - (P + g * pinf)) / (d * d);
    eT = cv * (P + g * pinf) / d;
  }
};

// The singularity-shaped (P,T) interface the dual solvers need.
template <typename M>
struct Adapter {
  const M *m;
  Real Pmin = 1.0, Pmax = 10 * kGPa, Tmin = 250.0;
  template <class L>
  void DensityEnergyFromPressureTemperature(Real P, Real T, L &&, Real &r, Real &s) const {
    Real tau, tP, tT, eP, eT;
    m->PT(P, T, tau, s, tP, tT, eP, eT);
    r = 1 / tau;
  }
  template <class L>
  void DensityEnergyDerivativesFromPressureTemperature(Real P, Real T, L &&, Real &r, Real &s,
                                                       Real &drdP, Real &drdT, Real &dedP,
                                                       Real &dedT) const {
    Real tau, tP, tT;
    m->PT(P, T, tau, s, tP, tT, dedP, dedT);
    r = 1 / tau;
    drdP = -r * r * tP;
    drdT = -r * r * tT;
  }
  Real MinimumPressure() const { return Pmin; }
  Real MaximumPressureAtTemperature(Real) const { return Pmax; }
  Real MinimumTemperature() const { return Tmin; }
  Real PressureFromDensityTemperature(Real, Real) const { return 1.0e6; }
};

// Two-material indexer over heterogeneous adapters, via a small polymorphic shim.
struct AnyEos {
  virtual ~AnyEos() {}
  virtual void DE(Real P, Real T, Real &r, Real &s) const = 0;
  virtual void DD(Real P, Real T, Real &r, Real &s, Real &a, Real &b, Real &c, Real &d) const = 0;
  virtual Real Tmin() const = 0;
  template <class L>
  void DensityEnergyFromPressureTemperature(Real P, Real T, L &&, Real &r, Real &s) const {
    DE(P, T, r, s);
  }
  template <class L>
  void DensityEnergyDerivativesFromPressureTemperature(Real P, Real T, L &&, Real &r, Real &s,
                                                       Real &a, Real &b, Real &c, Real &d) const {
    DD(P, T, r, s, a, b, c, d);
  }
  Real MinimumPressure() const { return g_p_min; }
  Real MaximumPressureAtTemperature(Real) const { return g_p_max; }
  Real MinimumTemperature() const { return Tmin(); }
  Real PressureFromDensityTemperature(Real, Real) const { return 1.0e6; }
};
template <typename M>
struct Wrap : AnyEos {
  Adapter<M> a;
  explicit Wrap(const M *m) { a.m = m; }
  void DE(Real P, Real T, Real &r, Real &s) const override {
    a.DensityEnergyFromPressureTemperature(P, T, nullptr, r, s);
  }
  void DD(Real P, Real T, Real &r, Real &s, Real &x1, Real &x2, Real &x3, Real &x4) const override {
    a.DensityEnergyDerivativesFromPressureTemperature(P, T, nullptr, r, s, x1, x2, x3, x4);
  }
  Real Tmin() const override { return a.Tmin; }
};
struct Idx {
  const AnyEos *e[2];
  PORTABLE_INLINE_FUNCTION const AnyEos &operator[](int i) const { return *e[i]; }
};
struct RawIdx {
  Real *d;
  PORTABLE_INLINE_FUNCTION Real &operator[](int i) const { return d[i]; }
};
struct NullLambda {
  PORTABLE_INLINE_FUNCTION Real *operator[](int) const { return nullptr; }
};

int g_failures = 0, g_checks = 0;
void check(bool ok, const char *what, Real got, Real want, Real tol) {
  ++g_checks;
  if (!ok) ++g_failures;
  std::printf("%s %-44s got=%.12g want=%.12g tol=%.1e\n", ok ? "PASS" : "FAIL", what, got, want,
              tol);
}
void check_rel(const char *what, Real got, Real want, Real tol) {
  const Real err = std::abs(got - want) / std::max(std::abs(want), 1e-300);
  check(std::isfinite(got) && err <= tol, what, got, want, tol);
}

MixParams Params(Real tol) {
  MixParams p;
  p.pte_rel_tolerance_p = p.pte_rel_tolerance_t = p.pte_rel_tolerance_e = tol;
  p.pte_rel_tolerance_v = p.pte_abs_tolerance_p = p.pte_abs_tolerance_t = tol;
  p.pte_max_iter_per_mat = 50;
  p.temperature_limit = 5000.0;
  return p;
}

} // namespace

int main() {
  const Vdw vdw;
  const Ideal ideal;
  const Stiff stiff;
  const Wrap<Vdw> w_vdw(&vdw);
  const Wrap<Ideal> w_ideal(&ideal);
  const Wrap<Stiff> w_stiff(&stiff);

  // ---- exact tie-line: T*, x*, Wood modulus, frozen c_v ----
  const Real tol = 1.0e-10;
  const MixParams params = Params(tol);
  const Real t_fracs[3] = {0.5, 0.7, 0.9};
  const Real qualities[3] = {0.05, 0.5, 0.95};
  const Real seeds[2] = {0.8, 1.25};
  const Real y_vdw = 0.9;
  for (Real tf : t_fracs) {
    for (Real xstar : qualities) {
      for (Real sf : seeds) {
        const Real Ts = tf * vdw.Tc;
        const Sat s = vdw.sat(Ts);
        Real tau_o, e_o, d1, d2, d3, d4;
        ideal.PT(s.p, Ts, tau_o, e_o, d1, d2, d3, d4);
        const Real Y[2] = {y_vdw, 1 - y_vdw};
        const Real tau_v = (1 - xstar) * s.tl + xstar * s.tv;
        const Real tau0 = Y[0] * tau_v + Y[1] * tau_o;
        const Real e0 = Y[0] * ((1 - xstar) * s.el + xstar * s.ev) + Y[1] * e_o;
        Real rho[2] = {1 / tau0, 1 / tau0}, vf[2] = {Y[0], Y[1]}, sie[2] = {0, 0};
        Real temp[2] = {0, 0}, press[2] = {0, 0};
        Real q[2], rl[2], rd[2], bm[2], cvk[2];
        int pin[2];
        DualKinkDiagnostics diag;
        diag.quality = q;
        diag.rho_light = rl;
        diag.rho_dense = rd;
        diag.bmod = bm;
        diag.cv = cvk;
        diag.pinned = pin;
        Real Tg = Ts * sf;
        Idx eos{{&w_vdw, &w_ideal}};
        const SolverStatus st =
            PTESolveDualKink(2, eos, 1.0, e0, RawIdx{rho}, RawIdx{vf}, RawIdx{sie}, RawIdx{temp},
                             RawIdx{press}, NullLambda{}, Tg, params, &diag);
        char label[96];
        std::snprintf(label, sizeof(label), "T*/Tc=%.1f x*=%.2f seed=%.2f", tf, xstar, sf);
        std::printf("-- %s: converged=%d kink=%zu iters=%zu phase2=%zu\n", label, st.converged,
                    st.pressure_jump_attempts, st.max_niter, st.safeguard_iters);
        check(st.converged && st.pressure_jump_attempts == 1 && diag.n_pinned == 1 &&
                  pin[0] == 1 && pin[1] == 0,
              "converged on the kink, vdW member pinned", st.converged, 1, 0);
        check_rel("T == T*", Tg, Ts, 1e-8);
        check_rel("P == P_sat(T*)", press[0], s.p, 1e-7);
        const Real x_out = (1 / rho[0] - s.tl) / (s.tv - s.tl);
        check(std::abs(x_out - xstar) < 1e-7, "lever quality from rho == x*", x_out, xstar, 1e-7);
        check(std::abs(q[0] - xstar) < 1e-7, "reported quality == x*", q[0], xstar, 1e-7);
        check_rel("rho_light == 1/tau_v(T*)", rl[0], 1 / s.tv, 1e-7);
        check_rel("rho_dense == 1/tau_l(T*)", rd[0], 1 / s.tl, 1e-7);
        const Real k_l = vdw.KS(s.tv, Ts), k_d = vdw.KS(s.tl, Ts);
        const Real a_l = xstar * s.tv / tau_v, a_d = (1 - xstar) * s.tl / tau_v;
        check_rel("Wood K_S == closed form", bm[0], 1.0 / (a_l / k_l + a_d / k_d), 1e-6);
        check_rel("frozen c_v == vdW c_v", cvk[0], vdw.cv, 1e-6);
        check(bm[1] == 0.0 && cvk[1] == 0.0, "unpinned member reports no modulus", bm[1], 0, 0);
        check_rel("vfrac closes", vf[0] + vf[1], 1.0, 1e-9);
      }
    }
  }

  // ---- bitwise parity with PTESolveDual where Newton alone closes the cell ----
  // Once with the default (fixed) dual, once with every legacy switch set.
  int parity_cells = 0;
  for (int legacy = 0; legacy < 2; ++legacy) {
  MixParams p6 = Params(1.0e-6);
  if (legacy) {
    p6.pte_dual_exact_certificate = false;
    p6.pte_dual_enthalpy_tstep = false;
    p6.pte_dual_enthalpy_slow_ratio = 0.0;
    p6.pte_dual_safeguard_fixes = false;
  }
  for (Real T0 : {400.0, 900.0, 2500.0}) {
    for (Real P0 : {1.0e7, 1.0e8, 1.0e9}) {
      for (Real sf : {0.7, 1.0, 1.6}) {
        Real tau_a, e_a, tau_b, e_b, d1, d2, d3, d4;
        ideal.PT(P0, T0, tau_a, e_a, d1, d2, d3, d4);
        stiff.PT(P0, T0, tau_b, e_b, d1, d2, d3, d4);
        const Real Y[2] = {0.3, 0.7};
        const Real tau0 = Y[0] * tau_a + Y[1] * tau_b, e0 = Y[0] * e_a + Y[1] * e_b;
        Real out[2][5][2], Tg_out[2];
        SolverStatus st[2];
        for (int arm = 0; arm < 2; ++arm) {
          Real rho[2] = {1 / tau0, 1 / tau0}, vf[2] = {Y[0], Y[1]}, sie[2] = {0, 0};
          Real temp[2] = {0, 0}, press[2] = {0, 0};
          Real Tg = T0 * sf;
          Idx eos{{&w_ideal, &w_stiff}};
          if (arm == 0)
            st[arm] = PTESolveDual(2, eos, 1.0, e0, RawIdx{rho}, RawIdx{vf}, RawIdx{sie},
                                   RawIdx{temp}, RawIdx{press}, NullLambda{}, Tg, p6);
          else
            st[arm] = PTESolveDualKink(2, eos, 1.0, e0, RawIdx{rho}, RawIdx{vf}, RawIdx{sie},
                                       RawIdx{temp}, RawIdx{press}, NullLambda{}, Tg, p6);
          for (int m = 0; m < 2; ++m) {
            out[arm][0][m] = rho[m];
            out[arm][1][m] = vf[m];
            out[arm][2][m] = sie[m];
            out[arm][3][m] = temp[m];
            out[arm][4][m] = press[m];
          }
          Tg_out[arm] = Tg;
        }
        if (!(st[0].converged && st[0].safeguard_iters == 0)) continue;
        ++parity_cells;
        bool same = st[1].converged && st[1].safeguard_iters == 0 &&
                    st[0].max_niter == st[1].max_niter && st[0].residual == st[1].residual &&
                    Tg_out[0] == Tg_out[1];
        for (int k = 0; k < 5; ++k)
          for (int m = 0; m < 2; ++m) same = same && out[0][k][m] == out[1][k][m];
        char label[96];
        std::snprintf(label, sizeof(label), "parity %s T=%g P=%g seed=%.1f (iters %zu)",
                      legacy ? "legacy" : "default", T0, P0, sf, st[0].max_niter);
        check(same, label, Tg_out[1], Tg_out[0], 0);
      }
    }
  }
  }
  check(parity_cells > 0, "parity: some cells closed by Newton alone", parity_cells, 1, 0);

  // ---- equilibrium below 1 barye: the box floor, not a 1-barye floor, brackets the volume root ----
  {
    g_p_min = 1.0e-12;
    g_p_max = 1.0e10;
    const MixParams pb = Params(1.0e-8);
    int n = 0;
    for (Real P0 : {1.0e-3, 1.0e-1}) {
      for (Real sf : {0.7, 1.6}) {
        const Real T0 = 1000.0;
        Real tau_a, e_a, tau_b, e_b, d1, d2, d3, d4;
        ideal.PT(P0, T0, tau_a, e_a, d1, d2, d3, d4);
        stiff.PT(P0, T0, tau_b, e_b, d1, d2, d3, d4);
        const Real Y[2] = {0.3, 0.7};
        const Real tau0 = Y[0] * tau_a + Y[1] * tau_b, e0 = Y[0] * e_a + Y[1] * e_b;
        Real rho[2] = {1 / tau0, 1 / tau0}, vf[2] = {Y[0], Y[1]}, sie[2] = {0, 0};
        Real temp[2] = {0, 0}, press[2] = {0, 0};
        Real Tg = T0 * sf;
        Idx eos{{&w_ideal, &w_stiff}};
        const SolverStatus st =
            PTESolveDualKink(2, eos, 1.0, e0, RawIdx{rho}, RawIdx{vf}, RawIdx{sie}, RawIdx{temp},
                             RawIdx{press}, NullLambda{}, Tg, pb);
        char label[96];
        std::snprintf(label, sizeof(label), "sub-barye P*=%g seed=%.1f", P0, sf);
        check(st.converged, label, st.converged, 1, 0);
        check_rel("sub-barye: T == T*", Tg, T0, 1e-6);
        check_rel("sub-barye: P == P*", press[0], P0, 1e-6);
        ++n;
      }
    }
    check(n == 4, "sub-barye cells run", n, 4, 0);
  }

  // ---- far seeds in a narrow box: the seed isochore has no volume root ----
  g_p_min = 1.0e5;
  g_p_max = 1.0e10;
  MixParams pf = Params(1.0e-8);
  pf.temperature_limit = 1.0e7;
  const AnyEos *pairs[3][2] = {{&w_ideal, &w_stiff}, {&w_vdw, &w_ideal}, {&w_vdw, &w_stiff}};
  auto solve = [&](int arm, int pr, Real T0, Real P0, Real Y0, Real sf, Real ps, Real &T_out,
                   Real &P_out) {
    Real tau[2], e[2];
    for (int m = 0; m < 2; ++m) {
      Real r, se;
      pairs[pr][m]->DE(P0, T0, r, se);
      tau[m] = 1 / r;
      e[m] = se;
    }
    const Real Y[2] = {Y0, 1 - Y0};
    const Real tau0 = Y[0] * tau[0] + Y[1] * tau[1], e0 = Y[0] * e[0] + Y[1] * e[1];
    Real rho[2] = {1 / tau0, 1 / tau0}, vf[2] = {Y[0], Y[1]}, sie[2] = {0, 0};
    Real temp[2] = {0, 0}, press[2] = {P0 * ps, P0 * ps};
    Real Tg = T0 * sf;
    Idx eos{{pairs[pr][0], pairs[pr][1]}};
    const SolverStatus st =
        (arm == 0) ? PTESolveDual(2, eos, 1.0, e0, RawIdx{rho}, RawIdx{vf}, RawIdx{sie},
                                  RawIdx{temp}, RawIdx{press}, NullLambda{}, Tg, pf)
                   : PTESolveDualKink(2, eos, 1.0, e0, RawIdx{rho}, RawIdx{vf}, RawIdx{sie},
                                      RawIdx{temp}, RawIdx{press}, NullLambda{}, Tg, pf);
    T_out = Tg;
    P_out = press[0];
    return st;
  };
  // Hot supercritical cells seeded at 1% and 10% of T*: without the isochore walk phase 2
  // stops on its first iteration.
  for (int pr = 0; pr < 3; ++pr) {
    for (Real Y0 : {0.1, 0.5, 0.9}) {
      for (Real sf : {0.01, 0.1}) {
        const Real T0 = 20000.0, P0 = 1.0e6;
        Real T, P;
        const SolverStatus st = solve(1, pr, T0, P0, Y0, sf, 1.0e4, T, P);
        char label[96];
        std::snprintf(label, sizeof(label), "far seed pair=%d Y=%.1f seed=%.2f closed by phase 2",
                      pr, Y0, sf);
        check(st.converged && st.stagnation_iters == 0 && st.safeguard_iters > 0, label,
              st.converged, 1, 0);
        check_rel("far seed: T == T*", T, T0, 1e-6);
        check_rel("far seed: P == P*", P, P0, 1e-5);
      }
    }
  }
  // No cell the dual closes is lost, over a grid of far seeds and warm-start pressures.
  long dual_ok = 0, lost = 0;
  for (int pr = 0; pr < 3; ++pr)
    for (Real T0 : {300.0, 600.0, 2000.0, 20000.0})
      for (Real P0 : {1.0e6, 1.0e8, 3.0e9})
        for (Real Y0 : {0.1, 0.5, 0.9})
          for (Real sf : {1e-2, 0.1, 10.0, 100.0})
            for (Real ps : {0.0, 1e-4, 1e4}) {
              Real Td, Pd, Tk, Pk;
              const bool d = solve(0, pr, T0, P0, Y0, sf, ps, Td, Pd).converged;
              const bool k = solve(1, pr, T0, P0, Y0, sf, ps, Tk, Pk).converged;
              dual_ok += d;
              lost += (d && !k);
            }
  check(dual_ok > 0 && lost == 0, "far seeds: no cell the dual closes is lost", lost, 0, 0);

  std::printf("%d/%d checks passed\n", g_checks - g_failures, g_checks);
  return g_failures == 0 ? 0 : 1;
}
