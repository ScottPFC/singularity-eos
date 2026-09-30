//------------------------------------------------------------------------------
// PTESolveDual's runtime switches and safeguard fixes.
//
//   * The legacy switch positions reproduce the pre-fix solver: recorded results of the
//   pinned
//     fork (79e77738) on a fixed set of cells, iteration counts exactly.
//   * Each switch, and each safeguard fix, owns a cell that converges with it and fails
//     without it.  The cells come from a scan of Clayton et al. Test 1 style random
//     trials (arXiv:2606.27726 Sec. 7.2.1) over the Table 6.1 ideal + stiffened gases.
//   * The exact admissibility certificate declines an unreachable target and reports it.
//------------------------------------------------------------------------------

#include <cmath>
#include <cstddef>

#include <singularity-eos/closure/mixed_cell_models.hpp>
#include <singularity-eos/closure/pte_dual.hpp>

#include <catch2/catch_test_macros.hpp>

#include "pte_dual_test_eos.hpp"

using namespace pte_dual_test;
using singularity::MixParams;
using singularity::SolverStatus;

namespace {

// The two boxes the cells are drawn in (barye): Clayton's Test 1 box and Test 2's lower
// floor.
constexpr Real kHighBox = 10.0;
constexpr Real kLowBox = 1.0e-2;

struct Cell {
  Real Y0, P0, T0, Pg, Tg; // mass fraction of the ideal gas, true state, seed
};

struct Result {
  SolverStatus status;
  Real P, T;
};

Result Solve(const Real p_min, const Cell &c, const MixParams &params) {
  BoxedStiffGas eos[2];
  ClaytonGases(eos, p_min, 1.0e13, 1.0e-3);
  // A uniform energy gauge (Clayton's e_ref = 0.2 kJ/g) so the 1 + |e0| normaliser is
  // physical.
  eos[0].q = eos[1].q = 2.0e9;
  const Real Y[2] = {c.Y0, 1.0 - c.Y0};
  Real tau0 = 0.0, e0 = 0.0;
  for (int m = 0; m < 2; ++m) {
    Real r, e;
    eos[m].DensityEnergyFromPressureTemperature(c.P0, c.T0, NoLambda{}, r, e);
    tau0 += Y[m] / r;
    e0 += Y[m] * e;
  }
  Real rho[2], vf[2], sie[2], temp[2], press[2];
  for (int m = 0; m < 2; ++m) {
    rho[m] = 1.0 / tau0;
    vf[m] = Y[m];
    sie[m] = e0;
    temp[m] = c.Tg;
    press[m] = c.Pg;
  }
  Real Tg = c.Tg;
  Result r;
  r.status = singularity::PTESolveDual(
      2, Idx<BoxedStiffGas>{eos}, 1.0, e0, Idx<Real>{rho}, Idx<Real>{vf}, Idx<Real>{sie},
      Idx<Real>{temp}, Idx<Real>{press}, NoLambda{}, Tg, params);
  r.P = press[0];
  r.T = Tg;
  return r;
}

MixParams Base() {
  MixParams p;
  p.pte_rel_tolerance_e = p.pte_rel_tolerance_v = 1.0e-10;
  p.pte_max_iter_per_mat = 50;
  p.temperature_limit = 1.0e5;
  return p;
}

MixParams Legacy() {
  MixParams p = Base();
  p.pte_dual_exact_certificate = false;
  p.pte_dual_enthalpy_tstep = false;
  p.pte_dual_enthalpy_slow_ratio = 0.0;
  p.pte_dual_safeguard_fixes = false;
  return p;
}

bool AtTruth(const Result &r, const Cell &c) {
  return r.status.converged && std::abs(r.P - c.P0) <= 1.0e-6 * c.P0 &&
         std::abs(r.T - c.T0) <= 1.0e-6 * c.T0;
}

// Pinned-fork results: converged, max_niter, safeguard_iters and the returned (press[0],
// T).
struct Golden {
  Real p_min;
  Cell cell;
  bool converged;
  std::size_t iters, safeguard;
  Real P, T;
};
const Golden kGolden[] = {
    {kHighBox,
     {0x1.7c0dd68f900aep-3, 0x1.2d64f55fc4508p+11, 0x1.9f28775c4d464p+11,
      0x1.927d3d0b4e36ap+9, 0x1.1653424bfe2efp-7},
     true,
     8,
     3,
     0x1.2d64f55fc4508p+11,
     0x1.9f28775c4d465p+11},
    {kHighBox,
     {0x1.f8c3c0bd8150cp-5, 0x1.a835e26a766a1p+22, 0x1.c76e8bce04ab1p-2,
      0x1.3f0df9dba603cp+35, 0x1.9066b923298e4p-8},
     true,
     10,
     1,
     0x1.a835e26a7688p+22,
     0x1.c76e8bce04aedp-2},
    {kHighBox,
     {0x1.d46f9010584f5p-2, 0x1.b5d185a9143a3p+24, 0x1.082c0abcad3f6p+11,
      0x1.daf2b81ee6fefp+21, 0x1.84ed49e552aa1p+1},
     true,
     8,
     1,
     0x1.b5d185a9142e3p+24,
     0x1.082c0abcad3e5p+11},
    {kHighBox,
     {0x1.8e2824c489578p-2, 0x1.5048799d159edp+16, 0x1.92d9267552eap+11,
      0x1.11868d71d5a21p+29, 0x1.39fe8b4660428p-9},
     true,
     7,
     2,
     0x1.5048799d2b87bp+16,
     0x1.92d926756d292p+11},
    {kHighBox,
     {0x1.e413921ce85c3p-1, 0x1.bccd998ab0075p+30, 0x1.aac383b343c6dp+14,
      0x1.e4b1fc5210517p+39, 0x1.efaec9722207ep+15},
     true,
     4,
     1,
     0x1.bccd998ab0079p+30,
     0x1.aac383b343c6fp+14},
    {kHighBox,
     {0x1.a113b75feb7fap-2, 0x1.baf566e51c887p+13, 0x1.61868e22b6fd3p+1,
      0x1.bd440ceb32e76p+33, 0x1.f644e754ccd2dp+5},
     true,
     8,
     1,
     0x1.baf566e51c84dp+13,
     0x1.61868e22b6fc6p+1},
    {kHighBox,
     {0x1.6dcd736141bd4p-1, 0x1.b454226cd0a4p+38, 0x1.343acc8e505afp+16,
      0x1.00a345eca7aa5p+8, 0x1.69d33e04d7908p+0},
     false,
     2,
     1,
     0x1.00a345eca7aa5p+8,
     0x1.69d33e04d7908p+0},
    {kHighBox,
     {0x1.3518695d66c1cp-3, 0x1.f00355521135fp+37, 0x1.0bb7d1718e1d3p+11,
      0x1.e4e9c75e01cedp+22, 0x1.77b65ff7764b1p+0},
     false,
     2,
     1,
     0x1.e4e9c75e01cedp+22,
     0x1.77b65ff7764b1p+0},
    {kHighBox,
     {0x1.fde3127d53302p-3, 0x1.b6e88d2725a79p+14, 0x1.590b716ea71c4p+16,
      0x1.96963cd19c82cp+37, 0x1.e6ad837d356f5p-8},
     false,
     1,
     0,
     0x1.96963cd19c82cp+37,
     0x1.e6ad837d356f5p-8},
    {kHighBox,
     {0x1.13e3e42246822p-1, 0x1.647630c707fc2p+34, 0x1.c02c871c70d37p+5,
      0x1.986d36d7b711ep+7, 0x1.3f3e36ce10a5ep-10},
     false,
     2,
     1,
     0x1.986d36d7b711ep+7,
     0x1.3f3e36ce10a5ep-10},
    {kHighBox,
     {0x1.fdd8ca2b7d4dep-2, 0x1.e591cd1a855cdp+35, 0x1.3b77c6a361b68p+6,
      0x1.0cde36407acb9p+34, 0x1.dfff245fa29e2p+12},
     true,
     13,
     0,
     0x1.e591cd19f77eap+35,
     0x1.3b77c6a361b65p+6},
    {kHighBox,
     {0x1.9b36d93e7b319p-4, 0x1.f4b76d7960c28p+20, 0x1.8a2d4ac4b738ap-10,
      0x1.e1767076722c5p+14, 0x1.7f4786d7d3631p-4},
     true,
     17,
     0,
     0x1.f4b76d797e635p+20,
     0x1.8a2d4ac4bd06ep-10},
    {kLowBox,
     {0x1.82343cb029e42p-3, 0x1.f585619b8b5b4p+40, 0x1.91a9d00255c22p+4,
      0x1.2482d0337d77dp+20, 0x1.9c15a09d2b8d8p-5},
     true,
     15,
     1,
     0x1.f585619b8b5bep+40,
     0x1.91a9d00255c2cp+4},
    {kLowBox,
     {0x1.3d936ddd26aacp-2, 0x1.f6c9dc1da9d87p+17, 0x1.48061a9990e1ap+0,
      0x1.66b15a4e6774ap+35, 0x1.a03fb819332ap+0},
     true,
     4,
     1,
     0x1.f6c9dc1da9d6fp+17,
     0x1.48061a9990e0ep+0},
    {kLowBox,
     {0x1.d016b10a57da2p-3, 0x1.65c1c245deef8p+42, 0x1.5ed972957e155p-10,
      0x1.98ac8b5c6683ap+4, 0x1.9c00c17680d8cp+2},
     true,
     19,
     1,
     0x1.65c1c245e1395p+42,
     0x1.5ed972958053dp-10},
    {kLowBox,
     {0x1.a4fd5fcbdd716p-3, 0x1.337443fb4fc4bp+18, 0x1.ae240def269f1p-2,
      0x1.142890e3ba48bp-3, 0x1.1f9bca7dd532dp-10},
     true,
     5,
     1,
     0x1.337443fb4e195p+18,
     0x1.ae240def269cfp-2},
    {kLowBox,
     {0x1.8e17d7f66a0c4p-3, 0x1.1e3580f48de2fp+14, 0x1.0c3ee1502f64ep+14,
      0x1.ccf20e9790f0cp+25, 0x1.f40ac16bae444p-1},
     true,
     6,
     1,
     0x1.1e3580f48e204p+14,
     0x1.0c3ee1502f9e5p+14},
    {kLowBox,
     {0x1.7051b4a006467p-1, 0x1.0677c960e758ep+31, 0x1.42ee0c307602bp+9,
      0x1.328b3a6e29052p+40, 0x1.eb86600cfc493p+1},
     true,
     8,
     2,
     0x1.0677c960e758bp+31,
     0x1.42ee0c307602cp+9},
    {kLowBox,
     {0x1.7589f99d34932p-2, 0x1.83a6d6b6127fdp-6, 0x1.9ec8fe09cba93p+9,
      0x1.54b40c9ae2787p+8, 0x1.c3d650e660adp-2},
     false,
     5,
     4,
     0x1.54b40c9ae2787p+8,
     0x1.c3d650e660adp-2},
    {kLowBox,
     {0x1.1bf5f99efb397p-2, 0x1.5400757e30e09p-1, 0x1.592efd82e5d79p+7,
      0x1.6376ea31eb181p+5, 0x1.1d267a7b8b3f1p+1},
     false,
     2,
     1,
     0x1.6376ea31eb181p+5,
     0x1.1d267a7b8b3f1p+1},
    {kLowBox,
     {0x1.b38e9821f65e8p-1, 0x1.6b68568821926p-5, 0x1.9f65c788efa9cp+13,
      0x1.2ac0bcd852f2dp+2, 0x1.2a1f6dd515d48p+11},
     false,
     2,
     1,
     0x1.2ac0bcd852f2dp+2,
     0x1.2a1f6dd515d48p+11},
    {kLowBox,
     {0x1.7640807275b4dp-1, 0x1.99e72e985a2a2p-5, 0x1.49e333f34200fp+8,
      0x1.f5ad19ac5782ep+27, 0x1.11a83ef26dc0bp-3},
     false,
     4,
     3,
     0x1.f5ad19ac5782ep+27,
     0x1.11a83ef26dc0bp-3},
    {kLowBox,
     {0x1.e9a0b0c0b4b6p-2, 0x1.62c5d5d78a782p+18, 0x1.21f7cb69a51b1p+1,
      0x1.fefa51e83b67dp+22, 0x1.7aa000f28863fp+1},
     true,
     6,
     0,
     0x1.62c5d5d78a78cp+18,
     0x1.21f7cb69a51b6p+1},
    {kLowBox,
     {0x1.ad85063e1cee3p-1, 0x1.6a80ae59c6324p+16, 0x1.111dccb6e7cf5p-6,
      0x1.0af7e85713728p+21, 0x1.42d93cb54a156p-3},
     true,
     8,
     0,
     0x1.6a80ae8917751p+16,
     0x1.111dccda8e28ep-6},
};

} // namespace

TEST_CASE("PTESolveDual defaults are the fixed solver", "[PTESolveDual]") {
  const MixParams p;
  CHECK(p.pte_dual_exact_certificate);
  CHECK(p.pte_dual_enthalpy_tstep);
  CHECK(p.pte_dual_enthalpy_slow_ratio == 0.7);
  CHECK(p.pte_dual_safeguard_fixes);
}

TEST_CASE("PTESolveDual legacy switches reproduce the pre-fix solver", "[PTESolveDual]") {
  // Iteration and safeguard counts pin the trajectory exactly.  The returned state is
  // compared at 1e-13 rather than bitwise so a compiler that contracts FMAs still passes;
  // the offline comparison in port_dual_fixes/ is the bitwise one.
  for (const Golden &g : kGolden) {
    const Result r = Solve(g.p_min, g.cell, Legacy());
    INFO("cell P0 = " << g.cell.P0 << " T0 = " << g.cell.T0 << " seed " << g.cell.Pg
                      << ", " << g.cell.Tg);
    CHECK(r.status.converged == g.converged);
    CHECK(r.status.max_niter == g.iters);
    CHECK(r.status.safeguard_iters == g.safeguard);
    CHECK(std::abs(r.P - g.P) <= 1.0e-13 * std::abs(g.P));
    CHECK(std::abs(r.T - g.T) <= 1.0e-13 * std::abs(g.T));
    // The legacy decline is the heuristic's alone; the exact test never runs.
    CHECK((r.status.admissibility == singularity::kAdmissibilityNotTested ||
           r.status.admissibility == singularity::kAdmissibilityHeuristic));
  }
}

TEST_CASE("PTESolveDual safeguard fixes each rescue a cell", "[PTESolveDual]") {
  MixParams fixes_off = Base();
  fixes_off.pte_dual_safeguard_fixes = false;
  SECTION("box P floor: the equilibrium lies below 1 barye") {
    const Cell c{0x1.dbcac6cae67fap-2, 0x1.1e7614ce37245p-3, 0x1.5cc35a8c3bd61p+7,
                 0x1.126a1a87fa48fp+38, 0x1.f6930fbc14e49p-2};
    CHECK(AtTruth(Solve(kLowBox, c, Base()), c));
    CHECK_FALSE(Solve(kLowBox, c, fixes_off).status.converged);
  }
  SECTION("projection: an empty coordinate bracket steps to the box end") {
    const Cell c{0x1.88931866fec8cp-3, 0x1.1da0a8ac7c517p+5, 0x1.694eb2d8eea17p+16,
                 0x1.e1985e8a45afep+8, 0x1.e17254cd2f6aap+13};
    CHECK(AtTruth(Solve(kHighBox, c, Base()), c));
    CHECK_FALSE(Solve(kHighBox, c, fixes_off).status.converged);
  }
  SECTION(
      "no-op intersection: a tangent step that clamps back onto the iterate is refused") {
    const Cell c{0x1.7c0dd68f900aep-3, 0x1.2d64f55fc4508p+11, 0x1.9f28775c4d464p+11,
                 0x1.927d3d0b4e36ap+9, 0x1.1653424bfe2efp-7};
    CHECK(AtTruth(Solve(kHighBox, c, Base()), c));
    CHECK_FALSE(Solve(kHighBox, c, fixes_off).status.converged);
  }
}

TEST_CASE("PTESolveDual switches each rescue a cell", "[PTESolveDual]") {
  SECTION("enthalpy T-step: the isoenergetic sub-solve runs to the iteration cap") {
    const Cell c{0x1.e10157bbac8e4p-3, 0x1.bc4f29dd8440fp+31, 0x1.d3ae0344bd8f7p+12,
                 0x1.8c1371e7b9b36p+12, 0x1.d5f669e5e3c0ep+2};
    MixParams off = Base();
    off.pte_dual_enthalpy_tstep = false;
    CHECK(AtTruth(Solve(kHighBox, c, Base()), c));
    CHECK_FALSE(Solve(kHighBox, c, off).status.converged);
  }
  SECTION("slow-ratio fallback: the enthalpy step zig-zags without it") {
    const Cell c{0x1.0a8211249e9bp-1, 0x1.9fe0bb32b61c7p+22, 0x1.c76d894ac343ap+12,
                 0x1.5c65b5e0bed3fp+36, 0x1.92cae4808becp+15};
    MixParams off = Base();
    off.pte_dual_enthalpy_slow_ratio = 0.0;
    CHECK(AtTruth(Solve(kHighBox, c, Base()), c));
    CHECK_FALSE(Solve(kHighBox, c, off).status.converged);
  }
  SECTION("exact certificate: the heuristic declines a cell whose state exists") {
    const Cell c{0x1.cbc72472e3728p-4, 0x1.5542866b51f4dp+18, 0x1.08906630fd828p-4,
                 0x1.9b36357bcff25p+42, 0x1.512bc2284867ap-1};
    MixParams off = Base();
    off.pte_dual_exact_certificate = false;
    const Result on = Solve(kHighBox, c, Base());
    CHECK(AtTruth(on, c));
    CHECK(on.status.admissibility == singularity::kAdmissibilityReachable);
    const Result heuristic = Solve(kHighBox, c, off);
    CHECK_FALSE(heuristic.status.converged);
    CHECK(heuristic.status.admissibility == singularity::kAdmissibilityHeuristic);
  }
}

TEST_CASE("PTESolveDual declines an unreachable target on the exact certificate",
          "[PTESolveDual]") {
  // e = Cv T (...) + q > q for every state, so a target below the gauge q has no
  // solution.  The density is chosen so the floor isochore (T = 1e-3 K) still has its
  // volume root inside the P box; otherwise the certificate can only report
  // kAdmissibilityUndecided.
  BoxedStiffGas eos[2];
  ClaytonGases(eos, kHighBox, 1.0e13, 1.0e-3);
  eos[0].q = eos[1].q = 2.0e9;
  const Real Y[2] = {0.4, 0.6};
  Real tau0 = 0.0;
  for (int m = 0; m < 2; ++m) {
    Real r, e;
    eos[m].DensityEnergyFromPressureTemperature(1.0e12, 1.0e3, NoLambda{}, r, e);
    tau0 += Y[m] / r;
  }
  const Real e0 = 0.5 * eos[0].q;
  Real rho[2], vf[2], sie[2], temp[2], press[2];
  for (int m = 0; m < 2; ++m) {
    rho[m] = 1.0 / tau0;
    vf[m] = Y[m];
    sie[m] = e0;
    temp[m] = 1.0e3;
    press[m] = 1.0e12;
  }
  Real Tg = 1.0e3;
  const SolverStatus st = singularity::PTESolveDual(
      2, Idx<BoxedStiffGas>{eos}, 1.0, e0, Idx<Real>{rho}, Idx<Real>{vf}, Idx<Real>{sie},
      Idx<Real>{temp}, Idx<Real>{press}, NoLambda{}, Tg, Base());
  CHECK_FALSE(st.converged);
  CHECK(st.admissibility == singularity::kAdmissibilityUnreachable);
  // A consistent EOS: the assembled Hessian's cross terms agree to round-off.
  CHECK(st.max_asymmetry < 1.0e-10);
}
