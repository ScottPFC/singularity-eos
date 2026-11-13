//------------------------------------------------------------------------------
// Python bindings for the single-cell, host-side PTE (pressure-temperature
// equilibrium) solvers of the mixed-cell closure.
//
// Design: the solvers are templates over an EOS indexer whose elements are the
// type-erased singularity::Variant. The Python EOS classes are the bare EOS
// types (IdealGas, StiffGas, ...), not Variants, so pte_solve() copies each
// bound EOS object into a Variant over a fixed, modifier-free subset of the EOS
// types. Everything else about the call is scalar and host side: no Kokkos, no
// device memory, no per-material lambda caches (nlambda == 0 for the subset).
//------------------------------------------------------------------------------
// clang-format off
#include "module.hpp"

#include <pybind11/stl.h>
#include <sstream>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

#ifdef SINGULARITY_BUILD_CLOSURE
#include <singularity-eos/closure/mixed_cell_models.hpp>
#include <singularity-eos/closure/pte_cyclic_rhoe.hpp>
#include <singularity-eos/closure/pte_dual.hpp>
#include <singularity-eos/closure/pte_dual_kink.hpp>
#include <singularity-eos/eos/eos_variant.hpp>

namespace {

// Modifier-free EOS subset that is bound in Python and valid without lambdas.
using PTEEos = singularity::Variant<IdealGas, StiffGas, Vinet, Gruneisen, JWL,
                                    DavisReactants, DavisProducts>;

struct EosIdx {
  PTEEos *d;
  PORTABLE_INLINE_FUNCTION PTEEos &operator[](const std::size_t i) const { return d[i]; }
};
struct NoLambda {
  PORTABLE_INLINE_FUNCTION Real *operator[](const std::size_t) const { return nullptr; }
};

PTEEos to_variant(const py::handle &h) {
#define SING_TRY_EOS(T) \
  if (py::isinstance<T>(h)) return PTEEos(h.cast<const T &>());
  SING_TRY_EOS(IdealGas)
  SING_TRY_EOS(StiffGas)
  SING_TRY_EOS(Vinet)
  SING_TRY_EOS(Gruneisen)
  SING_TRY_EOS(JWL)
  SING_TRY_EOS(DavisReactants)
  SING_TRY_EOS(DavisProducts)
#undef SING_TRY_EOS
  throw py::type_error(
      "pte_solve: unsupported EOS type; supported: IdealGas, StiffGas, Vinet, "
      "Gruneisen, JWL, DavisReactants, DavisProducts (modifiers and tables are not "
      "supported)");
}

struct PTEResult {
  std::vector<Real> vfrac, rho, sie, pressures, temperatures;
  Real pressure = 0, temperature = 0, residual = 0;
  bool converged = false;
  std::size_t niter = 0, safeguard_iters = 0;
  std::string solver;
  std::string repr() const {
    std::ostringstream ss;
    ss << "PTEResult(solver=" << solver << ", converged=" << converged
       << ", niter=" << niter << ", P=" << pressure << ", T=" << temperature << ")";
    return ss.str();
  }
};

template <template <typename, typename, typename> class Solver>
singularity::SolverStatus run_class_solver(std::size_t n, EosIdx eos, Real sie_tot,
                                           std::vector<Real> &rho, std::vector<Real> &vf,
                                           std::vector<Real> &sie, std::vector<Real> &T,
                                           std::vector<Real> &P, Real Tguess,
                                           const singularity::MixParams &params) {
  std::vector<Real> scratch(std::max(singularity::PTESolverRhoTRequiredScratch(n),
                                     singularity::PTESolverPTRequiredScratch(n)));
  NoLambda lambda;
  Solver<EosIdx, Real *, NoLambda> method(n, eos, 1.0, sie_tot, rho.data(), vf.data(),
                                          sie.data(), T.data(), P.data(), lambda,
                                          scratch.data(), Tguess, params);
  return singularity::PTESolver(method);
}

PTEResult pte_solve(const std::vector<py::object> &eos_list, const Real rho_cell,
                    const Real sie_cell, const std::vector<Real> &mass_fractions,
                    const std::string &solver, const py::object &tol,
                    const py::object &max_iter, const py::object &T_guess,
                    const py::object &vfrac_guess,
                    const singularity::MixParams *params_in) {
  const std::size_t n = eos_list.size();
  if (n == 0) throw std::invalid_argument("pte_solve: eos_list is empty");
  if (mass_fractions.size() != n)
    throw std::invalid_argument("pte_solve: mass_fractions must match eos_list in length");
  if (!(rho_cell > 0.0)) throw std::invalid_argument("pte_solve: rho_cell must be > 0");
  Real ysum = 0;
  for (Real y : mass_fractions) {
    if (!(y >= 0.0)) throw std::invalid_argument("pte_solve: mass_fractions must be >= 0");
    ysum += y;
  }
  if (!(ysum > 0.0)) throw std::invalid_argument("pte_solve: mass_fractions sum to 0");

  std::vector<PTEEos> eos_store;
  eos_store.reserve(n);
  for (const auto &h : eos_list) eos_store.push_back(to_variant(h));
  EosIdx eos{eos_store.data()};

  singularity::MixParams params = params_in ? *params_in : singularity::MixParams();
  if (!tol.is_none()) {
    const Real t = tol.cast<Real>();
    params.pte_rel_tolerance_p = params.pte_rel_tolerance_t = t;
    params.pte_rel_tolerance_e = params.pte_rel_tolerance_v = t;
    params.pte_abs_tolerance_p = std::min(params.pte_abs_tolerance_p, t);
    params.pte_abs_tolerance_t = std::min(params.pte_abs_tolerance_t, t);
    params.pte_rel_tolerance_p_sufficient = params.pte_rel_tolerance_e_sufficient =
        params.pte_rel_tolerance_v_sufficient = 1.0e2 * t;
    params.pte_rel_tolerance_t_sufficient = 10.0 * t;
    params.pte_abs_tolerance_p_sufficient = 1.0e2 * params.pte_abs_tolerance_p;
    params.pte_abs_tolerance_t_sufficient = 10.0 * params.pte_abs_tolerance_t;
  }
  if (!max_iter.is_none()) {
    const long mi = max_iter.cast<long>();
    if (mi < 1) throw std::invalid_argument("pte_solve: max_iter must be >= 1");
    // every solver caps iterations at nmat * pte_max_iter_per_mat
    params.pte_max_iter_per_mat = (static_cast<std::size_t>(mi) + n - 1) / n;
  }

  // Initial guess. mass density of material m is rhobar_m = vfrac_m * rho_m = Y_m * rho_cell
  // (vfrac_tot = 1), which the solvers hold fixed.
  std::vector<Real> Y(n), vf(n), rho(n), sie(n, sie_cell), T(n), P(n);
  for (std::size_t m = 0; m < n; ++m) Y[m] = mass_fractions[m] / ysum;
  if (!vfrac_guess.is_none()) {
    auto g = vfrac_guess.cast<std::vector<Real>>();
    if (g.size() != n) throw std::invalid_argument("pte_solve: vfrac_guess length mismatch");
    Real gs = 0;
    for (Real x : g) gs += x;
    for (std::size_t m = 0; m < n; ++m) vf[m] = g[m] / gs;
  } else {
    vf = Y;
  }
  for (std::size_t m = 0; m < n; ++m) {
    if (!(vf[m] > 0.0)) {
      // zero-mass or zero-vfrac material: give it a tiny share so rho stays finite
      vf[m] = 1.0e-12;
    }
    rho[m] = Y[m] * rho_cell / vf[m];
    if (!(rho[m] > 0.0)) rho[m] = rho_cell;
  }
  NoLambda lambda;
  Real Tg;
  if (!T_guess.is_none()) {
    Tg = T_guess.cast<Real>();
  } else {
    Tg = singularity::ApproxTemperatureFromRhoMatU(n, eos, rho_cell * sie_cell, rho.data(),
                                                    vf.data(), 0.0, lambda);
    // the estimate rails at its 1e-9 K floor when the equal-density energy is below the cold
    // curve (e.g. stiffened gases at the cell density); fall back to the solver default
    if (!(Tg > 1.0e-6)) Tg = params.default_tguess;
  }
  for (std::size_t m = 0; m < n; ++m) {
    T[m] = Tg;
    P[m] = eos[m].PressureFromDensityInternalEnergy(rho[m], sie_cell, lambda[m]);
  }

  using namespace singularity;
  SolverStatus st;
  Real Tout = 0;
  bool from_Tg = false;
  if (solver == "rhoT") {
    st = run_class_solver<PTESolverRhoT>(n, eos, sie_cell, rho, vf, sie, T, P, Tg, params);
  } else if (solver == "pt") {
    st = run_class_solver<PTESolverPT>(n, eos, sie_cell, rho, vf, sie, T, P, Tg, params);
  } else if (solver == "pt_analytic") {
    st = run_class_solver<PTESolverPTAnalytic>(n, eos, sie_cell, rho, vf, sie, T, P, Tg,
                                               params);
  } else if (solver == "pt_cyclic") {
    st = run_class_solver<PTESolverPTCyclic>(n, eos, sie_cell, rho, vf, sie, T, P, Tg,
                                             params);
  } else if (solver == "cyclic_rhoe") {
    Real Tgm = Tg;
    st = PTESolveCyclicRhoE(n, eos, 1.0, sie_cell, rho.data(), vf.data(), sie.data(),
                            T.data(), P.data(), lambda, Tgm, params);
    Tout = Tgm;
    from_Tg = true;
  } else if (solver == "dual") {
    Real Tgm = Tg;
    st = PTESolveDual(n, eos, 1.0, sie_cell, rho.data(), vf.data(), sie.data(), T.data(),
                      P.data(), lambda, Tgm, params);
    Tout = Tgm;
    from_Tg = true;
  } else if (solver == "dual_kink") {
    Real Tgm = Tg;
    st = PTESolveDualKink(n, eos, 1.0, sie_cell, rho.data(), vf.data(), sie.data(),
                          T.data(), P.data(), lambda, Tgm, params);
    Tout = Tgm;
    from_Tg = true;
  } else {
    throw std::invalid_argument(
        "pte_solve: unknown solver '" + solver +
        "'; one of rhoT, pt, pt_analytic, pt_cyclic, cyclic_rhoe, dual, dual_kink");
  }

  PTEResult r;
  r.solver = solver;
  r.vfrac = vf;
  r.rho = rho;
  r.sie = sie;
  r.pressures = P;
  r.temperatures = T;
  r.pressure = P[0];
  r.temperature = from_Tg ? Tout : T[0];
  r.converged = st.converged;
  r.niter = st.max_niter;
  r.safeguard_iters = st.safeguard_iters;
  r.residual = st.residual;
  return r;
}

} // namespace

void create_pte_bindings(py::module_ &m) {
  using singularity::MixParams;
  py::class_<MixParams>(m, "MixParams", "PTE solver parameters (see mixed_cell_models.hpp).")
      .def(py::init<>())
#define SING_MP(f) .def_readwrite(#f, &MixParams::f)
      SING_MP(verbose) SING_MP(iterate_t_guess) SING_MP(derivative_eps)
      SING_MP(pte_rel_tolerance_p) SING_MP(pte_rel_tolerance_t)
      SING_MP(pte_rel_tolerance_e) SING_MP(pte_rel_tolerance_v)
      SING_MP(pte_abs_tolerance_p) SING_MP(pte_abs_tolerance_t)
      SING_MP(pte_slow_convergence_thresh)
      SING_MP(pte_rel_tolerance_p_sufficient) SING_MP(pte_rel_tolerance_t_sufficient)
      SING_MP(pte_rel_tolerance_e_sufficient) SING_MP(pte_rel_tolerance_v_sufficient)
      SING_MP(pte_abs_tolerance_p_sufficient) SING_MP(pte_abs_tolerance_t_sufficient)
      SING_MP(pte_energy_norm) SING_MP(pte_energy_norm_floor)
      SING_MP(pte_max_iter_per_mat) SING_MP(line_search_alpha)
      SING_MP(line_search_max_iter) SING_MP(line_search_fac) SING_MP(vfrac_safety_fac)
      SING_MP(temperature_limit) SING_MP(pte_dual_exact_certificate)
      SING_MP(pte_dual_enthalpy_tstep) SING_MP(pte_dual_enthalpy_slow_ratio)
      SING_MP(pte_dual_safeguard_fixes) SING_MP(default_tguess) SING_MP(min_dtde)
      SING_MP(pte_small_step_tries) SING_MP(pte_small_step_thresh) SING_MP(pte_max_dpdv)
      SING_MP(pte_pressure_jump_enabled) SING_MP(pte_pressure_jump_bisect_iters)
      SING_MP(pte_pressure_jump_max_attempts) SING_MP(pte_pressure_jump_min_vfrac)
      SING_MP(pte_stagnation_thresh) SING_MP(pte_stagnation_tries)
#undef SING_MP
      ;

  py::class_<PTEResult>(m, "PTEResult")
      .def_readonly("vfrac", &PTEResult::vfrac)
      .def_readonly("rho", &PTEResult::rho)
      .def_readonly("sie", &PTEResult::sie)
      .def_readonly("pressures", &PTEResult::pressures)
      .def_readonly("temperatures", &PTEResult::temperatures)
      .def_readonly("pressure", &PTEResult::pressure)
      .def_readonly("temperature", &PTEResult::temperature)
      .def_readonly("converged", &PTEResult::converged)
      .def_readonly("niter", &PTEResult::niter)
      .def_readonly("safeguard_iters", &PTEResult::safeguard_iters)
      .def_readonly("residual", &PTEResult::residual)
      .def_readonly("solver", &PTEResult::solver)
      .def("__repr__", &PTEResult::repr);

  m.def("pte_solve", &pte_solve, py::arg("eos_list"), py::arg("rho_cell"),
        py::arg("sie_cell"), py::arg("mass_fractions"), py::arg("solver") = "rhoT",
        py::arg("tol") = py::none(), py::arg("max_iter") = py::none(),
        py::arg("T_guess") = py::none(), py::arg("vfrac_guess") = py::none(),
        py::arg("params") = nullptr,
        R"doc(Solve pressure-temperature equilibrium in one cell.

eos_list: bound EOS objects (IdealGas, StiffGas, Vinet, Gruneisen, JWL,
DavisReactants, DavisProducts). rho_cell: cell mass density; sie_cell: cell
specific internal energy; mass_fractions: per-material mass fractions (normalized).
solver: rhoT, pt, pt_analytic, pt_cyclic, cyclic_rhoe, dual or dual_kink.
tol: relative tolerance on p, T, e, v; max_iter: iteration cap (effective cap is
nmat * ceil(max_iter / nmat)). T_guess / vfrac_guess: optional initial guesses.
params: a MixParams for everything else (tol / max_iter are applied on top).
Returns a PTEResult with per-material vfrac, rho, sie, the common pressure and
temperature, converged and niter.)doc");
}
#else
void create_pte_bindings(py::module_ &) {}
#endif
