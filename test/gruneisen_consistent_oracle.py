#!/usr/bin/env python3
"""Independent reference for the consistent Steinberg Gruneisen model (eos_gruneisen.hpp).

mpmath (40 digits): the compression 0 K isentrope by direct quadrature of
de_c/drho = P(rho, e_c)/rho^2 (variation of constants, mpmath.quad), the expansion Vinet
shape and bounded tail in closed form, rho_lim by mpmath.findroot. Prints the gold values
used in test_eos_gruneisen.cpp / test_eos_gruneisen_consistent.cpp:

    python3 gruneisen_consistent_oracle.py
"""

import mpmath as mp

mp.mp.dps = 40


class Oracle:
    def __init__(self, C0, s1, s2, s3, G0, b, rho0, T0, P0, Cv, kfrac=0.02):
        m = mp.mpf
        self.C0, self.s1, self.s2, self.s3 = m(C0), m(s1), m(s2), m(s3)
        self.G0, self.b, self.rho0, self.T0, self.P0, self.Cv = (
            m(G0), m(b), m(rho0), m(T0), m(P0), m(Cv))
        self.ec0 = -self.Cv * self.T0
        self.pc0 = self.P0 + self.G0 * self.rho0 * self.ec0
        # K0 = rho0 d p_c/drho at rho0+ (numerical derivative of the compression curve)
        self.K0 = self.rho0 * mp.diff(lambda r: self.cold_comp(r)[1], self.rho0, direction=1)
        Kp = max(4 * self.s1 - 1, 2)
        self.ev = mp.mpf(3) / 2 * (Kp - 1)
        self.kmin = kfrac * self.K0
        xs = ((self.ev - 1) + mp.sqrt((self.ev - 1) ** 2 + 8 * self.ev)) / (2 * self.ev)
        x = mp.findroot(lambda x: self.kv(x) - self.kmin, (1, xs), solver="anderson")
        self.rho_lim = self.rho0 / x**3
        self.e_lim, self.p_lim = self.vinet(self.rho_lim)

    def hug(self, eta):
        s = self.s1 * eta + self.s2 * eta**2 + self.s3 * eta**3
        PH = self.P0 + self.C0**2 * self.rho0 * eta / (1 - s) ** 2
        return PH, (PH + self.P0) * eta / self.rho0 / 2

    def rhoG(self, rho):
        return self.G0 * self.rho0 + self.b * (rho - self.rho0)

    def cold_comp(self, rho):
        rho = mp.mpf(rho)
        # e_c' = (P_H + rhoG (e_c - E_H)) / rho^2, linear in e_c
        lth = lambda r: self.b * mp.log(r / self.rho0) + (self.G0 - self.b) * (1 - self.rho0 / r)
        def f(r):
            PH, EH = self.hug(1 - self.rho0 / r)
            return (PH - self.rhoG(r) * EH) / r**2 * mp.exp(-lth(r))
        ec = mp.exp(lth(rho)) * (self.ec0 + mp.quad(f, [self.rho0, rho]))
        PH, EH = self.hug(1 - self.rho0 / rho)
        return ec, PH + self.rhoG(rho) * (ec - EH)

    def kv(self, x):
        return self.K0 * mp.exp(self.ev * (1 - x)) / x**2 * (2 - x + self.ev * x * (1 - x))

    def vinet(self, rho):
        x = mp.cbrt(self.rho0 / rho)
        y = self.ev * (1 - x)
        p = self.pc0 + 3 * self.K0 * (1 - x) / x**2 * mp.exp(y)
        e = (self.ec0 - self.pc0 * (1 / rho - 1 / self.rho0)
             + 9 * self.K0 / (self.rho0 * self.ev**2) * (1 - (1 - y) * mp.exp(y)))
        return e, p

    def cold(self, rho):
        rho = mp.mpf(rho)
        if rho >= self.rho0:
            return self.cold_comp(rho)
        if rho >= self.rho_lim:
            return self.vinet(rho)
        pinf = self.p_lim - self.kmin
        sl = self.kmin / self.rho_lim
        e = self.e_lim + pinf * (1 / self.rho_lim - 1 / rho) + sl * mp.log(rho / self.rho_lim)
        return e, pinf + sl * rho

    def gamma(self, rho):
        return self.G0 if rho < self.rho0 else self.rhoG(rho) / rho

    def P_re(self, rho, e):
        rho = mp.mpf(rho)
        ec, pc = self.cold(rho)
        return pc + rho * self.gamma(rho) * (e - ec)

    def T_re(self, rho, e):
        return (e - self.cold(rho)[0]) / self.Cv

    def BS_re(self, rho, e):
        rho = mp.mpf(rho)
        dPdr = mp.diff(lambda r: self.P_re(r, e), rho, direction=-1 if rho < self.rho0 else 1)
        return rho * dPdr + self.P_re(rho, e) * self.gamma(rho)


def main():
    cu = Oracle(0.394e6, 1.489, 0, 0, 2.02, 0.47, 8.93, 298.0, 0.0, 0.383e-5 * 1e12)
    print("Cu: K0 %.17e rho_lim %.17e p_lim %.17e k_min %.17e p_inf %.17e" % (
        cu.K0, cu.rho_lim, cu.p_lim, cu.kmin, cu.p_lim - cu.kmin))
    for rho, e in [(8.0, 1e9), (9.0, 5e8), (9.5, 1e8), (1e-3, 0.0)]:
        print("rho %g e %g: P %.17e B_S %.17e T %.17e" % (
            rho, e, cu.P_re(rho, e), cu.BS_re(rho, e), cu.T_re(rho, e)))
    for rho in [1e-3, 3.0, 6.0, 7.0, 8.0, 8.93, 9.5, 11.0, 14.0, 20.0]:
        ec, pc = cu.cold(rho)
        print("cold rho %g: e_c %.17e p_c %.17e" % (rho, ec, pc))


if __name__ == "__main__":
    main()
