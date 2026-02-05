#!/usr/bin/env python3
"""
Module contains several classes that represent vertical structure of accretion disc in case
of (tabular) MESA opacity and(or) EOS.

Class MesaVerticalStructure --  for MESA opacities and EoS with radiative energy transport.
Class MesaIdealVerticalStructure -- for MESA opacities and ideal gas EoS with radiative energy transport.
Class MesaVerticalStructureAd -- for MESA opacities and EoS with adiabatic energy transport.
Class MesaVerticalStructureRadAd -- for MESA opacities and EoS with radiative+adiabatic energy transport.
Class MesaVerticalStructureRadConv -- for MESA opacities and EoS with radiative+convective energy transport.

Class MesaVerticalStructureExternalIrradiation -- for MESA opacities and EoS with radiative energy transport
    and advanced external irradiation scheme from (Mescheryakov et al. 2011).
Class MesaVerticalStructureRadAdExternalIrradiation -- for MESA opacities and EOS
    with radiative+adiabatic energy transport and advanced external irradiation scheme
    from (Mescheryakov et al. 2011).
Class MesaVerticalStructureRadConvExternalIrradiation -- for MESA opacities and EOS
    with radiative+convective energy transport and advanced external irradiation scheme
    from (Mescheryakov et al. 2011).

Class MesaVerticalStructureExternalIrradiationZeroAssumption -- for MESA opacities and EOS
    with radiative energy transport and simple external irradiation scheme via T_irr or C_irr.
Class MesaVerticalStructureRadAdExternalIrradiationZeroAssumption -- for MESA opacities and EOS
    with radiative+adiabatic energy transport and simple external irradiation scheme via T_irr or C_irr.
Class MesaVerticalStructureRadConvExternalIrradiationZeroAssumption -- for MESA opacities and EOS
    with radiative+convective energy transport and simple external irradiation scheme via T_irr or C_irr.

"""
import os
from types import FunctionType

import numpy as np
from astropy import constants as const
from astropy import units
from scipy.integrate import simpson
from scipy.optimize import root, least_squares, brentq

from alpha_disc.vs import BaseVerticalStructure, Vars, IdealGasMixin, RadiativeTempGradient, PgasPradNotConvergeError
from alpha_disc.magn import MagneticField
sigmaSB = const.sigma_sb.cgs.value
sigmaT = const.sigma_T.cgs.value
atomic_mass = const.u.cgs.value
c = const.c.cgs.value
proton_mass = const.m_p.cgs.value
G = const.G.cgs.value
M_sun = const.M_sun.cgs.value

try:
    from opacity import Opac
except ModuleNotFoundError as e:
    raise ModuleNotFoundError('Mesa2py is not installed') from e


class IrrNotConvergeError(Exception):
    def __init__(self, Sigma0_par, z0r):
        self.Sigma0_par = Sigma0_par
        self.z0r = z0r
        self.message = f'Not converged, try higher Sigma0_par or smaller z0r estimations. ' \
                       f'Current estimations are Sigma0_par = {self.Sigma0_par:g}, z0r = {self.z0r:g}.'

    def __str__(self):
        return self.message


class PphNotConvergeError(Exception):
    def __init__(self, func_Pph, P_ph=None, z0r=None, Sigma0_par=None):
        self.func_Pph = abs(func_Pph)
        self.P_ph = P_ph
        self.z0r = z0r
        self.Sigma0_par = Sigma0_par
        if self.P_ph is None:
            self.message = f'P_ph = P(z=z0) not converged. fun = {self.func_Pph}'
        elif Sigma0_par is None:
            self.message = f'P_ph = P(z=z0) not converged. Try another start estimation for P_ph. ' \
                           f'Current P_ph_0 = {self.P_ph:g}.\n' \
                           f'Also you can try another z0r estimation. Current estimations is z0r = {self.z0r:g}.'
        else:
            self.message = f'P_ph = P(z=z0) not converged. Try another start estimation for P_ph. ' \
                           f'Current P_ph_0 = {self.P_ph:g}.\n' \
                           f'Also you can try another z0r and Sigma0_par estimations. ' \
                           f'Current estimations are Sigma0_par = {self.Sigma0_par:g}, z0r = {self.z0r:g}.'

    def __str__(self):
        return self.message


class BaseMesaVerticalStructure(BaseVerticalStructure):
    def __init__(self, Mx, alpha, r, F, eps=1e-5, mu=0.6, abundance='solar', **kwargs):
        super().__init__(Mx, alpha, r, F, eps=eps, mu=mu, **kwargs)
        self.mesaop = Opac(abundance)


class BaseExternalIrradiation(BaseMesaVerticalStructure):
    def __init__(self, Mx, alpha, r, F, nu_irr, spectrum_irr, L_X_irr, spectrum_irr_par,
                 args_spectrum_irr=(), kwargs_spectrum_irr={}, cos_theta_irr=None, cos_theta_irr_exp=1 / 12,
                 eps=1e-5, abundance='solar'):
        super().__init__(Mx, alpha, r, F, eps=eps, mu=0.6, abundance=abundance)

        if spectrum_irr is None:
            raise Exception("spectrum_irr must be a function or an array-like, not None.")
        if L_X_irr is None:
            raise Exception("L_X_irr must be a double, not None.")
        if nu_irr is None:
            raise Exception("nu_irr must be an array-like, not None.")

        if isinstance(spectrum_irr, FunctionType):
            if spectrum_irr_par == 'nu':
                self.nu_irr = nu_irr
                spectrum_irr = spectrum_irr(self.nu_irr, *args_spectrum_irr, **kwargs_spectrum_irr) / simpson(
                    spectrum_irr(self.nu_irr, *args_spectrum_irr, **kwargs_spectrum_irr), self.nu_irr)
            elif spectrum_irr_par == 'E_in_keV':
                self.nu_irr = (nu_irr * units.keV).to('Hz', equivalencies=units.spectral()).value
                spectrum_irr = spectrum_irr(nu_irr, *args_spectrum_irr, **kwargs_spectrum_irr) / simpson(
                    spectrum_irr(nu_irr, *args_spectrum_irr, **kwargs_spectrum_irr), nu_irr) * \
                               units.Hz.to('keV', equivalencies=units.spectral())
            else:
                raise Exception("spectrum_irr_par must be 'nu' or 'E_in_keV', not None or anything else.")
        else:
            if len(nu_irr) != len(spectrum_irr):
                raise Exception("'nu_irr' and 'spectrum_irr' must have the same size.")
            if spectrum_irr_par == 'nu':
                self.nu_irr = nu_irr
                spectrum_irr = spectrum_irr
            elif spectrum_irr_par == 'E_in_keV':
                self.nu_irr = (nu_irr * units.keV).to('Hz', equivalencies=units.spectral()).value
                spectrum_irr = spectrum_irr * units.Hz.to('keV', equivalencies=units.spectral())
            else:
                raise Exception("spectrum_irr_par must be 'nu' or 'E_in_keV', not None or anything else.")

        F_nu_irr = L_X_irr / (4 * np.pi * r ** 2) * spectrum_irr
        self.F_nu_irr = F_nu_irr
        self.Sigma0_par = self.Sigma0_init()
        if cos_theta_irr is None:
            self.cos_theta_irr_key = False
            self.cos_theta_irr_exp = cos_theta_irr_exp
        else:
            self.cos_theta_irr_key = True
            self.cos_theta_irr = cos_theta_irr
        self.T_irr = None
        self.C_irr = None
        self.Q_irr = None
        self.Sigma_ph = None
        self.P_ph_0 = None
        self.P_ph_key = False
        self.P_ph_parameter = None

    @property
    def cos_theta_irr(self):
        if self.cos_theta_irr_key:
            return self.__cos_theta_irr
        else:
            return self.cos_theta_irr_exp * self.z0 / self.r

    @cos_theta_irr.setter
    def cos_theta_irr(self, value):
        self.__cos_theta_irr = value

    def Sigma0_init(self):
        return 2 * 9.73e-24 * (self.r / 1e10) ** (-11 / 10) * self.F ** (7 / 10) * (self.Mx / M_sun) ** (
                -1 / 10) * self.alpha ** (-4 / 5)


class BaseExternalIrradiationZeroAssumption(BaseMesaVerticalStructure):
    def __init__(self, Mx, alpha, r, F, C_irr=None, T_irr=None, eps=1e-5, abundance='solar', F_in=0):
        super().__init__(Mx, alpha, r, F, eps=eps, mu=0.6, abundance=abundance)
        h = np.sqrt(self.GM * self.r)
        eta_accr = 0.1
        rg = 2 * self.GM / c ** 2
        r_in = 3 * rg
        func = 1 - np.sqrt(r_in / r)
        Mdot = (self.F - F_in) / (h * func)
        if Mdot < 0:
            raise Exception(f'Mdot = {Mdot:g} g/s < 0, incorrect F_in = {F_in:g} g*cm^2/s^2.')
        if C_irr is None and T_irr is None:
            raise Exception('C_irr or T_irr is required.')
        elif T_irr is not None and C_irr is None:
            self.T_irr = T_irr
            self.Q_irr = sigmaSB * self.T_irr ** 4
            self.C_irr = self.Q_irr / (eta_accr * Mdot * c ** 2) * (4 * np.pi * self.r ** 2)
        elif C_irr is not None and T_irr is None:
            self.C_irr = C_irr
            self.Q_irr = self.C_irr * eta_accr * Mdot * c ** 2 / (4 * np.pi * self.r ** 2)
            self.T_irr = (self.Q_irr / sigmaSB) ** (1 / 4)
        else:
            raise Exception('Only one of (C_irr, T_irr) is required.')
        self.P_ph_0 = None
        self.P_ph_key = False
        self.P_ph_parameter = None


class MesaGasMixin:
    def law_of_rho(self, P, T, full_output):
        return self.mesaop.rho(P, T, full_output=full_output)  # input P=Pgas


class MesaOpacityMixin:
    def law_of_opacity(self, rho, T, lnfree_e):
        return self.mesaop.kappa(rho, T, lnfree_e=lnfree_e)


class AdiabaticTempGradient:
    """
    Temperature gradient class. Returns adiabatic d(lnT)/d(lnPtotal) from MESA.

    """

    def dlnTdlnP(self, y, t):
        rho, eos = self.mesaop.rho(y[Vars.P] * self.P_norm, y[Vars.T] * self.T_norm, full_output=True)
        return eos.grad_ad


class RadiativeAdiabaticGradient:
    """
    Temperature gradient class.
    If gradient is over-adiabatic, then returns adiabatic d(lnT)/d(lnPtotal),
    else calculates radiative d(lnT)/d(lnPtotal).

    """

    def dlnTdlnP(self, y, t):
        rho, eos = self.rho(y, True)
        varkappa = self.opacity(y, lnfree_e=eos.lnfree_e)
        # P_full = y[Vars.P] * self.P_norm + 4 * sigmaSB / (3 * c) * y[Vars.T] ** 4 * self.T_norm ** 4
        P_full = y[Vars.P] * self.P_norm + self.rad_pressure(y[Vars.T] * self.T_norm)

        if t == 1:
            dTdz_der = (self.dQdz(y, t) / y[Vars.T] ** 3) * 3 * varkappa * rho * self.z0 * self.Q_norm / (
                    16 * sigmaSB * self.T_norm ** 4)
            dP_full_der = - rho * self.omegaK ** 2 * self.z0 ** 2
            dlnTdlnP_rad = (P_full / y[Vars.T]) * (dTdz_der / dP_full_der)
        else:
            dTdz = (abs(y[Vars.Q]) / y[Vars.T] ** 3) * 3 * varkappa * rho * self.z0 * self.Q_norm / (
                    16 * sigmaSB * self.T_norm ** 4)

            dP_full = rho * (1 - t) * self.omegaK ** 2 * self.z0 ** 2
            dlnTdlnP_rad = (P_full / y[Vars.T]) * (dTdz / dP_full)

        if dlnTdlnP_rad < eos.grad_ad:
            return dlnTdlnP_rad
        else:
            return eos.grad_ad


class RadConvTempGradient:
    """
    Temperature gradient class. Calculates d(lnT)/d(lnPtotal) in presence of convection
    according to mixing length theory.

    """

    def dlnTdlnP(self, y, t):
        rho, eos = self.rho(y, True)
        varkappa = self.opacity(y, lnfree_e=eos.lnfree_e)
        P_full = y[Vars.P] * self.P_norm + self.rad_pressure(y[Vars.T] * self.T_norm)

        if t == 1:
            dTdz_der = (self.dQdz(y, t) / y[Vars.T] ** 3) * 3 * varkappa * rho * self.z0 * self.Q_norm / (
                    16 * sigmaSB * self.T_norm ** 4)
            dP_full_der = - rho * self.omegaK ** 2 * self.z0 ** 2
            dlnTdlnP_rad = (P_full / y[Vars.T]) * (dTdz_der / dP_full_der)
        else:
            dTdz = (abs(y[Vars.Q]) / y[Vars.T] ** 3) * 3 * varkappa * rho * self.z0 * self.Q_norm / (
                    16 * sigmaSB * self.T_norm ** 4)

            dP_full = rho * (1 - t) * self.omegaK ** 2 * self.z0 ** 2
            dlnTdlnP_rad = (P_full / y[Vars.T]) * (dTdz / dP_full)

        if dlnTdlnP_rad < eos.grad_ad:
            return dlnTdlnP_rad
        if t == 1:
            return dlnTdlnP_rad

        alpha_ml = 1.5
        H_p = P_full / (rho * self.omegaK ** 2 * self.z0 * (1 - t) + self.omegaK * np.sqrt(P_full * rho))
        H_ml = alpha_ml * H_p
        omega = varkappa * rho * H_ml
        A = 9 / 8 * omega ** 2 / (3 + omega ** 2)
        der = eos.dlnRho_dlnT_const_Pgas
        if der > 0:
            der = -1

        VV = -((3 + omega ** 2) / (
                3 * omega)) ** 2 * eos.c_p ** 2 * rho ** 2 * H_ml ** 2 * self.omegaK ** 2 * self.z0 * (1 - t) / (
                     512 * sigmaSB ** 2 * y[Vars.T] ** 6 * self.T_norm ** 6 * H_p) * der * (
                     dlnTdlnP_rad - eos.grad_ad)
        V = 1 / np.sqrt(VV)

        coeff = [2 * A, V, V ** 2, - V]
        try:
            x = np.roots(coeff)
        except np.linalg.LinAlgError as ex:
            print('Error with convective energy transport calculation.')
            # print('LinAlgError, coeff[2A, V, V ** 2, -V] = ', coeff)
            raise ex

        x = [a.real for a in x if a.imag == 0 and 0.0 < a.real < 1.0]
        if len(x) != 1:
            raise ValueError('Error with convective energy transport calculation.')
        x = x[0]

        dlnTdlnP_conv = eos.grad_ad + (dlnTdlnP_rad - eos.grad_ad) * x * (x + V)
        return dlnTdlnP_conv


class ExternalIrradiation:
    @staticmethod
    def sigma_d_nu(nu):  # cross-section in cm2 (Morrison & McCammon, 1983) from 0.03 to 10 keV
        E = (nu * units.Hz).to('keV', equivalencies=units.spectral()).value

        if 0.030 <= E <= 0.100:
            c_0 = 17.3
            c_1 = 608.1
            c_2 = -2150.0
        elif 0.100 < E <= 0.284:
            c_0 = 34.6
            c_1 = 267.9
            c_2 = -476.1
        elif 0.284 < E <= 0.400:
            c_0 = 78.1
            c_1 = 18.8
            c_2 = 4.3
        elif 0.400 < E <= 0.532:
            c_0 = 71.4
            c_1 = 66.8
            c_2 = -51.4
        elif 0.532 < E <= 0.707:
            c_0 = 95.5
            c_1 = 145.8
            c_2 = -61.1
        elif 0.707 < E <= 0.867:
            c_0 = 308.9
            c_1 = -380.6
            c_2 = 294.0
        elif 0.867 < E <= 1.303:
            c_0 = 120.6
            c_1 = 169.3
            c_2 = -47.7
        elif 1.303 < E <= 1.840:
            c_0 = 141.3
            c_1 = 146.8
            c_2 = -31.5
        elif 1.840 < E <= 2.471:
            c_0 = 202.7
            c_1 = 104.7
            c_2 = -17.0
        elif 2.471 < E <= 3.210:
            c_0 = 342.7
            c_1 = 18.7
            c_2 = 0.0
        elif 3.210 < E <= 4.038:
            c_0 = 352.2
            c_1 = 18.7
            c_2 = 0.0
        elif 4.038 < E <= 7.111:
            c_0 = 433.9
            c_1 = -2.4
            c_2 = 0.75
        elif 7.111 < E <= 8.331:
            c_0 = 629.0
            c_1 = 30.9
            c_2 = 0.0
        elif 8.331 < E <= 10.000:
            c_0 = 701.2
            c_1 = 25.2
            c_2 = 0.0

        result = (c_0 + c_1 * E + c_2 * E ** 2) * E ** (-3) * 1e-24  # cross-section in cm2
        return result

    def J_tot(self, F_nu, y, tau_Xray, t):  # mean intensity as function of tau
        sigma_sc = 0.34  # not exactly sigma_sc in case of not fully ionized gas
        k_d_nu = np.array([self.sigma_d_nu(nu) for nu in self.nu_irr]) / proton_mass  # cross-section per proton mass
        tau = (sigma_sc + k_d_nu) * \
              (y[Vars.S] * self.sigma_norm + 2 * self.Sigma_ph) / 2  # tau = varkappa * (Sigma + 2 * Sigma_ph) / 2

        lamb = sigma_sc / (sigma_sc + k_d_nu)
        k = np.sqrt(3 * (1 - lamb))
        zeta_0 = self.cos_theta_irr

        D_nu = 3 * lamb * zeta_0 ** 2 / (1 - k ** 2 * zeta_0 ** 2)

        C_nu = D_nu * (1 + np.exp(-tau_Xray / zeta_0) + 2 / (3 * zeta_0) * (1 + np.exp(-tau_Xray / zeta_0))) / (
                1 + np.exp(-k * tau_Xray) + 2 * k / 3 * (1 + np.exp(-k * tau_Xray)))

        J_tot = F_nu / (4 * np.pi) * (
                C_nu * (np.exp(-k * tau) + np.exp(-k * (tau_Xray - tau))) +
                (1 - D_nu) * (np.exp(-tau / zeta_0) + np.exp(-(tau_Xray - tau) / zeta_0))
        )

        if np.any(tau_Xray - tau < 0):
            raise IrrNotConvergeError(self.Sigma0_par, self.z0 / self.r)
        return J_tot

    def H_tot(self, F_nu, tau_Xray, Pph):  # eddington flux at the photosphere
        # Pph = P_total_ph
        sigma_sc = 0.34  # not exactly sigma_sc in case of not fully ionized gas
        k_d_nu = np.array([self.sigma_d_nu(nu) for nu in self.nu_irr]) / proton_mass  # cross-section per proton mass
        tau = (sigma_sc + k_d_nu) * Pph / (self.z0 * self.omegaK ** 2)  # tau at the photosphere
        lamb = sigma_sc / (sigma_sc + k_d_nu)
        k = np.sqrt(3 * (1 - lamb))
        zeta_0 = self.cos_theta_irr

        D_nu = 3 * lamb * zeta_0 ** 2 / (1 - k ** 2 * zeta_0 ** 2)

        C_nu = D_nu * (1 + np.exp(-tau_Xray / zeta_0) + 2 / (3 * zeta_0) * (1 + np.exp(-tau_Xray / zeta_0))) / (
                1 + np.exp(-k * tau_Xray) + 2 * k / 3 * (1 + np.exp(-k * tau_Xray)))

        H_tot = F_nu * (
                k * C_nu / 3 * (np.exp(-k * tau) - np.exp(-k * (tau_Xray - tau))) +
                (zeta_0 - D_nu / (3 * zeta_0)) * (np.exp(-tau / zeta_0) - np.exp(-(tau_Xray - tau) / zeta_0))
        )

        if np.any(tau_Xray - tau < 0):
            raise IrrNotConvergeError(self.Sigma0_par, self.z0 / self.r)
        return H_tot

    def epsilon(self, y, t):
        rho, eos = self.rho(y, full_output=True)
        sigma_sc = 0.34  # not exactly sigma_sc in case of not fully ionized gas
        k_d_nu = np.array([self.sigma_d_nu(nu) for nu in self.nu_irr]) / proton_mass  # cross-section per proton mass
        tau_Xray = (sigma_sc + k_d_nu) * (self.Sigma0_par + 2 * self.Sigma_ph)
        epsilon = 4 * np.pi * rho * simpson(k_d_nu * self.J_tot(self.F_nu_irr, y, tau_Xray, t), self.nu_irr)
        return epsilon

    def Q_irr_ph(self, Pph):
        # Pph = P_total_ph
        sigma_sc = 0.34  # not exactly sigma_sc in case of not fully ionized gas
        k_d_nu = np.array([self.sigma_d_nu(nu) for nu in self.nu_irr]) / proton_mass  # cross-section per proton mass
        tau_Xray = (sigma_sc + k_d_nu) * (self.Sigma0_par + 2 * Pph / (self.z0 * self.omegaK ** 2))
        Qirr = simpson(self.H_tot(self.F_nu_irr, tau_Xray, Pph), self.nu_irr)
        return Qirr

    def photospheric_pressure_equation_irr(self, tau, P, Pph):
        # Pph = P_total_ph, P = P_total
        T = (self.Teff ** 4 * (1 / 2 + 3 * tau / 4) + self.Q_irr_ph(Pph) / sigmaSB) ** (1 / 4)
        rho, eos = self.law_of_rho(P - 4 * sigmaSB / (3 * c) * T ** 4, T, True)
        varkappa = self.law_of_opacity(rho, T, lnfree_e=eos.lnfree_e)
        return self.z0 * self.omegaK ** 2 / varkappa

    def P_ph_irr(self, Pph):
        # Pph = P_total_ph, solution is P_total
        solution = self.photospheric_pressure_equation_irr(tau=2 / 3, P=Pph, Pph=Pph) * 2 / 3
        return solution

    def Q_initial(self, Pph):
        result = 1 + self.Q_irr_ph(Pph) / self.Q_norm
        return result

    def initial(self):
        """
        Initial conditions.

        Returns
        -------
        array

        """

        def fun_P_ph(x):
            # x = P_total_ph
            result = abs(x) - self.P_ph_irr(abs(x))
            return result

        if not self.P_ph_key:
            if self.P_ph_0 is None:
                self.P_ph_0 = self.P_ph() + 4 * sigmaSB / (3 * c) * self.Teff ** 4  # P_ph_0 = P_total
            sign_P_ph = fun_P_ph(self.P_ph_0)
            if np.isnan(sign_P_ph):
                raise PphNotConvergeError(sign_P_ph, self.P_ph_0, self.z0 / self.r, self.Sigma0_par)
            if sign_P_ph > 0:
                factor = 0.5
            else:
                factor = 2.0
            P_ph_a = self.P_ph_0

            while True:
                self.P_ph_0 *= factor
                if sign_P_ph * fun_P_ph(self.P_ph_0) < 0:
                    break
                if np.isnan(fun_P_ph(self.P_ph_0)):
                    factor = 1.02
                if factor != 1.02:
                    P_ph_a = self.P_ph_0
                if self.P_ph_0 > P_ph_a and factor == 1.02:
                    raise PphNotConvergeError(fun_P_ph(self.P_ph_0), self.P_ph_0, self.z0 / self.r, self.Sigma0_par)

            P_ph, res = brentq(fun_P_ph, self.P_ph_0, P_ph_a, full_output=True)
            if abs(fun_P_ph(P_ph)) > 1e-7:
                raise PphNotConvergeError(fun_P_ph(P_ph))
            self.P_ph_0 = P_ph * 1.98
            P_ph = abs(res.root)
            if self.fitted:
                self.P_ph_parameter = abs(res.root)
                P_ph = self.P_ph_parameter
                self.P_ph_key = True
        else:
            P_ph = self.P_ph_parameter

        self.Sigma_ph = P_ph / (self.z0 * self.omegaK ** 2)
        Q_initial = self.Q_initial(Pph=P_ph)
        Qirr = (Q_initial - 1) * self.Q_norm
        self.C_irr = Qirr / simpson(self.F_nu_irr, self.nu_irr)
        self.T_irr = (Qirr / sigmaSB) ** (1 / 4)
        self.Q_irr = Qirr
        P_gas_ph = P_ph - 4 * sigmaSB / (3 * c) * (self.Teff ** 4 + self.T_irr ** 4)
        if P_gas_ph < 0 or np.isnan(P_gas_ph):
            raise PgasPradNotConvergeError(P_gas=P_gas_ph,
                                           P_rad=4 * sigmaSB / (3 * c) * (self.Teff ** 4 + self.T_irr ** 4),
                                           t=0.0, z0r=self.z0 / self.r, Sigma0_par=self.Sigma0_par)
        y = np.empty(4, dtype=np.float64)
        y[Vars.S] = 0
        y[Vars.P] = P_gas_ph / self.P_norm
        y[Vars.Q] = Q_initial
        y[Vars.T] = (Q_initial * self.Q_norm / sigmaSB) ** (1 / 4) / self.T_norm  # Tph^4 = Teff^4 + Tirr^4
        return y

    def dQdz(self, y, t):
        w_r_phi = self.viscosity(y)
        result = -(3 / 2) * self.z0 * self.omegaK * w_r_phi / self.Q_norm - self.epsilon(y, t) * self.z0 / self.Q_norm
        return result

    def dq(self, x, norm, verbose):
        self.Sigma0_par = abs(x[1]) * norm
        self.z0 = abs(x[0]) * self.r
        q_c = np.array([self.y_c()[Vars.Q], self.Sigma0_par / self.parameters_C()[4] - 1])
        if verbose:
            print(f'z0r, Sigma0_par = {abs(x[0]):g}, {self.Sigma0_par:g}')
        return q_c

    def fit(self, z0r_estimation=None, Sigma0_estimation=None, verbose=False, P_ph_0=None):
        """
        Solves optimisation problem and calculate the vertical structure.

        Parameters
        ----------
        z0r_estimation : double
            Start estimation of z0r free parameter to fit the structure.
            Default is None, the estimation is calculated automatically.
        Sigma0_estimation : double
            Start estimation of Sigma0 free parameter to fit the structure.
            Default is None, the estimation is calculated automatically.
        verbose : bool
            Whether to print values of (z0r, Sigma0_par) at each iteration.
            Default is False, the fitting process performs silently.
        P_ph_0 : double
            Start estimation of pressure at the photosphere (pressure boundary condition).
            Default is None, the estimation is calculated automatically.

        Returns
        -------
        result : scipy.optimize.OptimizeResult
            Result of optimisation.

        """

        if z0r_estimation is None:
            x0_z0r = self.z0_init() / self.r
        else:
            x0_z0r = z0r_estimation

        if Sigma0_estimation is None:
            norm = self.Sigma0_init()
        else:
            norm = Sigma0_estimation
        self.P_ph_0 = P_ph_0
        self.fitted = False
        self.P_ph_key = False

        cost_root = -1
        try:
            result = root(self.dq, x0=np.array([x0_z0r, 1]), args=(norm, verbose), method='hybr')
            cost_root = result.fun[0] ** 2 + result.fun[1] ** 2
            result.update({'cost': cost_root})
        except Exception:
            try:
                result = least_squares(self.dq, x0=np.array([x0_z0r, 1]), loss='linear',
                                       kwargs={'norm': norm, 'verbose': verbose})
                result.cost *= 2
            except Exception as ex:
                raise ex from None

        if cost_root > 1e-16:
            try:
                result_least_squares = least_squares(self.dq, x0=np.array([x0_z0r, 1]), loss='linear',
                                                     kwargs={'norm': norm, 'verbose': verbose})
                result_least_squares.cost *= 2
                if result_least_squares.cost < cost_root:
                    result = result_least_squares
            except Exception:
                pass

        result.x = abs(result.x) * np.array([1, norm])
        self.Sigma0_par = result.x[1]
        self.z0 = result.x[0] * self.r
        self.fitted = True  # doesn't mean that structure converged successfully, need only for Pph
        return result


class ExternalIrradiationZeroAssumption:
    def photospheric_pressure_equation_irr(self, tau, P):  # P = P_gas
        T = (self.Teff ** 4 * (1 / 2 + 3 * tau / 4) + self.T_irr ** 4) ** (1 / 4)
        rho, eos = self.law_of_rho(P, T, True)
        varkappa = self.law_of_opacity(rho, T, lnfree_e=eos.lnfree_e)
        return self.z0 * self.omegaK ** 2 / varkappa

    def P_ph_irr(self, Pph):  # Pph = P_gas
        # solution is P_total, result is P_gas
        solution = self.photospheric_pressure_equation_irr(tau=2 / 3, P=Pph) * 2 / 3
        result = solution - 4 * sigmaSB / (3 * c) * (self.Teff ** 4 + self.T_irr ** 4)  # P_gas = P_tot - P_rad
        if result < 0:
            return np.nan
        return result

    def initial(self):
        """
        Initial conditions.

        Returns
        -------
        array

        """

        def fun_P_ph(x):
            # x = P_gas_ph
            return abs(x) - self.P_ph_irr(abs(x))

        if not self.P_ph_key:
            if self.P_ph_0 is None:
                self.P_ph_0 = self.P_ph()
            sign_P_ph = fun_P_ph(self.P_ph_0)
            if np.isnan(sign_P_ph):
                raise PphNotConvergeError(sign_P_ph, self.P_ph_0, self.z0 / self.r)

            if sign_P_ph > 0:
                factor = 0.5
            else:
                factor = 2.0
            P_ph_a = self.P_ph_0

            while True:
                self.P_ph_0 *= factor
                if sign_P_ph * fun_P_ph(self.P_ph_0) < 0:
                    break
                if np.isnan(fun_P_ph(self.P_ph_0)):
                    factor = 1.02
                if factor != 1.02:
                    P_ph_a = self.P_ph_0
                if self.P_ph_0 > P_ph_a and factor == 1.02:
                    raise PphNotConvergeError(fun_P_ph(self.P_ph_0), self.P_ph_0, self.z0 / self.r)

            P_ph, res = brentq(fun_P_ph, self.P_ph_0, P_ph_a, full_output=True)
            if abs(fun_P_ph(P_ph)) > 1e-7:
                raise PphNotConvergeError(fun_P_ph(P_ph))
            self.P_ph_0 = P_ph * 1.98
            P_ph = abs(res.root)
            if self.fitted:
                self.P_ph_parameter = abs(res.root)
                P_ph = self.P_ph_parameter
                self.P_ph_key = True
        else:
            P_ph = self.P_ph_parameter

        if P_ph < 0 or np.isnan(P_ph):
            raise PgasPradNotConvergeError(P_gas=P_ph,
                                           P_rad=4 * sigmaSB / (3 * c) * (self.Teff ** 4 + self.T_irr ** 4),
                                           t=0.0, z0r=self.z0 / self.r)
        Q_initial = self.Q_initial()
        y = np.empty(4, dtype=np.float64)
        y[Vars.S] = 0
        y[Vars.P] = P_ph / self.P_norm
        y[Vars.Q] = Q_initial
        y[Vars.T] = (self.Teff ** 4 + self.T_irr ** 4) ** (1 / 4) / self.T_norm
        return y

    def fit(self, z0r_estimation=None, verbose=False, P_ph_0=None):
        """
        Solves optimisation problem and calculates the vertical structure.

        Parameters
        ----------
        z0r_estimation : double
            Start estimation of z0r free parameter to fit the structure.
            Default is None, the estimation is calculated automatically.
        verbose : bool
            Whether to print value of z0r at each iteration.
            Default is False, the fitting process performs silently.
        P_ph_0 : double
            Start estimation of pressure at the photosphere (pressure boundary condition).
            Default is None, the estimation is calculated automatically.

        Returns
        -------
        double and scipy.optimize.RootResults
            The value of normalised unknown free parameter z_0 / r and result of optimisation.

        """

        self.P_ph_0 = P_ph_0
        self.fitted = False
        self.P_ph_key = False
        return super().fit(z0r_estimation=z0r_estimation, verbose=verbose)


# Classes with MESA opacity without Irradiation
class MesaVerticalStructure(MesaGasMixin, MesaOpacityMixin, RadiativeTempGradient, BaseMesaVerticalStructure):
    """
    Vertical structure class for MESA opacities and EoS with radiative energy transport.

    """
    pass

# Classes with MESA opacity without Irradiation
class MesaVerticalStructureMagnetic(MagneticField, MesaGasMixin, MesaOpacityMixin, 
                                    RadiativeTempGradient, BaseMesaVerticalStructure):
    """
    Vertical structure class for MESA opacities and EoS with radiative energy transport.
    Includes the magnetic field.

    """
    pass


class MesaIdealGasVerticalStructure(IdealGasMixin, MesaOpacityMixin, RadiativeTempGradient, BaseMesaVerticalStructure):
    """
    Vertical structure class for MESA opacities and ideal gas EoS with radiative energy transport.

    """
    pass

class MesaIdealGasVerticalStructureMagnetic(MagneticField, IdealGasMixin, 
                                            MesaOpacityMixin, RadiativeTempGradient, BaseMesaVerticalStructure):
    """
    Vertical structure class for MESA opacities and ideal gas EoS with radiative energy transport.
    Includes the magnetic field.

    """
    pass

class MesaVerticalStructureAd(MesaGasMixin, MesaOpacityMixin, AdiabaticTempGradient, BaseMesaVerticalStructure):
    """
    Vertical structure class for MESA opacities and EoS with adiabatic energy transport.

    """
    pass

class MesaVerticalStructureAdMagnetic(MagneticField, MesaGasMixin, MesaOpacityMixin,
                                      AdiabaticTempGradient, BaseMesaVerticalStructure):
    """
    Vertical structure class for MESA opacities and EoS with adiabatic energy transport.
    Includes the magnetic field.

    """
    pass



class MesaVerticalStructureRadAd(MesaGasMixin, MesaOpacityMixin, RadiativeAdiabaticGradient, BaseMesaVerticalStructure):
    """
    Vertical structure class for MESA opacities and EoS with radiative+adiabatic energy transport.

    """
    pass

class MesaVerticalStructureRadAdMagnetic(MagneticField, MesaGasMixin, MesaOpacityMixin,
                                         RadiativeAdiabaticGradient, BaseMesaVerticalStructure):
    """
    Vertical structure class for MESA opacities and EoS with radiative+adiabatic energy transport.
    Includes the magnetic field.


    """
    pass


class MesaVerticalStructureRadConv(MesaGasMixin, MesaOpacityMixin, RadConvTempGradient, BaseMesaVerticalStructure):
    """
    Vertical structure class for MESA opacities and EoS with radiative+convective energy transport.

    """
    pass

class MesaVerticalStructureRadConvMagnetic(MagneticField, MesaGasMixin, 
                                MesaOpacityMixin, RadConvTempGradient, BaseMesaVerticalStructure):
    """
    Vertical structure class for MESA opacities and EoS with radiative+convective energy transport.
    Includes the magnetic field.

    """
    pass


# Classes with MESA opacity and with advanced Irradiation (Mescheryakov et al. 2011)
class MesaVerticalStructureExternalIrradiation(MesaGasMixin, MesaOpacityMixin, RadiativeTempGradient,
                                               ExternalIrradiation, BaseExternalIrradiation):
    """
    Vertical structure class for MESA opacities and EoS with radiative energy transport
    and advanced external irradiation scheme from (Mescheryakov et al. 2011).

    """
    pass


class MesaVerticalStructureRadAdExternalIrradiation(MesaGasMixin, MesaOpacityMixin, RadiativeAdiabaticGradient,
                                                    ExternalIrradiation, BaseExternalIrradiation):
    """
    Vertical structure class for MESA opacities and EoS with radiative+adiabatic energy transport
    and advanced external irradiation scheme from (Mescheryakov et al. 2011).

    """
    pass


class MesaVerticalStructureRadConvExternalIrradiation(MesaGasMixin, MesaOpacityMixin, RadConvTempGradient,
                                                      ExternalIrradiation, BaseExternalIrradiation):
    """
    Vertical structure class for MESA opacities and EoS with radiative+convective energy transport
    and advanced external irradiation scheme from (Mescheryakov et al. 2011).

    """
    pass


# Classes with MESA opacity and with Zero Assumption Irradiation
class MesaVerticalStructureExternalIrradiationZeroAssumption(MesaGasMixin, MesaOpacityMixin, RadiativeTempGradient,
                                                             ExternalIrradiationZeroAssumption,
                                                             BaseExternalIrradiationZeroAssumption):
    """
    Vertical structure class for MESA opacities and EoS with radiative energy transport
    and simple external irradiation scheme via T_irr or C_irr.

    """
    pass


class MesaVerticalStructureRadAdExternalIrradiationZeroAssumption(MesaGasMixin, MesaOpacityMixin,
                                                                  RadiativeAdiabaticGradient,
                                                                  ExternalIrradiationZeroAssumption,
                                                                  BaseExternalIrradiationZeroAssumption):
    """
    Vertical structure class for MESA opacities and EoS with radiative+adiabatic energy transport
    and simple external irradiation scheme via T_irr or C_irr.

    """
    pass


class MesaVerticalStructureRadConvExternalIrradiationZeroAssumption(MesaGasMixin, MesaOpacityMixin, RadConvTempGradient,
                                                                    ExternalIrradiationZeroAssumption,
                                                                    BaseExternalIrradiationZeroAssumption):
    """
    Vertical structure class for MESA opacities and EoS with radiative+convective energy transport
    and simple external irradiation scheme via T_irr or C_irr.

    """
    pass


def main():
    M = 10 * M_sun
    alpha = 0.01
    Mdot = 1e18
    rg = 2 * G * M / c ** 2
    r = 400 * rg
    print('Calculating structure and making a structure plot. '
          '\nStructure with tabular MESA opacity and EOS.'
          '\nChemical composition is solar.')
    print(f'M = {M:g} grams = {M / M_sun:g} M_sun \nr = {r:g} cm = {r / rg:g} rg '
          f'\nalpha = {alpha:g} \nMdot = {Mdot:g} g/s')
    h = np.sqrt(G * M * r)
    r_in = 3 * rg
    F = Mdot * h * (1 - np.sqrt(r_in / r))
    vs = MesaVerticalStructureRadConv(M, alpha, r, F)
    z0r, result = vs.fit()
    if result.converged:
        print('The vertical structure has been calculated successfully.')
    print('z0/r = ', z0r)
    varkappa_C, rho_C, T_C, P_C, Sigma0 = vs.parameters_C()
    print('Prad/Pgas_c = ', 4 * sigmaSB / (3 * c) * T_C ** 4 / P_C)

    t = np.linspace(0, 1, 100)
    S, P, Q, T = vs.integrate(t)[0]
    import matplotlib.pyplot as plt
    plt.plot(1 - t, S, label=r'$\hat{\Sigma}$')
    plt.plot(1 - t, P, label=r'$\hat{P}$')
    plt.plot(1 - t, Q, label=r'$\hat{Q}$')
    plt.plot(1 - t, T, label=r'$\hat{T}$')
    plt.xlabel('$z / z_0$')
    plt.title(rf'$M = {M / M_sun:g}\, M_{{\odot}},\, \dot{{M}} = {Mdot:g}\, {{\rm g/s}},\, '
              rf'\alpha = {alpha:g}, r = {r:1.3g} \,\rm cm$')
    plt.grid()
    plt.legend()
    os.makedirs('fig/', exist_ok=True)
    plt.savefig('fig/vs_mesa.pdf')
    plt.close()
    print('Plot of structure is successfully saved to fig/vs_mesa.pdf.')
    return


if __name__ == '__main__':
    main()
