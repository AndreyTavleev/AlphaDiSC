#!/usr/bin/env python3
"""
Module contains several classes that represent vertical structure of accretion disc in case
of analytical opacity and ideal gas EOS.

Class IdealKramersVerticalStructure --  for Kramers opacity law and ideal gas EOS.
Class IdealBellLin1994VerticalStructure -- for opacity laws from (Bell & Lin, 1994) and ideal gas EOS.

"""
import os
from collections import namedtuple
from enum import IntEnum

import numpy as np
from astropy import constants as const
from scipy.integrate import solve_ivp, simpson
from scipy.optimize import brentq
from alpha_disc.magn import MagneticField
sigmaSB = const.sigma_sb.cgs.value
R_gas = const.R.cgs.value
G = const.G.cgs.value
M_sun = const.M_sun.cgs.value
c = const.c.cgs.value


class Vars(IntEnum):
    """
    Enumerate that contains names of unknown functions.
    All functions are dimensionless.

    Attributes
    ----------
    S
        Mass coordinate (column density).
    P
        Gas pressure.
    Q
        Flux of energy.
    T
        Temperature.
    """

    S = 0
    P = 1
    Q = 2
    T = 3


class PgasPradNotConvergeError(Exception):
    def __init__(self, P_gas, P_rad, t, z0r, Sigma0_par=None):
        self.P_gas = P_gas
        self.P_rad = P_rad
        self.t = t
        self.z0r = z0r
        self.Sigma0_par = Sigma0_par
        if self.Sigma0_par is None:
            self.message = f'Not converged, P_gas = {self.P_gas:g} < 0 or NaN at t = {self.t:g}, ' \
                           f'P_rad = {self.P_rad:g}. ' \
                           f'Try another z0r estimation. Current estimation is z0r = {self.z0r:g}.'
        else:
            self.message = f'Not converged, P_gas = {self.P_gas:g} < 0 or Nan at t = {self.t:g}, ' \
                           f'P_rad = {self.P_rad:g}. ' \
                           f'Try another Sigma0_par or z0r estimations. ' \
                           f'Current estimations are Sigma0_par = {self.Sigma0_par:g}, z0r = {self.z0r:g}.'

    def __str__(self):
        return self.message


class BaseVerticalStructure:
    """
    Base class for Vertical structure, solver of the system of dimensionless vertical structure ODEs.
    The system contains four linear differential equations for pressure P, mass coordinate S, energy flux Q
    and temperature T as functions of vertical coordinate z. The only unknown free parameter
    is semi-thickness of accretion disc z_0. System is supplemented by four first-type
    boundary conditions (one for each variable). Method `fit` serve to find the free
    parameter z_0 and get solve the system. Integration of system is carried out by `integrate` method.

    Attributes
    ----------
    Mx : double
        Mass of central star in grams.
    alpha : double
        Alpha parameter for alpha-prescription of viscosity.
    r : double
        Distance from central star (radius in cylindrical coordinate system) in cm.
    F : double
        Moment of viscosity forces in g*cm^2/s^2.
    eps : double, optional
        Accuracy of vertical structure calculation. Default 1e-5.
    mu : double, optional
        Molecular weight for ideal gas equation of state. Default 0.6.
    rad_pressure_include : bool
        Whether to consider the radiation pressure contribution to the total
        pressure. Default True.

    Methods
    -------
    fit()
        Solves optimisation problem and calculate the vertical structure.
    integrate()
        Integrates the system and return values of four dimensionless functions.
    Pi_finder()
        Returns the Pi values (see Ketsaris & Shakura, 1998).
    parameters_C()
        Calculates parameters of disc in the symmetry plane.
    tau()
        Calculates optical thickness of the disc.

    """

    def __init__(self, Mx, alpha, r, F, eps=1e-5, mu=0.6, 
                 rad_pressure_include=True):
        self.mu = mu
        self.Mx = Mx
        self.GM = G * Mx
        self.alpha = alpha
        self.r = r
        self.F = F
        if rad_pressure_include:
            self.eta_rad = 1.0
        else:
            self.eta_rad = 0.0
        
        self.omegaK = np.sqrt(self.GM / self.r ** 3)
        self.eps = eps

        self.Q_norm = self.Q0 = (3 / (8 * np.pi)) * F * self.omegaK / self.r ** 2

        self.z0 = self.z0_init()
        self.Teff = (self.Q0 / sigmaSB) ** (1 / 4)
        self.fitted = False

    @property
    def z0(self):
        return self.__z0

    @z0.setter
    def z0(self, z0):
        self.__z0 = z0
        self.P_norm = (4 / 3) * self.Q_norm / (self.alpha * z0 * self.omegaK)
        self.T_norm = self.omegaK ** 2 * self.mu * z0 ** 2 / R_gas
        self.sigma_norm = 28 * self.Q_norm / (3 * self.alpha * z0 ** 2 * self.omegaK ** 3)
        
    def rad_pressure(self, T):
        return 4 * sigmaSB / (3 * c) * T ** 4 * self.eta_rad

    def law_of_viscosity(self, P):
        return self.alpha * P

    def law_of_rho(self, P, T, full_output):
        """If full_output==False, should return rho(P, T).
        
        Else, should return rho(P, T) and namedtuple containing:
            dlnRho_dlnPgas_const_T,
            dlnRho_dlnT_const_Pgas,
            mu,
            lnfree_e,
            grad_ad
            """
        raise NotImplementedError

    def law_of_opacity(self, rho, T, lnfree_e):
        """Should return kappa(rho, T)"""
        raise NotImplementedError

    def viscosity(self, y):
        return self.law_of_viscosity(y[Vars.P] * self.P_norm +
                                self.rad_pressure(y[Vars.T] * self.T_norm))

    def rho(self, y, full_output):
        return self.law_of_rho(P=y[Vars.P] * self.P_norm, T=y[Vars.T] * self.T_norm, full_output=full_output)

    def opacity(self, y, lnfree_e):
        rho = self.rho(y, full_output=False)
        return self.law_of_opacity(rho=rho, T=y[Vars.T] * self.T_norm, lnfree_e=lnfree_e)
    
    def additional_photospheric_pressure(self):
        """Additional pressure [cgs] to be added to the photospheric pressure
        in CGS.
        Zero in absence of the field; in case of field present it should be overwritten by mixins.
        """
        return 0.0
    
    def dP_magn_induced_dz(self, z):
        """Returns d(P magn induced)/dz === d(b_induced_phi**2/8pi)/dz in CGS.    
        Used as an additional right-hand side for the dPtot/dz:
            dPtot/dz = (dPtot/dz)_no_field - dP_magn_induced_dz.
        Zero in absence of the field; in case of field present it should be overwritten by mixins.
        """
        return z*0

    def photospheric_pressure_equation(self, tau, P_total):  # P = P_total
        T = self.Teff * (1 / 2 + 3 * tau / 4) ** (1 / 4)
        P_rad =  self.rad_pressure(T)
        if P_total - P_rad < 0 or np.isnan(P_total - P_rad):
            raise PgasPradNotConvergeError(*(P_total - P_rad), P_rad=P_rad, t=0.0, z0r=self.z0 / self.r)
        rho, eos = self.law_of_rho(P = P_total - P_rad, T=T, full_output=True)
        varkappa = self.law_of_opacity(rho=rho, T=T, lnfree_e=eos.lnfree_e)
        return self.z0 * self.omegaK ** 2 / varkappa

    def P_ph(self):
        # solution is P_total, result is P_gas
        solution = solve_ivp(
            self.photospheric_pressure_equation,
            [0, 2 / 3],
            [1e-7 * self.P_norm + 0.5 * self.rad_pressure(self.Teff)],
            rtol=self.eps, method='RK23'
        )
        P_rad = self.rad_pressure(self.Teff)
        result = solution.y[0][-1] - P_rad  # P_gas = P_tot - P_rad
        result += self.additional_photospheric_pressure()
        if result < 0 or np.isnan(result):
            raise PgasPradNotConvergeError(P_gas=result, P_rad=P_rad, t=0.0, z0r=self.z0 / self.r)
        return result

    def Q_initial(self):
        return 1

    def initial(self):
        """
        Initial conditions.

        Returns
        -------
        array

        """

        Q_initial = self.Q_initial()
        y = np.empty(4, dtype=np.float64)
        y[Vars.S] = 0
        y[Vars.P] = self.P_ph() / self.P_norm
        y[Vars.Q] = Q_initial
        y[Vars.T] = (Q_initial * self.Q_norm / sigmaSB) ** (1 / 4) / self.T_norm
        return y

    def dlnTdlnP(self, y, t):
        """Returns d(lnT)/d(lnPtotal) """
        raise NotImplementedError

    def dQdz(self, y, t):
        w_r_phi = self.viscosity(y)
        return -(3 / 2) * self.z0 * self.omegaK * w_r_phi / self.Q_norm

    def dydt(self, t, y):
        """
        The right side of ODEs system. Magnetic fields modify this system only
        as an additional source in a hydrostatic equation:
            dPtot/dz = -rho wk^2 z - dPmagn/dz,
        where dPmagn/dz is a known function of z.
        Parameters
        ----------
        t : array-like
            Modified vertical coordinate (t = 1 - z/z0, so z = z0(1-t)).
        y :
            Current values of (dimensionless) unknown functions.

        Returns
        -------
        array

        """
        dy = np.empty(4)
        if y[Vars.P] < 0 or np.isnan(y[Vars.P]):
            try:
                raise PgasPradNotConvergeError(P_gas=y[Vars.P] * self.P_norm,
                                               P_rad=self.rad_pressure(y[Vars.T] * self.T_norm),
                                               t=t, z0r=self.z0 / self.r, Sigma0_par=self.Sigma0_par)
            except AttributeError:
                raise PgasPradNotConvergeError(P_gas=y[Vars.P] * self.P_norm,
                                               P_rad=self.rad_pressure(y[Vars.T] * self.T_norm),
                                               t=t, z0r=self.z0 / self.r) from None
        rho, eos = self.rho(y, full_output=True)

        dp_full_dt = rho * (1 - t) * self.omegaK ** 2 * self.z0 ** 2 / self.P_norm # d (P total dimentionless)/dt
        dp_full_dt += self.dP_magn_induced_dz(z=self.z0 * (1 - t)) * self.z0 / self.P_norm
        B = 4 * self.rad_pressure(self.T_norm * y[Vars.T]) / y[Vars.T] / self.P_norm
        P_full = y[Vars.P] * self.P_norm + self.rad_pressure(y[Vars.T] * self.T_norm)
        
        dP_full_dt = dp_full_dt * self.P_norm # dP_full/dt

        grad = self.dlnTdlnP(y, t) # d(ln T) / d(ln P_full) === (P_full/T) * (dT/dP_full)
        dTdz = grad * dP_full_dt * y[Vars.T] / P_full
        dy[Vars.S] = 2 * rho * self.z0 / self.sigma_norm
        dy[Vars.P] = dp_full_dt - B * dTdz
        dy[Vars.Q] = self.dQdz(y, t)
        dy[Vars.T] = dTdz
        return dy

    def integrate(self, t):
        """
        Integrates ODEs and return list that contains array with values of
        four dimensionless functions and a message from the solver.

        Parameters
        ----------
        t : array-like
            Interval of modified vertical coordinate (t = 1 - z/z0) for integration and evaluation.
            t[0] must be equal to zero, t[-1] must be less or equal to unity.

        Returns
        -------
        list
            List containing the array with values of dimensionless functions
            S (column density), P (gas pressure), Q (energy flux), T (temperature)
            calculating at points of `t` array. Also list contains the
            message from the integrator.

        """
        assert t[0] == 0
        assert t[-1] <= 1
        solution = solve_ivp(self.dydt, (t[0], t[-1]), self.initial(), t_eval=t, rtol=self.eps, method='RK23')
        return [solution.y, solution.message]

    def tau(self):
        """
        Calculates total optical thickness of the disc.

        Returns
        -------
        double
            Total optical thickness.

        """
        t = np.linspace(0, 1, 100)
        y = self.integrate(t)[0]
        rho, eos = self.rho(y, full_output=True)
        varkappa = self.opacity(y, lnfree_e=eos.lnfree_e)
        tau_norm = simpson(varkappa * rho, t)
        return self.z0 * tau_norm + 2 / 3

    def y_c(self):
        y = self.integrate([0, 1])
        return y[0][:, -1]

    def parameters_C(self):
        """
        Calculates parameters of disc in the symmetry plane.

        Returns
        -------
        array
            Opacity, bulk density, temperature, gas pressure at the symmetry plane and surface density of disc.

        """
        y_c = self.y_c()
        Sigma0 = y_c[Vars.S] * self.sigma_norm
        T_C = y_c[Vars.T] * self.T_norm
        P_C = y_c[Vars.P] * self.P_norm
        rho_C, eos = self.rho(y_c, full_output=True)
        varkappa_C = self.opacity(y_c, lnfree_e=eos.lnfree_e)
        return np.array([varkappa_C, rho_C, T_C, P_C, Sigma0])

    def tau0(self):
        y = self.parameters_C()
        Sigma0 = y[4]
        varkappa_C = y[0]
        return Sigma0 * varkappa_C / 2

    def Pi_finder(self):
        """
        Calculates the so-called Pi parameters (see Ketsaris & Shakura, 1998).

        Returns
        -------
        array
            Contains the values of Pi.

        """
        varkappa_C, rho_C, T_C, P_C, Sigma0 = self.parameters_C()
        P_C_tot = P_C + self.rad_pressure(T_C)
        Pi_1 = (self.omegaK ** 2 * self.z0 ** 2 * rho_C) / P_C_tot
        Pi_2 = Sigma0 / (2 * self.z0 * rho_C)
        Pi_3 = (3 / 4) * (self.alpha * self.omegaK * P_C_tot * Sigma0) / (
                self.Q0 * rho_C)
        Pi_4 = (3 / 32) * (self.Teff / T_C) ** 4 * (Sigma0 * varkappa_C)

        Pi_real = np.array([Pi_1, Pi_2, Pi_3, Pi_4])

        return Pi_real

    def z0_init(self):
        return (self.r * 2.86e-7 * self.F ** (3 / 20) * (self.Mx / M_sun) ** (-9 / 20)
                * self.alpha ** (-1 / 10) * (self.r / 1e10) ** (1 / 20))

    def fit(self, z0r_estimation=None, verbose=False):
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

        Returns
        -------
        double and scipy.optimize.RootResults
            The value of normalised unknown free parameter z_0 / r and result of optimisation.

        """

        def dq(z0r):
            self.z0 = z0r * self.r
            q_c = self.y_c()[Vars.Q]
            if verbose:
                print(f'z0r = {z0r:g}')
            return q_c

        if z0r_estimation is None:
            z0r = self.z0 / self.r
        else:
            z0r = z0r_estimation
        sign_dq = dq(z0r)
        if sign_dq > 0:
            factor = 2.0
        else:
            factor = 0.5

        while True:
            z0r *= factor
            if sign_dq * dq(z0r) < 0:
                break

        z0r, result = brentq(dq, z0r, z0r / factor, full_output=True)
        self.z0 = z0r * self.r
        self.fitted = True
        return z0r, result


class RadiativeTempGradient:
    """
    Temperature gradient class. Calculates radiative d(lnT)/d(lnPtotal) in the Eddington approximation.

    """
    def dlnTdlnP(self, y, t):
        rho, eos = self.rho(y, full_output=True)
        varkappa = self.opacity(y, lnfree_e=eos.lnfree_e)

        if t == 1:
            dTdz_der = (self.dQdz(y, t) / y[Vars.T] ** 3) * 3 * varkappa * rho * self.z0 * self.Q_norm / (
                    16 * sigmaSB * self.T_norm ** 4)
            P_full = y[Vars.P] * self.P_norm + self.rad_pressure(y[Vars.T]*self.T_norm)
            dP_full_der = - rho * self.omegaK ** 2 * self.z0 ** 2
            dlnTdlnP_rad = (P_full / y[Vars.T]) * (dTdz_der / dP_full_der)
        else:
            dTdz = (abs(y[Vars.Q]) / y[Vars.T] ** 3) * 3 * varkappa * rho * self.z0 * self.Q_norm / (
                    16 * sigmaSB * self.T_norm ** 4)
            P_full = y[Vars.P] * self.P_norm + self.rad_pressure(y[Vars.T]*self.T_norm)
            dP_full = rho * (1 - t) * self.omegaK ** 2 * self.z0 ** 2
            dlnTdlnP_rad = (P_full / y[Vars.T]) * (dTdz / dP_full)
        return dlnTdlnP_rad


class IdealGasMixin:
    def law_of_rho(self, P, T, full_output):
        if not full_output:
            return P * self.mu / (R_gas * T)
        else:
            eos_temp = namedtuple(
                'eos_temp',
                ('dlnRho_dlnPgas_const_T', 'dlnRho_dlnT_const_Pgas',
                 'mu', 'lnfree_e', 'grad_ad',), )
            eos = eos_temp(dlnRho_dlnPgas_const_T=1.0, dlnRho_dlnT_const_Pgas=-1.0, mu=self.mu, lnfree_e=0.0,
                           grad_ad=0.4)
            return P * self.mu / (R_gas * T), eos


class KramersOpacityMixin:
    varkappa0 = 5e24  # Kramers law
    zeta = 1
    gamma = -7 / 2
    varkappa_sc = 0.34  # Thomson scattering

    def law_of_opacity(self, rho, T, lnfree_e):
        varkappa_kram = self.varkappa0 * (rho ** self.zeta) * (T ** self.gamma)
        return varkappa_kram + self.varkappa_sc


class BellLin1994TwoComponentOpacityMixin:
    varkappa0_ff = 1.5e20  # BB AND FF, OPAL
    zeta_ff = 1
    gamma_ff = - 5 / 2
    varkappa0_h = 1.0e-36  # H-scattering
    zeta_h = 1 / 3
    gamma_h = 10
    varkappa_sc = 0.34  # Thomson scattering

    def law_of_opacity(self, rho, T, lnfree_e):
        opacity_h = self.varkappa0_h * (rho ** self.zeta_h) * (T ** self.gamma_h)
        opacity_ff = self.varkappa0_ff * (rho ** self.zeta_ff) * (T ** self.gamma_ff)
        return np.where(opacity_h < opacity_ff, opacity_h, opacity_ff + self.varkappa_sc)


class IdealKramersVerticalStructure(IdealGasMixin, KramersOpacityMixin, RadiativeTempGradient, BaseVerticalStructure):
    """
    Vertical structure class for Kramers opacity law and ideal gas EOS.

    """
    pass


class IdealBellLin1994VerticalStructure(IdealGasMixin, BellLin1994TwoComponentOpacityMixin, RadiativeTempGradient,
                                        BaseVerticalStructure):
    """
    Vertical structure class for opacity laws from (Bell & Lin, 1994) and ideal gas EOS.

    """
    pass

class IdealKramersVerticalStructureMagnetic(MagneticField, IdealGasMixin, KramersOpacityMixin, 
                                            RadiativeTempGradient, BaseVerticalStructure):
    """
    Vertical structure class for Kramers opacity law and ideal gas EOS.
    Includes the magnetic field.

    """
    pass


class IdealBellLin1994VerticalStructureMagnetic(MagneticField, IdealGasMixin,
                                    BellLin1994TwoComponentOpacityMixin, RadiativeTempGradient,
                                        BaseVerticalStructure):
    """
    Vertical structure class for opacity laws from (Bell & Lin, 1994) and ideal gas EOS.
    Includes the magnetic field.

    """
    pass




def main():
    M = 10 * M_sun
    alpha = 0.01
    Mdot = 1e18
    rg = 2 * G * M / c ** 2
    r = 400 * rg
    print('Finding Pi parameters of structure and making a structure plot. '
          '\nStructure with opacity laws from (Bell & Lin, 1994) and ideal gas EOS.')
    print(f'M = {M:g} grams = {M / M_sun:g} M_sun \nr = {r:g} cm = {r / rg:g} rg '
          f'\nalpha = {alpha:g} \nMdot = {Mdot:g} g/s')
    h = np.sqrt(G * M * r)
    r_in = 3 * rg
    F = Mdot * h * (1 - np.sqrt(r_in / r))
    vs = IdealBellLin1994VerticalStructure(M, alpha, r, F)
    z0r, result = vs.fit()
    if result.converged:
        print('The vertical structure has been calculated successfully.')
    Pi = vs.Pi_finder()
    print('Pi parameters =', Pi)
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
    plt.savefig('fig/vs.pdf')
    plt.close()
    print('Plot of structure is successfully saved to fig/vs.pdf.')
    return


if __name__ == '__main__':
    main()
