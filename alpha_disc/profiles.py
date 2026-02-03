#!/usr/bin/env python3
"""
Module contains functions that calculate vertical and radial structure and S-curve.
Functions return tables with calculated data of structure or S-curve.

StructureChoice -- Initialise the chosen vertical structure.
    It serves as interface for initialising the structure object in a simpler way.
Vertical_Profile -- calculates vertical structure and makes table with disc parameters as functions of vertical
    coordinate. Table also contains input parameters of structure, parameters in the symmetry plane and
    parameter normalisations.
S_curve -- Calculates S-curve and makes table with disc parameters on the S-curve.
    Table contains input parameters of system, surface density Sigma0, viscous torque F,
    accretion rate Mdot, effective temperature Teff (or Tvis in case of external irradiation),
    geometrical half-thickness of the disc z0r, parameters in the symmetry plane of disc on the S-curve.
Radial_Profile -- Calculates radial structure of disc. Return table, which contains input parameters of the system,
    surface density Sigma0, viscous torque F, accretion rate Mdot, effective temperature Teff
    (or Tvis in case of external irradiation), geometrical half-thickness of the disc z0r and
    parameters in the symmetry plane of disc as functions of radius.

"""
import numpy as np
from astropy import constants as const
from scipy.integrate import simpson
from scipy.interpolate import InterpolatedUnivariateSpline
from collections import namedtuple
from alpha_disc.magn import ViscousTorque
import alpha_disc.vs as vert

try:
    import alpha_disc.mesa_vs as mesa_vs
except ImportError:
    mesa_vs = np.nan

sigmaSB = const.sigma_sb.cgs.value
G = const.G.cgs.value
M_sun = const.M_sun.cgs.value
c = const.c.cgs.value


def StructureChoice(M, alpha, r, Par, input, structure, mu=0.6, abundance='solar', F_in=0, nu_irr=None, L_X_irr=None,
                    spectrum_irr=None, spectrum_irr_par=None, args_spectrum_irr=(), kwargs_spectrum_irr={},
                    C_irr=None, T_irr=None, cos_theta_irr=None, cos_theta_irr_exp=1 / 12,
                    magn_args=None, rad_pressure_include=True):
    """
    Initialise the chosen vertical structure object.

    Parameters
    ----------
    M : double
        Mass of central object in grams.
    alpha : double
        Alpha parameter for alpha-prescription of viscosity.
    r : double
        Distance from central star (radius in cylindrical coordinate system) in cm.
    Par : double
        Can be viscous torque in g*cm^2/s^2, effective temperature in K,
        accretion rate in g/s, in eddington limits or in Msun/yr.
        Choice depends on 'input' parameter.
    input : str
        Define the choice of 'Par' parameter. Can be 'F' (viscous torque),
        'Teff' (effective temperature, or viscous temperature in case of irradiation),
        'Mdot' (accretion rate), 'Mdot_Mdot_edd' (Mdot in eddington limits) or 'Mdot_Msun_yr' (Mdot in Msun/yr).
    structure : str
        Type of vertical structure. Possible options are:
        'Kramers', 'BellLin' -- ideal gas EoS, analytical opacities and radiative temperature gradient;
        'Mesa', 'MesaAd', 'MesaRadAd', 'MesaRadConv' -- MESA EoS and opacities and
                                                        radiative, adiabatic, rad+ad
                                                        and rad+conv temperature gradient;
        'MesaIdealGas' -- ideal gas EoS, MESA opacities and radiative temperature gradient;
        'MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero' -- simple external irradiation scheme
                                                                   via T_irr or C_irr
        'MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr' -- advanced external irradiation scheme
                                                       from (Mescheryakov et al. 2011).
    mu : double
        Mean molecular weight. Use in case of ideal gas EoS.
    abundance : dict or str
        Chemical composition of disc. Use in case of MESA EoS.
        Format: {'isotope_name': abundance}. For example: {'h1': 0.7, 'he4': 0.3}.
        Use 'solar' str in case of solar composition.
    F_in : double
        Viscous torque in g*cm^2/s^2 at the inner disc radius.
        The viscous torque F = Mdot * h * (1-sqrt(r_in/r)) + F_in.
        Default F_in=0 (when the central source is Schwarzschild black hole).
    nu_irr : array-like
        If structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr'], nu_irr is the irradiation spectral flux argument.
        Can be (X-ray) frequency (in Hz) or energy (in keV) array for spectral external irradiation flux.
        Choose depends on the 'spectrum_irr_par'.
    spectrum_irr : array-like or callable
        If structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr'],
        spectrum_irr is the spectrum of external irradiation flux, i.e.
        the spectral (X-ray) external irradiation flux F_nu_irr = F_irr * spectrum_irr.
        If spectrum_irr is array-like it must be in 1/Hz or in 1/keV depending on 'spectrum_irr_par',
        must be normalised to unity, and its size must be equal to nu_irr.size.
        If spectrum_irr is callable, then
        ``F_nu_irr = F_irr * spectrum_irr(nu_irr, *args_spectrum_irr, **kwargs_spectrum_irr)``.
        The normalisation of the spectrum_irr in that case is performed automatically.
    spectrum_irr_par : str
        Defines the irradiation spectral flux argument.
        Can be 'nu' (frequency in Hz) and 'E_in_keV' (energy in keV).
    args_spectrum_irr, kwargs_spectrum_irr : tuple and dict
        Extra arguments and keyword arguments of spectrum_irr, if it's callable.
        The calling signature is ``spectrum_irr(nu_irr, *args_spectrum_irr, **kwargs_spectrum_irr)``
    L_X_irr : double or None
        If structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr'],
        L_X_irr is the (X-ray) bolometric luminosity of external irradiation source.
        If None, then ``L_X_irr = 0.1 * Mdot * c ** 2``.
        The irradiation flux ``F_irr = L_X_irr / (4 * pi * r ** 2)``.
    cos_theta_irr : double or None
        If structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr'], cos_theta_irr is the cosine of angle
        of incidence for external irradiation flux. If None, ``cos_theta_irr = cos_theta_irr_exp * (z0/r)``.
    cos_theta_irr_exp : double
        If cos_theta_irr is None, ``cos_theta_irr = cos_theta_irr_exp * (z0/r)``.
    C_irr : double
        If structure in ['MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero'],
        C_irr is the irradiation parameter.
    T_irr : double
        If structure in ['MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero'],
        T_irr is the irradiation temperature.
    magn_args : dict
        If the magnetic field treatment is to be included, this dicionary 
        should comtain the parameters necessary for the calculation of the 
        magneic torque, inner radius, etc. Only possible for models without
        irratiation. Default None.

    Returns
    -------
    vs : vertical structure
        Chosen NON-FITTED vertical structure.
    F : double
        Viscous torque in g*cm^2/s^2.
    Teff : double
        Effective temperature (viscous temperature in case of irradiation) in Kelvins.
    Mdot : double
        Accretion rate in g/s.

    """
    Torques = ViscousTorque(Mx=M, F_in=F_in, magn_args=magn_args)
    rg = Torques.rg
    # r_in = Torques.r0
    if r <= 3*rg:
        raise Exception(f'Radius r should be greater than R_ISCO = 3*rg. '
                        f'Actual radius r = {r / rg:g} rg.')
    if input == 'Teff':
        Teff = Par
        Mdot = Torques.Mdot_from_Teff(Teff=Teff, r=r)
        if Mdot < 0:
            raise Exception(f'Mdot = {Mdot:g} g/s < 0, incorrect F_in = {F_in:g} g*cm^2/s^2.')  
        F = Torques.F_vis(Mdot, r)
    elif input in ('Mdot', 'Mdot_Mdot_edd', 'Mdot_Msun_yr'):
        if input == 'Mdot': Mdot = Par
        if input == 'Mdot_Mdot_edd': Mdot = Par * 1.39e18 * M / M_sun
        if input == 'Mdot_Msun_yr': Mdot = Par * M_sun / 31557600.0
        F = Torques.F_vis(Mdot, r)
        Teff = Torques.T_eff(Mdot, r)
    elif input == 'F':
        F = Par
        Mdot = Torques.Mdot_from_F(F=F, r=r)
        if Mdot < 0:
            raise Exception(f'Mdot = {Mdot:g} g/s < 0, incorrect F_in = {F_in:g} g*cm^2/s^2.')
        Teff = Torques.T_eff(Mdot, r)
    else:
        raise Exception("Incorrect input, try 'Teff', 'F', 'Mdot', 'Mdot_Mdot_edd' or 'Mdot_Msun_yr'.")
        
    if magn_args is not None and structure in ('MesaIrrZero', 'MesaRadAdIrrZero',
                                              'MesaRadConvIrrZero', 'MesaIrr',
                                              'MesaRadAdIrr', 'MesaRadConvIrr'):
        print("""A structure with irradiation was chosen, but the magnetic field is not supported for this case.
              \nMagnetic field is set to zero.""")
    r_in = Torques.r0(Mdot)
    if r <= r_in:
        raise Exception(f'Radius r should be greater than the inner radius = {r_in:g} cm. '
                        f'Actual radius r = {r / r_in:g} r_in.')

    if structure == 'Kramers':
        if magn_args is None:
            vs = vert.IdealKramersVerticalStructure(M, alpha, r, F, mu=mu,
                                    rad_pressure_include=rad_pressure_include)
        else:
            vs = vert.IdealKramersVerticalStructureMagnetic(M, alpha, r, F, mu=mu,
                                                            magn_args=magn_args,
                                                            rad_pressure_include=rad_pressure_include)
    elif structure == 'BellLin':
        if magn_args is None:
            vs = vert.IdealBellLin1994VerticalStructure(M, alpha, r, F, mu=mu,
                                                        rad_pressure_include=rad_pressure_include)
        else:
            vs= vert.IdealBellLin1994VerticalStructureMagnetic(M, alpha, r, F, mu=mu,
                                                            magn_args=magn_args,
                                                            rad_pressure_include=rad_pressure_include)
    elif structure == 'Mesa':
        try:
            if np.isnan(mesa_vs):
                raise ModuleNotFoundError('Mesa2py is not installed')
        except TypeError:
            if magn_args is None:
                vs = mesa_vs.MesaVerticalStructure(M, alpha, r, F, abundance=abundance,
                                                   rad_pressure_include=rad_pressure_include)
            else:
                vs = mesa_vs.MesaVerticalStructureMagnetic(M, alpha, r, F, abundance=abundance,
                                                           magn_args=magn_args,
                                                           rad_pressure_include=rad_pressure_include)
    elif structure == 'MesaIdealGas':
        try:
            if np.isnan(mesa_vs):
                raise ModuleNotFoundError('Mesa2py is not installed')
        except TypeError:
            if magn_args is None:
                vs = mesa_vs.MesaIdealGasVerticalStructure(M, alpha, r, F, mu=mu, abundance=abundance,
                                                           rad_pressure_include=rad_pressure_include)
            else:
                vs = mesa_vs.MesaIdealGasVerticalStructureMagnetic(M, alpha, r, F, mu=mu, abundance=abundance,
                                                                   magn_args = magn_args,
                                                                   rad_pressure_include=rad_pressure_include)
    elif structure == 'MesaAd':
        try:
            if np.isnan(mesa_vs):
                raise ModuleNotFoundError('Mesa2py is not installed')
        except TypeError:
            if magn_args is None:
                vs = mesa_vs.MesaVerticalStructureAd(M, alpha, r, F, abundance=abundance,
                                                     rad_pressure_include=rad_pressure_include)
            else:
                vs = mesa_vs.MesaVerticalStructureAdMagnetic(M, alpha, r, F, abundance=abundance,
                                                             magn_args=magn_args,
                                                             rad_pressure_include=rad_pressure_include)
    elif structure == 'MesaRadAd':
        try:
            if np.isnan(mesa_vs):
                raise ModuleNotFoundError('Mesa2py is not installed')
        except TypeError:
            if magn_args is None:
                vs = mesa_vs.MesaVerticalStructureRadAd(M, alpha, r, F, abundance=abundance,
                                                        rad_pressure_include=rad_pressure_include)
            else:
                vs = mesa_vs.MesaVerticalStructureRadAdMagnetic(M, alpha, r, F, abundance=abundance,
                                                                magn_args=magn_args,
                                                                rad_pressure_include=rad_pressure_include)
    elif structure == 'MesaRadConv':
        try:
            if np.isnan(mesa_vs):
                raise ModuleNotFoundError('Mesa2py is not installed')
        except TypeError:
            if magn_args is None:
                vs = mesa_vs.MesaVerticalStructureRadConv(M, alpha, r, F, abundance=abundance,
                                                          rad_pressure_include=rad_pressure_include)
            else:
                vs = mesa_vs.MesaVerticalStructureRadConvMagnetic(M, alpha, r, F, abundance=abundance,
                                                          magn_args=magn_args,
                                                          rad_pressure_include=rad_pressure_include)
    elif structure == 'MesaIrrZero':
        try:
            if np.isnan(mesa_vs):
                raise ModuleNotFoundError('Mesa2py is not installed')
        except TypeError:
            vs = mesa_vs.MesaVerticalStructureExternalIrradiationZeroAssumption(M, alpha, r, F, C_irr=C_irr,
                                                                                T_irr=T_irr, abundance=abundance,
                                                                                F_in=F_in)
    elif structure == 'MesaRadAdIrrZero':
        try:
            if np.isnan(mesa_vs):
                raise ModuleNotFoundError('Mesa2py is not installed')
        except TypeError:
            vs = mesa_vs.MesaVerticalStructureRadAdExternalIrradiationZeroAssumption(M, alpha, r, F, C_irr=C_irr,
                                                                                     T_irr=T_irr, abundance=abundance,
                                                                                     F_in=F_in)
    elif structure == 'MesaRadConvIrrZero':
        try:
            if np.isnan(mesa_vs):
                raise ModuleNotFoundError('Mesa2py is not installed')
        except TypeError:
            vs = mesa_vs.MesaVerticalStructureRadConvExternalIrradiationZeroAssumption(M, alpha, r, F, C_irr=C_irr,
                                                                                       T_irr=T_irr, abundance=abundance,
                                                                                       F_in=F_in)
    elif structure == 'MesaIrr':
        try:
            if np.isnan(mesa_vs):
                raise ModuleNotFoundError('Mesa2py is not installed')
        except TypeError:
            if L_X_irr is None:
                eta = 0.1
                L_X_irr = eta * Mdot * c ** 2
            vs = mesa_vs.MesaVerticalStructureExternalIrradiation(M, alpha, r, F, nu_irr=nu_irr,
                                                                  L_X_irr=L_X_irr, spectrum_irr=spectrum_irr,
                                                                  spectrum_irr_par=spectrum_irr_par,
                                                                  args_spectrum_irr=args_spectrum_irr,
                                                                  kwargs_spectrum_irr=kwargs_spectrum_irr,
                                                                  cos_theta_irr=cos_theta_irr,
                                                                  cos_theta_irr_exp=cos_theta_irr_exp,
                                                                  abundance=abundance)
    elif structure == 'MesaRadAdIrr':
        try:
            if np.isnan(mesa_vs):
                raise ModuleNotFoundError('Mesa2py is not installed')
        except TypeError:
            if L_X_irr is None:
                eta = 0.1
                L_X_irr = eta * Mdot * c ** 2
            vs = mesa_vs.MesaVerticalStructureRadAdExternalIrradiation(M, alpha, r, F, nu_irr=nu_irr,
                                                                       L_X_irr=L_X_irr, spectrum_irr=spectrum_irr,
                                                                       spectrum_irr_par=spectrum_irr_par,
                                                                       args_spectrum_irr=args_spectrum_irr,
                                                                       kwargs_spectrum_irr=kwargs_spectrum_irr,
                                                                       cos_theta_irr=cos_theta_irr,
                                                                       cos_theta_irr_exp=cos_theta_irr_exp,
                                                                       abundance=abundance)
    elif structure == 'MesaRadConvIrr':
        try:
            if np.isnan(mesa_vs):
                raise ModuleNotFoundError('Mesa2py is not installed')
        except TypeError:
            if L_X_irr is None:
                eta = 0.1
                L_X_irr = eta * Mdot * c ** 2
            vs = mesa_vs.MesaVerticalStructureRadConvExternalIrradiation(M, alpha, r, F, nu_irr=nu_irr,
                                                                         L_X_irr=L_X_irr, spectrum_irr=spectrum_irr,
                                                                         spectrum_irr_par=spectrum_irr_par,
                                                                         args_spectrum_irr=args_spectrum_irr,
                                                                         kwargs_spectrum_irr=kwargs_spectrum_irr,
                                                                         cos_theta_irr=cos_theta_irr,
                                                                         cos_theta_irr_exp=cos_theta_irr_exp,
                                                                         abundance=abundance)
    else:
        raise Exception("Incorrect structure. Possible options are: 'Kramers', 'BellLin',\n"
                        "'Mesa', 'MesaAd', 'MesaRadAd', 'MesaRadConv', 'MesaIdealGas',\n"
                        "'MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero',\n"
                        "'MesaIrr', 'MesaRadAdIrr' or 'MesaRadConvIrr'")

    return vs, F, Teff, Mdot


def Convective_parameter(vs):
    """
    Calculates convective parameter of structure. This parameter shows what part of disc is convective.

    Parameters
    ----------
    vs : vertical structure
        FITTED vertical structure, for which convective parameter should be calculated.

    Returns
    -------
    conv_param_z : double
        z_convective / z0, z-fraction of convective region, from 0 to 1.
    conv_param_sigma : double
        sigma_convective / sigma0, mass fraction of convective region, from 0 to 1.

    """
    n = 1000
    t = np.linspace(0, 1, n)
    y = vs.integrate(t)[0]
    S, P, Q, T = y
    P_full = P * vs.P_norm + 4 * sigmaSB / (3 * c) * T ** 4 * vs.T_norm ** 4
    grad_plot = InterpolatedUnivariateSpline(np.log(P_full), np.log(T)).derivative()
    rho, eos = vs.law_of_rho(P * vs.P_norm, T * vs.T_norm, True)
    try:
        _ = eos.c_p
    except AttributeError:
        raise Exception('Incorrect vertical structure for convective parameter calculation. '
                        'Use vertical structure with MESA EoS.') from None
    conv_param_sigma = simpson(2 * rho * (grad_plot(np.log(P_full)) > eos.grad_ad), t * vs.z0) / (
            S[-1] * vs.sigma_norm)
    conv_param_z = simpson(grad_plot(np.log(P_full)) > eos.grad_ad, t * vs.z0) / vs.z0
    return conv_param_z, conv_param_sigma


def Vertical_Profile(M, alpha, r, Par, input, structure, mu=0.6, abundance='solar', F_in=0, nu_irr=None, #!!!
                     L_X_irr=None, spectrum_irr=None, spectrum_irr_par=None,
                     args_spectrum_irr=(), kwargs_spectrum_irr={},
                     cos_theta_irr=None, cos_theta_irr_exp=1 / 12, C_irr=None, T_irr=None,
                     z0r_estimation=None, Sigma0_estimation=None, P_ph_0=None, verbose=False,
                     n=100, add_Pi_values=True, path_dots=None,
                     to_return=False, to_save=True, magn_args=None,
                     rad_pressure_include=True):
    """
    Calculates vertical structure and makes table with disc parameters as functions of vertical coordinate.
    Table also contains input parameters of structure, parameters in the symmetry plane and parameter normalisations.

    Parameters
    ----------
    M : double
        Mass of central object in grams.
    alpha : double
        Alpha parameter for alpha-prescription of viscosity.
    r : double
        Distance from central star (radius in cylindrical coordinate system) in cm.
    Par : double
        Can be viscous torque in g*cm^2/s^2, effective temperature in K,
        accretion rate in g/s, in eddington limits or in Msun/yr.
        Choice depends on 'input' parameter.
    input : str
        Define the choice of 'Par' parameter. Can be 'F' (viscous torque),
        'Teff' (effective temperature, or viscous temperature in case of irradiation),
        'Mdot' (accretion rate), 'Mdot_Mdot_edd' (Mdot in eddington limits) or 'Mdot_Msun_yr' (Mdot in Msun/yr).
    mu : double
        Mean molecular weight. Use in case of ideal gas EoS.
    structure : str
        Type of vertical structure. Possible options are:
        'Kramers', 'BellLin' -- ideal gas EoS, analytical opacities and radiative temperature gradient;
        'Mesa', 'MesaAd', 'MesaRadAd', 'MesaRadConv' -- MESA EoS and opacities and
                                                        radiative, adiabatic, rad+ad
                                                        and rad+conv temperature gradient;
        'MesaIdealGas' -- ideal gas EoS, MESA opacities and radiative temperature gradient;
        'MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero' -- simple external irradiation scheme
                                                                   via T_irr or C_irr
        'MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr' -- advanced external irradiation scheme
                                                       from (Mescheryakov et al. 2011).
    abundance : dict or str
        Chemical composition of disc. Use in case of MESA EoS.
        Format: {'isotope_name': abundance}. For example: {'h1': 0.7, 'he4': 0.3}.
        Use 'solar' str in case of solar composition.
    F_in : double
        Viscous torque in g*cm^2/s^2 at the inner disc radius.
        The viscous torque F = Mdot * h * (1-sqrt(r_in/r)) + F_in.
        Default F_in=0 (when the central source is Schwarzschild black hole).
    nu_irr : array-like
        If structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr'], nu_irr is the irradiation spectral flux argument.
        Can be (X-ray) frequency (in Hz) or energy (in keV) array for spectral external irradiation flux.
        Choose depends on the 'spectrum_irr_par'.
    spectrum_irr : array-like or callable
        If structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr'],
        spectrum_irr is the spectrum of external irradiation flux, i.e.
        the spectral (X-ray) external irradiation flux F_nu_irr = F_irr * spectrum_irr.
        If spectrum_irr is array-like it must be in 1/Hz or in 1/keV depending on 'spectrum_irr_par',
        must be normalised to unity, and its size must be equal to nu_irr.size.
        If spectrum_irr is callable, then
        ``F_nu_irr = F_irr * spectrum_irr(nu_irr, *args_spectrum_irr, **kwargs_spectrum_irr)``.
        The normalisation of the spectrum_irr in that case is performed automatically.
    spectrum_irr_par : str
        Defines the irradiation spectral flux argument.
        Can be 'nu' (frequency in Hz) and 'E_in_keV' (energy in keV).
    args_spectrum_irr, kwargs_spectrum_irr : tuple and dict
        Extra arguments and keyword arguments of spectrum_irr, if it's callable.
        The calling signature is ``spectrum_irr(nu_irr, *args_spectrum_irr, **kwargs_spectrum_irr)``
    L_X_irr : double or None
        If structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr'],
        L_X_irr is the (X-ray) bolometric luminosity of external irradiation source.
        If None, then ``L_X_irr = 0.1 * Mdot * c ** 2``.
        The irradiation flux ``F_irr = L_X_irr / (4 * pi * r ** 2)``.
    cos_theta_irr : double or None
        If structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr'], cos_theta_irr is the cosine of angle
        of incidence for external irradiation flux. If None, ``cos_theta_irr = cos_theta_irr_exp * (z0/r)``.
    cos_theta_irr_exp : double
        If cos_theta_irr is None, ``cos_theta_irr = cos_theta_irr_exp * (z0/r)``.
    C_irr : double
        If structure in ['MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero'],
        C_irr is the irradiation parameter.
    T_irr : double
        If structure in ['MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero'],
        T_irr is the irradiation temperature.
    n : int
        Number of dots to calculate.
    z0r_estimation : double
        Start estimation of z0r free parameter to fit the structure. Default is None,
        the estimation is calculated automatically.
    Sigma0_estimation : double
        If structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr'],
        it's the start estimation of Sigma0 free parameter to fit the structure.
        Default is None, the estimation is calculated automatically.
    P_ph_0 : double
        If structure contains Irradiation (either irradiation scheme),
        it's the start estimation for pressure at the photosphere (pressure boundary condition).
        Default is None, the estimation is calculated automatically.
    verbose : bool
        Whether to print values of free parameters at each iteration during fitting.
        Default is False, the fitting process performs silently.
    add_Pi_values : bool
        Whether to write Pi-parameters (see Ketsaris & Shakura, 1998) to the output file header.
    path_dots : str
        Where to save data table.
    to_return : bool
        Whether to return a namedtuple containing the calculated quantities and
        debug data. Default is False.
    to_save : bool
        Whether to save the calculated data in a file. Default is True.
    magn_args : dict or None, optional
        The dictionary containing the parameters necessary for the magentic field
        tretment. If None, the behaviour switches to a non-magnetic case.
        Default is None.
    rad_pressure_include : bool, optional
        Whether to include the radiation pressure contribution in the equation
        of state. Default is True.

    Returns
    -------
    Table with calculated Vertical disc profile will save to path_dots. Table contains:
        1) input parameters of the system -- M in Msun, alpha, r in cm and in rg, effective temperature Teff
           (viscous temperature Tvis in case of irradiation), accretion rate Mdot, viscous torque F,
           structure type, mu (in case of analytical EoS) or abundance (in case of tabular EoS);
        2) varkappa_c, rho_c, T_c, P_c, Sigma0, PradPgas_c, z0r, tau -- opacity, bulk density, temperature,
           gas pressure, surface density, Prad/Pgas, half-thickness and full Rosseland optical depth of the disc;
        3) Sigma_norm, P_norm, T_norm, Q_norm -- normalisations of column density,
           gas pressure, temperature and flux coordinates;
    In case of convection:
        5) conv_param_z, conv_param_sigma -- z-fraction and mass-fraction of convective region.
           Changes from 0 (fully radiative) to 1 (fully convective);
    If add_Pi_values:
        6) Pi1, Pi2, Pi3, Pi4 -- Pi-parameters (see Ketsaris & Shakura, 1998).
    If structure contains Irradiation (either irradiation scheme):
        7) T_irr, C_irr, QirrQvis -- irradiation temperature, irradiation parameter and Qirr/Qvis.
    If structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr']:
        8) cost, Sigma_ph -- cost function and column density of the layers above the photosphere.
           If structure is fitted successfully, cost must be less than 1e-16.

    Table contains as functions of t=1-z/z0:
        1) S, P, Q, T -- normalised column density, gas pressure, flux and temperature;
        2) rho, varkappa, tau -- bulk density, opacity and Rosseland optical depth;
        3) grad -- actual temperature gradient dlnT/dlnP.
    In case of tabular EoS:
        5) grad_ad, free_e -- adiabatic temperature gradient dlnT/dlnP and
           mean number of free electrons per nucleon.

    """
    if path_dots is None and to_save:
        raise Exception("ATTENTION: the data wil not be saved, since 'path_dots' is None.")
    vs, F, Teff, Mdot = StructureChoice(M, alpha, r, Par, input, structure, mu, abundance,
                                        nu_irr=nu_irr, L_X_irr=L_X_irr,
                                        spectrum_irr=spectrum_irr, spectrum_irr_par=spectrum_irr_par,
                                        args_spectrum_irr=args_spectrum_irr, kwargs_spectrum_irr=kwargs_spectrum_irr,
                                        cos_theta_irr=cos_theta_irr, cos_theta_irr_exp=cos_theta_irr_exp,
                                        C_irr=C_irr, T_irr=T_irr, F_in=F_in, magn_args=magn_args,
                                        rad_pressure_include=rad_pressure_include)
    if structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr']:
        kwargs_fit = {'z0r_estimation': z0r_estimation, 'Sigma0_estimation': Sigma0_estimation,
                      'verbose': verbose, 'P_ph_0': P_ph_0}
    elif structure in ['MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero']:
        kwargs_fit = {'z0r_estimation': z0r_estimation, 'verbose': verbose, 'P_ph_0': P_ph_0}
    else:
        kwargs_fit = {'z0r_estimation': z0r_estimation, 'verbose': verbose}
    result = vs.fit(**kwargs_fit)
    try:
        z0r, sigma_par = result.x
    except AttributeError:
        z0r, result = result

    rg = 2 * G * M / c ** 2
    t = np.linspace(0, 1, n)
    S, P, Q, T = vs.integrate(t)[0]
    varkappa_C, rho_C, T_C, P_C, Sigma0 = vs.parameters_C()
    tau = vs.tau()
    delta = (4 * sigmaSB) / (3 * c) * T_C ** 4 / P_C
    P_full = P * vs.P_norm + 4 * sigmaSB / (3 * c) * T ** 4 * vs.T_norm ** 4
    grad_plot = InterpolatedUnivariateSpline(np.log(P_full), np.log(T)).derivative()
    rho, eos = vs.law_of_rho(P * vs.P_norm, T * vs.T_norm, True)
    varkappa = vs.law_of_opacity(rho, T * vs.T_norm, lnfree_e=eos.lnfree_e)
    tau_arr = np.array([simpson(rho[:i] * varkappa[:i], t[:i] * z0r * r) for i in range(2, n + 1)]) + 2 / 3
    tau_arr = np.r_[2 / 3, tau_arr]
    dots_arr = np.c_[t, S, P, abs(Q), T, rho, varkappa, tau_arr, grad_plot(np.log(P_full))]
    header_input_irr = ''
    header_Pi = ''
    if structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr',
                     'MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero']:
        Teff_string = 'Tvis'
    else:
        Teff_string = 'Teff'
    try:
        _ = eos.c_p
        dots_arr = np.c_[dots_arr, eos.grad_ad, np.exp(eos.lnfree_e)]
        header = 't\t S\t P\t Q\t T\t rho\t varkappa\t tau\t grad\t grad_ad\t free_e'
        conv_param_z, conv_param_sigma = Convective_parameter(vs)
        header_conv = f'\nconv_param_z = {conv_param_z} \tconv_param_sigma = {conv_param_sigma}'
    except AttributeError:
        header = 't\t S\t P\t Q\t T\t rho\t varkappa\t tau\t grad'
        header_conv = ''
    header_input = '\nS, P, Q, T -- normalised values, rho -- in g/cm^3, ' \
                   'varkappa -- in cm^2/g \nt = 1 - z/z0 ' \
                   f'\nM = {M / M_sun:e} Msun, alpha = {alpha}, r = {r:e} cm, r = {r / rg} rg, ' \
                   f'{Teff_string} = {Teff} K, Mdot = {Mdot:e} g/s, F = {F:e} g*cm^2/s^2, structure = {structure}'
    if structure in ['Kramers', 'BellLin', 'MesaIdealGas']:
        header_input += f', mu = {mu}'
    else:
        header_input += f', abundance = {abundance}'
    if structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr',
                     'MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero']:
        header_input_irr = f'\nT_irr = {vs.T_irr:g} K, C_irr = {vs.C_irr:g}, QirrQvis = {vs.Q_irr / vs.Q0:g}'
        if structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr']:
            header_input_irr += f', cost = {result.cost:g}, Sigma_ph = {vs.Sigma_ph:g} g/cm^2'
    header_C = f'\nvarkappa_c = {varkappa_C:e} cm^2/g, rho_c = {rho_C:e} g/cm^3, T_c = {T_C:e} K, P_c = {P_C:e} dyn, ' \
               f'Sigma0 = {Sigma0:e} g/cm^2, PradPgas_c = {delta:e}, z0r = {z0r:e}, tau = {tau:e}'
    header_norm = f'\nSigma_norm = {vs.sigma_norm:e}, P_norm = {vs.P_norm:e}, ' \
                  f'T_norm = {vs.T_norm:e}, Q_norm = {vs.Q_norm:e}'
    if add_Pi_values:
        Pi1, Pi2, Pi3, Pi4 = vs.Pi_finder()
        header_Pi = '\nPi1 = {:f}, Pi2 = {:f}, Pi3 = {:f}, Pi4 = {:f}'.format(*vs.Pi_finder())
    header = header + header_input + header_C + header_norm + header_conv + header_Pi + header_input_irr
    if path_dots is not None:
        np.savetxt(path_dots, dots_arr, header=header)
    if to_return:
        VertStructureResults = namedtuple("VertStructureResults", [
                    "t", "S", "P", 'T', "Q", "rho", "varkappa", "tau", "grad",
                    "grad_ad", "free_e",
                    "r", "r_rg", "Sigma0", "Mdot", "Teff", "F", "z0r",
                    "rho_C", "T_C", "P_C", "structure", 'mu',
                     "PradPgas_C", "varkappa_C",
                    "tau_C",
                    "Pi1", "Pi2", "Pi3", "Pi4",
                ])
        _res = VertStructureResults(t=t, S=S, P=P, T=T, Q=Q, rho=rho,
                varkappa=varkappa, tau=tau_arr, grad=grad_plot(np.log(P_full)),
                r=r, r_rg=r/rg, grad_ad=None, free_e=None, Sigma0=Sigma0,
                    Mdot=Mdot, Teff=Teff, F=F, z0r=z0r, rho_C=rho_C, T_C=T_C,
                    P_C=P_C, structure=structure, mu=mu, PradPgas_C=delta,
                    varkappa_C=varkappa_C, tau_C=tau, 
                    Pi1=Pi1, Pi2=Pi2, Pi3=Pi3, Pi4=Pi4)
        return _res
    return


def S_curve(Par_min, Par_max, M, alpha, r, input, structure, mu=0.6, abundance='solar', F_in=0, nu_irr=None,#!!!
            L_X_irr=None, spectrum_irr=None, spectrum_irr_par=None, args_spectrum_irr=(), kwargs_spectrum_irr={},
            cos_theta_irr=None, cos_theta_irr_exp=1 / 12, C_irr=None, T_irr=None,
            z0r_start_estimation=None, Sigma0_start_estimation=None, P_ph_0=None, verbose=False,
            verbose_scurve=True,
            n=100, tau_break=True, add_Pi_values=True, path_dots=None,
            magn_args=None, rad_pressure_include=True, to_save=True, to_return=False):
    """
    Calculates S-curve and makes table with disc parameters on the S-curve.
    Table contains input parameters of system, surface density Sigma0, viscous torque F,
    accretion rate Mdot, effective temperature Teff (or Tvis in case of external irradiation),
    geometrical half-thickness of the disc z0r, parameters in the symmetry plane of disc on the S-curve.

    Parameters
    ----------
    Par_min : double
        The starting value of Par. Par_min and Par_max can be viscous torque in g*cm^2/s^2, effective temperature in K,
        accretion rate in g/s, in eddington limits or in Msun/yr. Choice depends on 'input' parameter.
    Par_max : double
        The end value of Par. Par_min and Par_max can be viscous torque in g*cm^2/s^2, effective temperature in K,
        accretion rate in g/s, in eddington limits or in Msun/yr. Choice depends on 'input' parameter.
    M : double
        Mass of central object in grams.
    alpha : double
        Alpha parameter for alpha-prescription of viscosity.
    r : double
        Distance from central star (radius in cylindrical coordinate system) in cm.
    input : str
        Define the choice of 'Par_min' and 'Par_max' parameters.
        Can be 'F' (viscous torque), 'Teff' (effective temperature, or viscous temperature in case of irradiation),
        'Mdot' (accretion rate), 'Mdot_Mdot_edd' (Mdot in eddington limits) or 'Mdot_Msun_yr' (Mdot in Msun/yr).
    structure : str
        Type of vertical structure. Possible options are:
        'Kramers', 'BellLin' -- ideal gas EoS, analytical opacities and radiative temperature gradient;
        'Mesa', 'MesaAd', 'MesaRadAd', 'MesaRadConv' -- MESA EoS and opacities and
                                                        radiative, adiabatic, rad+ad
                                                        and rad+conv temperature gradient;
        'MesaIdealGas' -- ideal gas EoS, MESA opacities and radiative temperature gradient;
        'MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero' -- simple external irradiation scheme
                                                                   via T_irr or C_irr
        'MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr' -- advanced external irradiation scheme
                                                       from (Mescheryakov et al. 2011).
    mu : double
        Mean molecular weight. Use in case of ideal gas EoS.
    abundance : dict or str
        Chemical composition of disc. Use in case of MESA EoS.
        Format: {'isotope_name': abundance}. For example: {'h1': 0.7, 'he4': 0.3}.
        Use 'solar' str in case of solar composition.
    F_in : double
        Viscous torque in g*cm^2/s^2 at the inner disc radius.
        The viscous torque F = Mdot * h * (1-sqrt(r_in/r)) + F_in.
        Default F_in=0 (when the central source is Schwarzschild black hole).
    nu_irr : array-like
        If structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr'], nu_irr is the irradiation spectral flux argument.
        Can be (X-ray) frequency (in Hz) or energy (in keV) array for spectral external irradiation flux.
        Choose depends on the 'spectrum_irr_par'.
    spectrum_irr : array-like or callable
        If structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr'],
        spectrum_irr is the spectrum of external irradiation flux, i.e.
        the spectral (X-ray) external irradiation flux F_nu_irr = F_irr * spectrum_irr.
        If spectrum_irr is array-like it must be in 1/Hz or in 1/keV depending on 'spectrum_irr_par',
        must be normalised to unity, and its size must be equal to nu_irr.size.
        If spectrum_irr is callable, then
        ``F_nu_irr = F_irr * spectrum_irr(nu_irr, *args_spectrum_irr, **kwargs_spectrum_irr)``.
        The normalisation of the spectrum_irr in that case is performed automatically.
    spectrum_irr_par : str
        Defines the irradiation spectral flux argument.
        Can be 'nu' (frequency in Hz) and 'E_in_keV' (energy in keV).
    args_spectrum_irr, kwargs_spectrum_irr : tuple and dict
        Extra arguments and keyword arguments of spectrum_irr, if it's callable.
        The calling signature is ``spectrum_irr(nu_irr, *args_spectrum_irr, **kwargs_spectrum_irr)``
    L_X_irr : double or None
        If structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr'],
        L_X_irr is the (X-ray) bolometric luminosity of external irradiation source.
        If None, then ``L_X_irr = 0.1 * Mdot * c ** 2``.
        The irradiation flux ``F_irr = L_X_irr / (4 * pi * r ** 2)``.
    cos_theta_irr : double or None
        If structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr'], cos_theta_irr is the cosine of angle
        of incidence for external irradiation flux. If None, ``cos_theta_irr = cos_theta_irr_exp * (z0/r)``.
    cos_theta_irr_exp : double
        If cos_theta_irr is None, ``cos_theta_irr = cos_theta_irr_exp * (z0/r)``.
    C_irr : double
        If structure in ['MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero'],
        C_irr is the irradiation parameter.
    T_irr : double
        If structure in ['MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero'],
        T_irr is the irradiation temperature.
    z0r_start_estimation : double
        Start estimation of z0r free parameter to fit the first point of S-curve.
        Further, z0r estimation at the next point is the z0r value at the previous point.
        Default is None, the start estimation is calculated automatically.
    Sigma0_start_estimation : double
        If structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr'],
        it's the start estimation at Sigma0 free parameter to fit the first point of S-curve.
        Further, Sigma0 estimation at the next point is the 2*Sigma0 value at the previous point.
        Default is None, the start estimation is calculated automatically.
    P_ph_0 : double
        If structure contains Irradiation (either irradiation scheme),
        it's the start estimation for pressure at the photosphere (pressure boundary condition)
        to fit the first point of S-curve. Further, P_ph estimation at the next point
        is the P_ph value at the previous point.
        Default is None, the estimation is calculated automatically.
    verbose : bool
        Whether to print values of free parameters at each iteration during fitting.
        Default is False, the fitting process performs silently.
    n : int
        Number of dots to calculate.
    tau_break : bool
        Whether to end calculation, when tau<1.
    add_Pi_values : bool
        Whether to write Pi-parameters (see Ketsaris & Shakura, 1998) to the output file.
    path_dots : str
        Where to save data table.
    verbose_scurve : bool
        Whether to print info during the S-curve calculation.
        Default is False.
    to_return : bool
        Whether to return a namedtuple containing the calculated quantities and
        debug data. Default is False.
    to_save : bool
        Whether to save the calculated data in a file. Default is True.
    magn_args : dict or None, optional
        The dictionary containing the parameters necessary for the magentic field
        tretment. If None, the behaviour switches to a non-magnetic case.
        Default is None.
    rad_pressure_include : bool, optional
        Whether to include the radiation pressure contribution in the equation
        of state. Default is True.
    Returns
    -------
    Table with calculated S-curve will save to path_dots. Table contains:
        1) input parameters of the system -- M in Msun, alpha, r in cm and in rg, structure type,
           mu (in case of analytical EoS) or abundance (in case of tabular EoS);
        2) Sigma0, Teff (Tvis), Mdot, F, z0r -- surface density,
           effective temperature (viscous temperature in case of irradiation),
           accretion rate, viscous torque, half-thickness of the disc;
        3) rho_c, T_c, P_c, PradPgas_c, varkappa_c, free_e_c -- bulk density, temperature, gas pressure,
           Prad/Pgas, opacity and mean number of free electrons per nucleon in the central plane;
        4) tau -- full Rosseland optical depth of the disc;
    In case of convection:
        5) conv_param_z, conv_param_sigma -- z-fraction and mass-fraction of convective region.
           Changes from 0 (fully radiative) to 1 (fully convective).
    If add_Pi_values:
        6) Pi1, Pi2, Pi3, Pi4 -- Pi-parameters (see Ketsaris & Shakura, 1998).
    If structure contains Irradiation (either irradiation scheme):
        7) QirrQvis, T_irr, C_irr -- Qirr/Qvis, irradiation temperature and irradiation parameter.
    If structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr']:
        8) cost, Sigma_ph -- cost function and column density of the layers above the photosphere.
           If structure is fitted successfully, cost must be less than 1e-16.

    Also table contains Sigma_plus_index, Sigma_minus_index -- turn point indexes of the S-curve.
    Finally, table contains 'Non-converged_fits' -- number of unsuccessfully fitted structures.

    """
    if path_dots is None and to_return:
        raise Exception("ATTENTION: the data wil not be saved, since 'path_dots' is None.")

    Sigma_minus_index = 0
    Sigma_plus_index = 0
    Sigma_minus_key = True
    Sigma_plus_key = True
    if Par_max > Par_min:
        delta_Sigma_plus = -1
    else:
        delta_Sigma_plus = +1
    z0r_estimation = z0r_start_estimation
    sigma_par_estimation = Sigma0_start_estimation

    sigma_temp = np.inf
    Non_converged_fits = 0
    if structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr',
                     'MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero']:
        Teff_string = 'Tvis'
    else:
        Teff_string = 'Teff'

    if path_dots is not None:
        rg = 2 * G * M / c ** 2
        header = f'Sigma0 \t{Teff_string} \tMdot \tF \tz0r \trho_c \tT_c \tP_c \ttau \tPradPgas_c \tvarkappa_c'
        header_end = f'\nM = {M / M_sun:e} Msun, alpha = {alpha}, ' \
                     f'r = {r:e} cm, r = {r / rg} rg, structure = {structure}'
        if structure in ['Kramers', 'BellLin', 'MesaIdealGas']:
            header_end += f', mu = {mu}'
        else:
            header_end += f', abundance = {abundance}'
            header += ' \tfree_e_c \tconv_param_z \tconv_param_sigma'
        if add_Pi_values:
            header += ' \tPi1 \tPi2 \tPi3 \tPi4'
        if structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr',
                         'MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero']:
            header += ' \tQirrQvis \tT_irr \tC_irr'
        if structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr']:
            header += ' \tcost \tSigma_ph'
        header = header + '\nAll values are in CGS units.' + header_end
        if to_return:
            np.savetxt(path_dots, [], header=header)
        
    SCurveResults = namedtuple("SCurveResults", [
                # "r", "r_rg", 
                "Sigma0", "Mdot", "Teff", "F", "z0r", "T_C",
                "rho_C", "P_C", "tau", "PradPgas_C", "varkappa_C",
                # "free_e", "conv_param_z", "conv_param_sigma",
                # "Pi1", "Pi2", "Pi3", "Pi4",
            ])
    acc = {k: [] for k in SCurveResults._fields}

    for i, Par in enumerate(np.geomspace(Par_max, Par_min, n)):
        if verbose_scurve:
            print(i)
        vs, F, Teff, Mdot = StructureChoice(M, alpha, r, Par, input, structure, mu, abundance, F_in=F_in,
                                            nu_irr=nu_irr, L_X_irr=L_X_irr,
                                            spectrum_irr=spectrum_irr, spectrum_irr_par=spectrum_irr_par,
                                            args_spectrum_irr=args_spectrum_irr,
                                            kwargs_spectrum_irr=kwargs_spectrum_irr,
                                            cos_theta_irr=cos_theta_irr, cos_theta_irr_exp=cos_theta_irr_exp,
                                            C_irr=C_irr, T_irr=T_irr, magn_args=magn_args,
                                            rad_pressure_include=rad_pressure_include)
        if structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr']:
            kwargs_fit = {'z0r_estimation': z0r_estimation, 'Sigma0_estimation': sigma_par_estimation,
                          'verbose': verbose, 'P_ph_0': P_ph_0}
        elif structure in ['MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero']:
            kwargs_fit = {'z0r_estimation': z0r_estimation, 'verbose': verbose, 'P_ph_0': P_ph_0}
        else:
            kwargs_fit = {'z0r_estimation': z0r_estimation, 'verbose': verbose}
        try:
            result = vs.fit(**kwargs_fit)
        except Exception as e:
            if verbose_scurve:
                print(e)
                print('Non-converged fit')
            Non_converged_fits += 1
            continue
        try:
            z0r, sigma_par = result.x
            z0r_estimation, sigma_par_estimation = z0r, 2 * sigma_par
        except AttributeError:
            z0r, result = result
            z0r_estimation = z0r
        try:
            P_ph_0 = vs.P_ph_0
        except AttributeError:
            pass

        tau = vs.tau()
        if verbose_scurve:
            print(f'Mdot = {Mdot:1.3e} g/s, {Teff_string} = {Teff:g} K, tau = {tau:g}, z0r = {z0r:g}')

        if tau < 1 and tau_break:
            if verbose_scurve:
                print('Note: tau<1, tau_break=True. Cycle ends, when tau<1.')
            break

        varkappa_C, rho_C, T_C, P_C, Sigma0 = vs.parameters_C()
        PradPgas_C = (4 * sigmaSB) / (3 * c) * T_C ** 4 / P_C

        output_string = [Sigma0, Teff, Mdot, F, z0r, rho_C, T_C, P_C, tau, PradPgas_C, varkappa_C]
        if verbose_scurve:
            print(f'Sigma0 = {Sigma0:g} g/cm^2')
            print('Prad/Pgas_C = ', PradPgas_C)

        rho, eos = vs.law_of_rho(P_C, T_C, full_output=True)
        try:
            _ = eos.c_p
            free_e = np.exp(eos.lnfree_e)
            conv_param_z, conv_param_sigma = Convective_parameter(vs)
            output_string.extend([free_e, conv_param_z, conv_param_sigma])
        except AttributeError:
            pass

        if add_Pi_values:
            output_string.extend(vs.Pi_finder())

        if structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr',
                         'MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero']:
            QirrQvis = vs.Q_irr / vs.Q0
            T_irr_, C_irr_ = vs.T_irr, vs.C_irr
            output_string.extend([QirrQvis, T_irr_, C_irr_])
            if verbose_scurve:
                print(f'T_irr, C_irr = {T_irr_:g} K, {C_irr_:g}')

        if structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr']:
            output_string.extend([result.cost, vs.Sigma_ph])

        if i == 0:
            sigma_temp = Sigma0
        else:
            delta_Sigma_plus = Sigma0 - sigma_temp
            sigma_temp = Sigma0

        if Par_max > Par_min:
            if delta_Sigma_plus > 0.0 and Sigma_plus_key:
                Sigma_plus_index = i - 1 - Non_converged_fits
                Sigma_plus_key = False
            if delta_Sigma_plus < 0.0 and not Sigma_plus_key and Sigma_minus_key:
                Sigma_minus_index = i - 1 - Non_converged_fits
                Sigma_minus_key = False
        else:
            if delta_Sigma_plus < 0.0 and Sigma_minus_key:
                Sigma_minus_index = i - 1 - Non_converged_fits
                Sigma_minus_key = False
            if delta_Sigma_plus > 0.0 and not Sigma_minus_key and Sigma_plus_key:
                Sigma_plus_index = i - 1 - Non_converged_fits
                Sigma_plus_key = False

        output_string = np.array(output_string)
        if to_save:
            with open(path_dots, 'a') as file:
                np.savetxt(file, output_string, newline=' ')
                file.write('\n')
        acc["Sigma0"].append(Sigma0)
        acc["Teff"].append(Teff)
        acc["z0r"].append(z0r)
        acc["Mdot"].append(Mdot)
        acc["F"].append(F)
        acc["rho_C"].append(rho_C)
        acc["P_C"].append(P_C)
        acc["varkappa_C"].append(varkappa_C)
        acc["tau"].append(tau)
        acc["PradPgas_C"].append(PradPgas_C)
        
    if to_save:
        with open(path_dots, 'a') as file:
            file.write(f'# Sigma_plus_index = {Sigma_plus_index:d}  Sigma_minus_index = {Sigma_minus_index:d}')
            file.write(f'\n# Non-converged_fits = {Non_converged_fits}')
    if to_return: 
        acc_np = {k: np.asarray(v, dtype=float) for k, v in acc.items()}
        results_tuple = SCurveResults(**acc_np)
        return Sigma_minus_index, Sigma_plus_index, results_tuple
    return Sigma_minus_index, Sigma_plus_index


def Radial_Profile(M, alpha, r_start, r_end, Par, input, structure, mu=0.6, abundance='solar', F_in=0, #!!!
                   nu_irr=None, L_X_irr=None, spectrum_irr=None, spectrum_irr_par=None, args_spectrum_irr=(),
                   kwargs_spectrum_irr={}, cos_theta_irr=None, cos_theta_irr_exp=1 / 12, C_irr=None, T_irr=None,
                   z0r_start_estimation=None, Sigma0_start_estimation=None, P_ph_0=None, verbose=False,
                   verbose_rad=False, to_return=False, to_save=True,
                   n=100, tau_break=True, add_Pi_values=True, path_dots=None,
                   magn_args=None,
                   rad_pressure_include=True):
    """
    Calculates radial structure of disc. Return table, which contains input parameters of the system,
    surface density Sigma0, viscous torque F, accretion rate Mdot, effective temperature Teff
    (or Tvis in case of external irradiation), geometrical half-thickness of the disc z0r and
    parameters in the symmetry plane of disc as functions of radius.

    Parameters
    ----------
    M : double
        Mass of central object in grams.
    alpha : double
        Alpha parameter for alpha-prescription of viscosity.
    r_start : double
        The starting value of radius. Radius (in cylindrical coordinate system, in cm)
        is the distance from central star.
    r_end : double
        The end value of radius. Radius (in cylindrical coordinate system, in cm)
        is the distance from central star.
    Par : double or array-like
        Can be viscous torque in g*cm^2/s^2, effective temperature in K,
        accretion rate in g/s, in eddington limits or in Msun/yr.
        Choice depends on 'input' parameter.
        Par can be array-like, that is, it changes along the radius.
        In that case its size must be equal to n (number of dots to calculate).
    input : str
        Define the choice of 'Par' parameter.
        Can be 'F' (viscous torque), 'Teff' (effective temperature, or viscous temperature in case of irradiation),
        'Mdot' (accretion rate), 'Mdot_Mdot_edd' (Mdot in eddington limits) or 'Mdot_Msun_yr' (Mdot in Msun/yr).
    structure : str
        Type of vertical structure. Possible options are:
        'Kramers', 'BellLin' -- ideal gas EoS, analytical opacities and radiative temperature gradient;
        'Mesa', 'MesaAd', 'MesaRadAd', 'MesaRadConv' -- MESA EoS and opacities and
                                                        radiative, adiabatic, rad+ad
                                                        and rad+conv temperature gradient;
        'MesaIdealGas' -- ideal gas EoS, MESA opacities and radiative temperature gradient;
        'MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero' -- simple external irradiation scheme
                                                                   via T_irr or C_irr
        'MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr' -- advanced external irradiation scheme
                                                       from (Mescheryakov et al. 2011).
    mu : double
        Mean molecular weight. Use in case of ideal gas EoS.
    abundance : dict or str
        Chemical composition of disc. Use in case of MESA EoS.
        Format: {'isotope_name': abundance}. For example: {'h1': 0.7, 'he4': 0.3}.
        Use 'solar' str in case of solar composition.
    F_in : double
        Viscous torque in g*cm^2/s^2 at the inner disc radius.
        The viscous torque F = Mdot * h * (1-sqrt(r_in/r)) + F_in.
        Default F_in=0 (when the central source is Schwarzschild black hole).
    nu_irr : array-like
        If structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr'], nu_irr is the irradiation spectral flux argument.
        Can be (X-ray) frequency (in Hz) or energy (in keV) array for spectral external irradiation flux.
        Choose depends on the 'spectrum_irr_par'.
    spectrum_irr : array-like or callable
        If structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr'],
        spectrum_irr is the spectrum of external irradiation flux, i.e.
        the spectral (X-ray) external irradiation flux F_nu_irr = F_irr * spectrum_irr.
        If spectrum_irr is array-like it must be in 1/Hz or in 1/keV depending on 'spectrum_irr_par',
        must be normalised to unity, and its size must be equal to nu_irr.size.
        If spectrum_irr is callable, then
        ``F_nu_irr = F_irr * spectrum_irr(nu_irr, *args_spectrum_irr, **kwargs_spectrum_irr)``.
        The normalisation of the spectrum_irr in that case is performed automatically.
    spectrum_irr_par : str
        Defines the irradiation spectral flux argument.
        Can be 'nu' (frequency in Hz) and 'E_in_keV' (energy in keV).
    args_spectrum_irr, kwargs_spectrum_irr : tuple and dict
        Extra arguments and keyword arguments of spectrum_irr, if it's callable.
        The calling signature is ``spectrum_irr(nu_irr, *args_spectrum_irr, **kwargs_spectrum_irr)``
    L_X_irr : double or None
        If structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr'],
        L_X_irr is the (X-ray) bolometric luminosity of external irradiation source.
        If None, then ``L_X_irr = 0.1 * Mdot * c ** 2``.
        The irradiation flux ``F_irr = L_X_irr / (4 * pi * r ** 2)``.
    cos_theta_irr : double or array-like or None
        If structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr'], cos_theta_irr is the cosine of angle
        of incidence for external irradiation flux. If None, ``cos_theta_irr = cos_theta_irr_exp * (z0/r)``.
        cos_theta_irr can be array-like, that is, it changes along the radius.
        In that case its size must be equal to n (number of dots to calculate).
    cos_theta_irr_exp : double or array-like
        If cos_theta_irr is None, ``cos_theta_irr = cos_theta_irr_exp * (z0/r)``.
        cos_theta_irr_exp can be array-like, that is, it changes along the radius.
        In that case its size must be equal to n (number of dots to calculate).
    C_irr : double or array-like
        If structure in ['MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero'],
        C_irr is the irradiation parameter.
        C_irr can be array-like, that is, it changes along the radius.
        In that case its size must be equal to n (number of dots to calculate).
    T_irr : double or array-like
        If structure in ['MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero'],
        T_irr is the irradiation temperature.
        T_irr can be array-like, that is, it changes along the radius.
        In that case its size must be equal to n (number of dots to calculate).
    z0r_start_estimation : double
        Start estimation of z0r free parameter to fit the first point of radial structure.
        Further, z0r estimation at the next point is the z0r value at the previous point.
        Default is None, the start estimation is calculated automatically.
    Sigma0_start_estimation : double
        If structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr'],
        it's the start estimation of Sigma0 free parameter to fit the first point of radial structure.
        Further, Sigma0 estimation at the next point is the 2*Sigma0 value at the previous point.
        Default is None, the start estimation is calculated automatically.
    P_ph_0 : double
        If structure contains Irradiation (either irradiation scheme),
        it's the start estimation for pressure at the photosphere (pressure boundary condition)
        to fit the first point of S-curve. Further, P_ph estimation at the next point
        is the P_ph value at the previous point.
        Default is None, the estimation is calculated automatically.
    verbose : bool
        Whether to print values of free parameters at each iteration during fitting.
        Default is False, the fitting process performs silently.
    verbose_rad : bool
        Whether to print info during radial structure calculation at every radius.
        Default is False, the radial calculation structure calculation process
        performs silently.
    to_return : bool
        Whether to return a namedtuple containing the calculated quantities and
        debug data. Default is False.
    to_save : bool
        Whether to save the calculated data in a file. Default is True.
    magn_args : dict or None, optional
        The dictionary containing the parameters necessary for the magentic field
        tretment. If None, the behaviour switches to a non-magnetic case.
        Default is None.
    rad_pressure_include : bool, optional
        Whether to include the radiation pressure contribution in the equation
        of state. Default is True.
        
    n : int
        Number of dots to calculate.
    tau_break : bool
        Whether to end calculation, when tau<1.
    add_Pi_values : bool
        Whether to write Pi-parameters (see Ketsaris & Shakura, 1998) to the output file.
    path_dots : str
        Where to save data table.

    Returns
    -------
    -------------------------- if to_save == True: ----------------------------
        
    Table with calculated Radial disc profile will save to path_dots. Table contains:
        1) input parameters of the system -- M in Msun, alpha, structure type,
           mu (in case of analytical EoS) or abundance (in case of tabular EoS);
        2) r, r/rg, Sigma0, Teff (Tvis), Mdot, F, z0r -- surface density,
           effective temperature (viscous temperature in case of irradiation),
           accretion rate, viscous torque, half-thickness of the disc;
        3) rho_c, T_c, P_c, PradPgas_c, varkappa_c, free_e_c -- bulk density, temperature, gas pressure,
           Prad/Pgas, opacity and mean number of free electrons per nucleon in the central plane;
        4) tau -- full Rosseland optical depth of the disc.
    In case of convection:
        5) conv_param_z, conv_param_sigma -- z-fraction and mass-fraction of convective region.
           Changes from 0 (fully radiative) to 1 (fully convective).
    If add_Pi_values:
        6) Pi1, Pi2, Pi3, Pi4 -- Pi-parameters (see Ketsaris & Shakura, 1998).
    If structure contains Irradiation (either irradiation scheme):
        7) QirrQvis, T_irr, C_irr -- Qirr/Qvis, irradiation temperature and irradiation parameter.
    If structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr']:
        8) cost, Sigma_ph -- cost function and column density of the layers above the photosphere.
           If structure is fitted successfully, cost must be less than 1e-16.
    Table also contains 'Non-converged_fits' -- number of unsuccessfully fitted structures.
    
    ------------------------- if to_return == True: --------------------------
    RadStructureResults : namedtuple 
        Contains the calculated quantities as numpy arrays. The attributes
        have the same names as described above, 
        e.g., the dependence of the Teff on radius may be plotted as
         plt.plot(res.r, res.Teff).
    
    """
    if (path_dots is None) and to_save:
        raise Exception("ATTENTION: the data wil not be saved, since 'path_dots' is None.")

    z0r_estimation = z0r_start_estimation
    sigma_par_estimation = Sigma0_start_estimation
    Non_converged_fits = 0
    r_arr = np.geomspace(r_start, r_end, n)

    if structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr',
                     'MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero']:
        Teff_string = 'Tvis'
    else:
        Teff_string = 'Teff'

    if path_dots is not None:
        header = f'r \tr/rg \tSigma0 \tMdot \t{Teff_string} \tF \tz0r \trho_c \tT_c \tP_c ' \
                 f'\ttau \tPradPgas_c \tvarkappa_c'
        header_end = f'\nM = {M / M_sun:e} Msun, alpha = {alpha}, structure = {structure}'
        if structure in ['Kramers', 'BellLin', 'MesaIdealGas']:
            header_end += f', mu = {mu}'
        else:
            header_end += f', abundance = {abundance}'
            header += ' \tfree_e_c \tconv_param_z \tconv_param_sigma'
        if add_Pi_values:
            header += ' \tPi1 \tPi2 \tPi3 \tPi4'
        if structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr',
                         'MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero']:
            header += ' \tQirrQvis \tT_irr \tC_irr'
        if structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr']:
            header += ' \tcost \tSigma_ph'
        header = header + '\nAll values are in CGS units.' + header_end
        if to_save:
            np.savetxt(path_dots, [], header=header)

    try:
        input_broadcast = np.broadcast(r_arr, Par, cos_theta_irr, cos_theta_irr_exp, C_irr, T_irr)
    except ValueError as e:
        raise ValueError(f'Array-like input parameters must have the same size n = {n}.') from e
    RadStructureResults = namedtuple("RadStructureResults", [
                "r", "r_rg", "Sigma0", "Mdot", "Teff", "F", "z0r",
                "rho_C", "T_C", "P_C", "tau", "PradPgas_C", "varkappa_C",
                "free_e", "conv_param_z", "conv_param_sigma",
                "Pi1", "Pi2", "Pi3", "Pi4",
                "QirrQvis", "T_irr_", "C_irr_",
                "result_cost", "vs_Sigma_ph",
            ])
    acc = {k: [] for k in RadStructureResults._fields}

    for i, input_pars in enumerate(input_broadcast):
        if verbose_rad:
            print(i)
        r = input_pars[0]
        vs, F, Teff, Mdot = StructureChoice(M=M, alpha=alpha, r=r, Par=input_pars[1], input=input,
                                            structure=structure, mu=mu, abundance=abundance, F_in=F_in,
                                            nu_irr=nu_irr, L_X_irr=L_X_irr,
                                            spectrum_irr=spectrum_irr, spectrum_irr_par=spectrum_irr_par,
                                            args_spectrum_irr=args_spectrum_irr,
                                            kwargs_spectrum_irr=kwargs_spectrum_irr,
                                            cos_theta_irr=input_pars[2], cos_theta_irr_exp=input_pars[3],
                                            C_irr=input_pars[4], T_irr=input_pars[5],
                                            magn_args=magn_args,
                                            rad_pressure_include=rad_pressure_include)
        if structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr']:
            kwargs_fit = {'z0r_estimation': z0r_estimation, 'Sigma0_estimation': sigma_par_estimation,
                          'verbose': verbose, 'P_ph_0': P_ph_0}
        elif structure in ['MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero']:
            kwargs_fit = {'z0r_estimation': z0r_estimation, 'verbose': verbose, 'P_ph_0': P_ph_0}
        else:
            kwargs_fit = {'z0r_estimation': z0r_estimation, 'verbose': verbose}
        try:
            result = vs.fit(**kwargs_fit)
        except Exception as e:
            if verbose_rad:
                print(e)
                print('Non-converged fit')
            Non_converged_fits += 1
            continue
        try:
            z0r, sigma_par = result.x
            z0r_estimation, sigma_par_estimation = z0r, 2 * sigma_par
        except AttributeError:
            z0r, result = result
            z0r_estimation = z0r
        try:
            P_ph_0 = vs.P_ph_0
        except AttributeError:
            pass

        tau = vs.tau()
        rg = 2 * G * M / c ** 2
        if verbose_rad:
            print(f'r = {r:1.3e} cm = {r / rg:g} rg, Mdot = {Mdot:1.3e} g/s, '
              f'{Teff_string} = {Teff:g} K, tau = {tau:g}, z0r = {z0r:g}')

        if tau < 1 and tau_break:
            if verbose_rad:
                print('Note: tau<1, tau_break=True. Cycle ends, when tau<1.')
            break

        varkappa_C, rho_C, T_C, P_C, Sigma0 = vs.parameters_C()
        PradPgas_C = (4 * sigmaSB) / (3 * c) * T_C ** 4 / P_C

        output_string = [r, r / rg, Sigma0, Mdot, Teff, F, z0r, rho_C, T_C, P_C, tau, PradPgas_C, varkappa_C] 
        if verbose_rad:
            print(f'Sigma0 = {Sigma0:g} g/cm^2')
            print('Prad/Pgas_C = ', PradPgas_C)

        rho, eos = vs.law_of_rho(P_C, T_C, full_output=True)
        free_e, conv_param_z, conv_param_sigma = np.nan, np.nan, np.nan
        try:
            _ = eos.c_p
            free_e = np.exp(eos.lnfree_e)
            conv_param_z, conv_param_sigma = Convective_parameter(vs)
            output_string.extend([free_e, conv_param_z, conv_param_sigma])
        except AttributeError:
            pass
        Pi1, Pi2, Pi3, Pi4 = np.nan, np.nan, np.nan, np.nan
        if add_Pi_values:
            Pi1, Pi2, Pi3, Pi4 = vs.Pi_finder()
            output_string.extend([Pi1, Pi2, Pi3, Pi4])
        QirrQvis, T_irr_, C_irr_ = np.nan, np.nan, np.nan
        if structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr',
                         'MesaIrrZero', 'MesaRadAdIrrZero', 'MesaRadConvIrrZero']:
            QirrQvis = vs.Q_irr / vs.Q0
            T_irr_, C_irr_ = vs.T_irr, vs.C_irr
            output_string.extend([QirrQvis, T_irr_, C_irr_])
            if verbose_rad:
                print(f'T_irr, C_irr = {T_irr_:g} K, {C_irr_:g}')
        
        result_cost, vs_Sigma_ph = np.nan, np.nan
        if structure in ['MesaIrr', 'MesaRadAdIrr', 'MesaRadConvIrr']:
            result_cost, vs_Sigma_ph = result.cost, vs.Sigma_ph
            output_string.extend([result_cost, vs_Sigma_ph])

        output_string = np.array(output_string)
        if to_save:
            with open(path_dots, 'a') as file:
                np.savetxt(file, output_string, newline=' ')
                file.write('\n')
        acc["r"].append(r)
        acc["r_rg"].append(r / rg)
        acc["Sigma0"].append(Sigma0)
        acc["Mdot"].append(Mdot)
        acc["Teff"].append(Teff)
        acc["F"].append(F)
        acc["z0r"].append(z0r)
        acc["rho_C"].append(rho_C)
        acc["T_C"].append(T_C)
        acc["P_C"].append(P_C)
        acc["tau"].append(tau)
        acc["PradPgas_C"].append(PradPgas_C)
        acc["varkappa_C"].append(varkappa_C)
        acc["free_e"].append(free_e)
        acc["conv_param_z"].append(conv_param_z)
        acc["conv_param_sigma"].append(conv_param_sigma)
        acc["Pi1"].append(Pi1)
        acc["Pi2"].append(Pi2)
        acc["Pi3"].append(Pi3)
        acc["Pi4"].append(Pi4)
        acc["QirrQvis"].append(QirrQvis)
        acc["T_irr_"].append(T_irr_)
        acc["C_irr_"].append(C_irr_)
        acc["result_cost"].append(result_cost)
        acc["vs_Sigma_ph"].append(vs_Sigma_ph)
    if to_save:
        with open(path_dots, 'a') as file:
            file.write(f'# Non-converged_fits = {Non_converged_fits}')
    acc_np = {k: np.asarray(v, dtype=float) for k, v in acc.items()}
    results_tuple = RadStructureResults(**acc_np)
    if to_return:
        return results_tuple
    return


def main():
    from matplotlib import pyplot as plt
    import os
    M = 5 * M_sun
    alpha = 0.1
    r = 1e10
    Teff = 1e4
    os.makedirs('fig/', exist_ok=True)

    print('Calculation of vertical structure. Return structure table.')
    print(f'M = {M / M_sun:g} M_sun \nr = {r:g} cm \nalpha = {alpha:g} \nTeff = {Teff:g} K\n')

    Vertical_Profile(M, alpha, r, Teff, input='Teff', structure='BellLin', mu=0.62,
                     n=100, add_Pi_values=True, path_dots='fig/vs.dat')
    print('Structure is calculated successfully. Table is saved to fig/vs.dat.')
    vs_data = np.genfromtxt('fig/vs.dat', names=True)
    print('Making the structure plot.')
    plt.plot(1 - vs_data['t'], vs_data['S'], label=r'$\hat{\Sigma}$')
    plt.plot(1 - vs_data['t'], vs_data['P'], label=r'$\hat{P}$')
    plt.plot(1 - vs_data['t'], vs_data['Q'], label=r'$\hat{Q}$')
    plt.plot(1 - vs_data['t'], vs_data['T'], label=r'$\hat{T}$')
    plt.grid()
    plt.legend()
    plt.xlabel('$z / z_0$')
    plt.title(rf'$M = {M / M_sun:g} \, M_{{\odot}}, r = {r:g} \, {{\rm cm}}, '
              rf'\alpha = {alpha:g}, T_{{\rm eff}} = {Teff:g} \, {{\rm K}}$')
    plt.tight_layout()
    plt.savefig('fig/vs.pdf')
    plt.close()
    print('Plot of structure is successfully saved to fig/vs.pdf.\n')

    print('Calculation of S-curve for Teff from 4e3 K to 1e4 K. Return S-curve table.')
    S_curve(4e3, 1e4, M, alpha, r, input='Teff', structure='BellLin', mu=0.62, n=200, tau_break=True,
            path_dots='fig/S-curve.dat', add_Pi_values=True)
    print('S-curve is calculated successfully. Table is saved to fig/S-curve.dat.')
    s_curve_data = np.genfromtxt('fig/S-curve.dat', names=True)
    tau = s_curve_data['tau']
    print('Making the S-curve plot.')
    plt.plot(s_curve_data['Sigma0'][tau > 1], s_curve_data['Teff'][tau > 1])
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'$T_{\rm eff}, \, \rm K$')
    plt.xlabel(r'$\Sigma_0, \, \rm g/cm^2$')
    plt.grid(True, which='both', ls='-')
    plt.title(rf'$M = {M / M_sun:g} \, M_{{\odot}}, r = {r:g} \, {{\rm cm}}, \alpha = {alpha:g}$')
    plt.tight_layout()
    plt.savefig('fig/S-curve.pdf')
    plt.close()
    print('Plot of S-curve is successfully saved to fig/S-curve.pdf.\n')

    print('Calculation of radial structure of disc for radius from 1e9 cm to 1e12 cm and Mdot = Mdot_edd. '
          'Return radial structure table.')
    Radial_Profile(M, alpha, 1e9, 1e12, 1, input='Mdot_Mdot_edd', structure='BellLin', mu=0.62, n=200,
                   tau_break=True, path_dots='fig/radial_struct.dat', add_Pi_values=True)
    print('Radial structure is calculated successfully. Table is saved to fig/radial_struct.dat.')
    rad_struct_data = np.genfromtxt('fig/radial_struct.dat', names=True)
    print('Making the radial structure plot.')
    plt.plot(rad_struct_data['r'], rad_struct_data['z0r'])
    plt.xscale('log')
    plt.ylabel(r'$z_0 / r$')
    plt.xlabel(r'$r, \,\rm cm$')
    plt.grid(True, which='both', ls='-')
    plt.title(rf'$M = {M / M_sun:g} \, M_{{\odot}}, \dot{{M}} = 1\,\dot{{M}}_{{\rm edd}}, \alpha = {alpha:g}$')
    plt.tight_layout()
    plt.savefig('fig/radial_struct.pdf')
    plt.close()
    print('Plot of radial structure is successfully saved to fig/radial_struct.pdf.')

    return


if __name__ == '__main__':
    main()
