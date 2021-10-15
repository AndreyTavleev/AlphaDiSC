#!/usr/bin/env python3
"""
Module contains functions that calculate vertical structure and S-curve. Functions return tables with calculated data
and make plots of structure or S-curve.

Structure_Plot -- calculates vertical structure and makes table with disc parameters as functions of vertical
    coordinate. Table also contains input parameters of structure, parameters in the symmetry plane and
    parameter normalisations. Also makes a plot of structure (if 'make_pic' parameter is True).
S_curve -- Calculates S-curve and makes table with disc parameters on the S-curve.
    Table contains input parameters of system, surface density Sigma0, viscous torque F,
    accretion rate Mdot, effective temperature Teff, geometrical half-thickness of the disc z0r,
    parameters in the symmetry plane of disc on the S-curve.
    Also makes a plot of S-curve (if 'make_pic' parameter is True).
Radial_Plot -- Calculates radial structure of disc. Return table, which contains input parameters of the system,
    surface density Sigma0, viscous torque F, accretion rate Mdot, effective temperature Teff,
    geometrical half-thickness of the disc z0r and parameters in the symmetry plane of disc
    as functions of radius.

"""
import numpy as np
from astropy import constants as const
from matplotlib import pyplot as plt
from scipy.integrate import simps
from scipy.interpolate import InterpolatedUnivariateSpline

import vs as vert

try:
    import mesa_vs
except ImportError:
    mesa_vs = np.nan

sigmaSB = const.sigma_sb.cgs.value
G = const.G.cgs.value
M_sun = const.M_sun.cgs.value
c = const.c.cgs.value


def StructureChoice(M, alpha, r, Par, input, structure, mu=0.6, abundance='solar'):
    h = np.sqrt(G * M * r)
    rg = 2 * G * M / c ** 2
    r_in = 3 * rg
    if r <= r_in:
        raise Exception('Radius r should be greater than inner radius r_in = 3*rg. '
                        'Actual radius r = {:g} rg'.format(r / rg))
    func = 1 - np.sqrt(r_in / r)
    if input == 'Teff':
        Teff = Par
        F = 8 * np.pi / 3 * h ** 7 / (G * M) ** 4 * sigmaSB * Par ** 4
        Mdot = F / (h * func)
    elif input == 'Mdot':
        Mdot = Par
        F = Par * h * func
        Teff = (3 / (8 * np.pi) * (G * M) ** 4 * F / (sigmaSB * h ** 7)) ** (1 / 4)
    elif input == 'F':
        F = Par
        Mdot = Par / (h * func)
        Teff = (3 / (8 * np.pi) * (G * M) ** 4 * Par / (sigmaSB * h ** 7)) ** (1 / 4)
    elif input == 'Mdot_Mdot_edd':
        Mdot = Par * 1.39e18 * M / M_sun
        F = Mdot * h * func
        Teff = (3 / (8 * np.pi) * (G * M) ** 4 * F / (sigmaSB * h ** 7)) ** (1 / 4)
    else:
        print('Incorrect input, try Teff, Mdot, F of Mdot_Mdot_edd')
        raise Exception

    if structure == 'Kramers':
        vs = vert.IdealKramersVerticalStructure(M, alpha, r, F, mu=mu)
    elif structure == 'BellLin':
        vs = vert.IdealBellLin1994VerticalStructure(M, alpha, r, F, mu=mu)
    elif structure == 'Mesa':
        try:
            if np.isnan(mesa_vs):
                raise ModuleNotFoundError('Mesa2py is not installed')
        except TypeError:
            vs = mesa_vs.MesaVerticalStructure(M, alpha, r, F, abundance=abundance)
    elif structure == 'MesaIdeal':
        try:
            if np.isnan(mesa_vs):
                raise ModuleNotFoundError('Mesa2py is not installed')
        except TypeError:
            vs = mesa_vs.MesaIdealVerticalStructure(M, alpha, r, F, mu=mu, abundance=abundance)
    elif structure == 'MesaAd':
        try:
            if np.isnan(mesa_vs):
                raise ModuleNotFoundError('Mesa2py is not installed')
        except TypeError:
            vs = mesa_vs.MesaVerticalStructureAdiabatic(M, alpha, r, F, abundance=abundance)
    elif structure == 'MesaFirst':
        try:
            if np.isnan(mesa_vs):
                raise ModuleNotFoundError('Mesa2py is not installed')
        except TypeError:
            vs = mesa_vs.MesaVerticalStructureFirstAssumption(M, alpha, r, F, abundance=abundance)
    elif structure == 'MesaRadConv':
        try:
            if np.isnan(mesa_vs):
                raise ModuleNotFoundError('Mesa2py is not installed')
        except TypeError:
            vs = mesa_vs.MesaVerticalStructureRadConv(M, alpha, r, F, abundance=abundance)
    else:
        print('Incorrect structure, try Kramers, BellLin, Mesa, MesaIdeal, MesaAd, MesaFirst or MesaRadConv')
        raise Exception

    return vs, F, Teff, Mdot


def Convective_parameter(vs):
    """
    Calculates convective parameter of structure. This parameter shows what part of disc is convective.

    Parameters
    ----------
    vs : vertical structure
        Fitted vertical structure, for which convective parameter is calculated.

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
    grad_plot = InterpolatedUnivariateSpline(np.log(P), np.log(T)).derivative()
    rho, eos = vs.law_of_rho(P * vs.P_norm, T * vs.T_norm, True)
    try:
        _ = eos.grad_ad
    except AttributeError:
        print('Incorrect vertical structure. Use vertical structure with Mesa EOS.')
        raise Exception
    conv_param_sigma = simps(2 * rho * (grad_plot(np.log(P)) > eos.grad_ad), t * vs.z0) / (
            S[-1] * vs.sigma_norm)
    conv_param_z = simps(grad_plot(np.log(P)) > eos.grad_ad, t * vs.z0) / vs.z0
    return conv_param_z, conv_param_sigma


def Structure_Plot(M, alpha, r, Par, input='Teff', mu=0.6, structure='BellLin', abundance='solar', n=100,
                   add_Pi_values=True, savedots=True, path_dots='vs.dat', make_pic=True,
                   save_plot=True, path_plot='vs.pdf', set_title=True, title='Vertical structure'):
    """
    Calculates vertical structure and makes table with disc parameters as functions of vertical coordinate.
    Table also contains input parameters of structure, parameters in the symmetry plane and parameter normalisations.
    Also makes a plot of structure (if 'make_pic' parameter is True).

    Parameters
    ----------
    M : double
        Mass of central object in grams.
    alpha : double
        Alpha parameter for alpha-prescription of viscosity.
    r : double
        Distance from central star (radius in cylindrical coordinate system) in cm.
    Par : double
        Can be viscous torque in g*cm^2/s^2, effective temperature in K, accretion rate in g/s or in eddington limits.
        Choice depends on 'input' parameter.
    input : str
        Define the choice of 'Par' parameter. Can be 'F' (viscous torque), 'Teff' (effective temperature),
        'Mdot' (accretion rate) or 'Mdot_Mdot_edd' (Mdot in eddington limits).
    mu : double
        Molecular weight. Use in case of ideal gas EOS.
    structure : str
        Type of vertical structure. Can be 'Kramers', 'BellLin',
        'Mesa', 'MesaIdeal', 'MesaAd', 'MesaFirst' or 'MesaRadConv'.
    abundance : dict or str
        Chemical composition of disc. Use in case of Mesa EOS.
        Format: {'isotope_name': abundance}. For example: {'h1': 0.7, 'he4': 0.3}.
        Use 'solar' str in case of solar composition.
    n : int
        Number of dots to calculate.
    add_Pi_values : bool
        Whether to write Pi-parameters (see Ketsaris & Shakura, 1998) to the output file header.
    savedots : bool
        Whether to save data in table.
    path_dots : str
        Where to save data table.
    make_pic : bool
        Whether to make plot of structure.
    save_plot : bool
        Whether to save plot.
    path_plot : str
        Where to save structure plot.
    set_title : bool
        Whether to make title of the plot.
    title : str
        The title of the plot.

    """
    vs, F, Teff, Mdot = StructureChoice(M, alpha, r, Par, input, structure, mu, abundance)
    z0r, result = vs.fit()
    rg = 2 * G * M / c ** 2
    t = np.linspace(0, 1, n)
    S, P, Q, T = vs.integrate(t)[0]
    varkappa_C, rho_C, T_C, P_C, Sigma0 = vs.parameters_C()
    tau = vs.tau()
    delta = (4 * sigmaSB) / (3 * c) * T_C ** 4 / P_C
    grad_plot = InterpolatedUnivariateSpline(np.log(P), np.log(T)).derivative()
    rho, eos = vs.law_of_rho(P * vs.P_norm, T * vs.T_norm, True)
    varkappa = vs.law_of_opacity(rho, T * vs.T_norm, lnfree_e=eos.lnfree_e)
    dots_arr = np.c_[t, S, P, Q, T, rho, varkappa, grad_plot(np.log(P))]
    try:
        dots_arr = np.c_[dots_arr, eos.grad_ad, np.exp(eos.lnfree_e)]
        header = 't\t S\t P\t Q\t T\t rho\t varkappa\t grad\t grad_ad\t free_e'
        conv_param_z, conv_param_sigma = Convective_parameter(vs)
        header_conv = '\nconv_param_z = {} \tconv_param_sigma = {}'.format(conv_param_z, conv_param_sigma)
    except AttributeError:
        header = 't\t S\t P\t Q\t T\t rho\t varkappa\t grad'
        header_conv = ''
    header_input = '\nS, P, Q, T -- normalized values, rho -- in g/cm^3, ' \
                   'varkappa -- in cm^2/g \nt = 1 - z/z0 ' \
                   '\nM = {:e} Msun, alpha = {}, r = {:e} cm, r = {} rg, Teff = {} K, Mdot = {:e} g/s, ' \
                   'F = {:e} g*cm^2/s^2, structure = {}'.format(M / M_sun, alpha, r, r / rg, Teff, Mdot, F, structure)
    if structure in ['Kramers', 'BellLin', 'MesaIdeal']:
        header_input += ', mu = {}'.format(mu)
    else:
        header_input += ', abundance = {}'.format(abundance)
    header_C = '\nvarkappa_C = {:e} cm^2/g, rho_C = {:e} g/cm^3, T_C = {:e} K, P_C = {:e} dyn, Sigma0 = {:e} g/cm^2, ' \
               'PradPgas_C = {:e}, z0r = {:e}, tau = {:e}'.format(varkappa_C, rho_C, T_C, P_C, Sigma0, delta, z0r, tau)
    header_norm = '\nSigma_norm = {:e}, P_norm = {:e}, T_norm = {:e}, Q_norm = {:e}'.format(
        vs.sigma_norm, vs.P_norm, vs.T_norm, vs.Q_norm)
    header = header + header_input + header_C + header_norm + header_conv
    if add_Pi_values:
        header += '\nPi1 = {:f}, Pi2 = {:f}, Pi3 = {:f}, Pi4 = {:f}'.format(*vs.Pi_finder())
    if savedots:
        np.savetxt(path_dots, dots_arr, header=header)
    if not make_pic:
        return
    plt.plot(1 - t, S, label=r'$\hat{\Sigma}$')
    plt.plot(1 - t, P, label=r'$\hat{P}$')
    plt.plot(1 - t, Q, label=r'$\hat{Q}$')
    plt.plot(1 - t, T, label=r'$\hat{T}$')
    plt.grid()
    plt.legend()
    plt.xlabel('$z / z_0$')
    if set_title:
        plt.title(title)
    plt.tight_layout()
    if save_plot:
        plt.savefig(path_plot)
        plt.close()


def S_curve(Par_min, Par_max, M, alpha, r, input='Teff', structure='BellLin', mu=0.6, abundance='solar', n=100,
            tau_break=True, savedots=True, path_dots='S_curve.dat', add_Pi_values=True, make_pic=True,
            output='Mdot', xscale='log', yscale='log', save_plot=True, path_plot='S-curve.pdf',
            set_title=True, title='S-curve'):
    """
    Calculates S-curve and makes table with disc parameters on the S-curve.
    Table contains input parameters of system, surface density Sigma0, viscous torque F,
    accretion rate Mdot, effective temperature Teff, geometrical half-thickness of the disc z0r,
    parameters in the symmetry plane of disc on the S-curve.
    Also makes a plot of S-curve (if 'make_pic' parameter is True).

    Parameters
    ----------
    Par_min : double
        The starting value of Par. Par_min and Par_max can be viscous torque in g*cm^2/s^2, effective temperature in K,
        accretion rate in g/s or in eddington limits. Choice depends on 'input' parameter.
    Par_max : double
        The end value of Par. Par_min and Par_max can be viscous torque in g*cm^2/s^2, effective temperature in K,
        accretion rate in g/s or in eddington limits. Choice depends on 'input' parameter.
    M : double
        Mass of central object in grams.
    alpha : double
        Alpha parameter for alpha-prescription of viscosity.
    r : double
        Distance from central star (radius in cylindrical coordinate system) in cm.
    input : str
        Define the choice of 'Par_min' and 'Par_max' parameters.
        Can be 'F' (viscous torque), 'Teff' (effective temperature),
        'Mdot' (accretion rate) or 'Mdot_Mdot_edd' (Mdot in eddington limits).
    structure : str
        Type of vertical structure. Can be 'Kramers', 'BellLin',
        'Mesa', 'MesaIdeal', 'MesaAd', 'MesaFirst' or 'MesaRadConv'.
    mu : double
        Molecular weight. Use in case of ideal gas EOS.
    abundance : dict or str
        Chemical composition of disc. Use in case of Mesa EOS.
        Format: {'isotope_name': abundance}. For example: {'h1': 0.7, 'he4': 0.3}.
        Use 'solar' str in case of solar composition.
    n : int
        Number of dots to calculate.
    tau_break : bool
        Whether to end calculation, when tau<1.
    savedots : bool
        Whether to save data in table.
    path_dots : str
        Where to save data table.
    add_Pi_values : bool
        Whether to write Pi-parameters (see Ketsaris & Shakura, 1998) to the output file.
    make_pic : bool
        Whether to make S-curve plot.
    output : str
        In which y-coordinate draw S-curve plot. Can be 'Teff', 'Mdot', 'Mdot_Mdot_edd', 'F', 'z0r' or 'T_C'.
    xscale : str
        Scale of x-axis. Can be 'linear', 'log' or 'parlog' (linear scale, but logarithmic values).
    yscale : str
        Scale of y-axis. Can be 'linear', 'log' or 'parlog' (linear scale, but logarithmic values).
    save_plot : bool
        Whether to save plot.
    path_plot : str
        Where to save structure plot.
    set_title : bool
        Whether to make title of the plot.
    title : str
        The title of the plot.

    """
    if xscale not in ['linear', 'log', 'parlog']:
        print('Incorrect xscale, try linear, log or parlog')
        raise Exception
    if yscale not in ['linear', 'log', 'parlog']:
        print('Incorrect yscale, try linear, log or parlog')
        raise Exception
    Sigma_plot, Mdot_plot, Teff_plot, F_plot = [], [], [], []
    Output_Plot = []
    varkappa_c_plot, T_c_plot, P_c_plot, rho_c_plot = [], [], [], []
    z0r_plot, tau_plot, PradPgas_Plot = [], [], []
    conv_param_z_plot, conv_param_sigma_plot = [], []
    free_e_plot = []
    Pi_plot = []

    PradPgas10_index = 0  # where Prad = Pgas
    tau_index = n  # where tau < 1
    Sigma_minus_index = 0  # where free_e < 0.5, Sigma_minus
    key = True  # for Prad = Pgas
    tau_key = True  # for tau < 1
    Sigma_minus_key = True  # for free_e < 0.5, Sigma_minus

    Sigma_plus_key = True  # for Sigma_plus
    Sigma_plus_index = 0  # for Sigma_plus
    delta_Sigma_plus = -1

    for i, Par in enumerate(np.geomspace(Par_max, Par_min, n)):
        vs, F, Teff, Mdot = StructureChoice(M, alpha, r, Par, input, structure, mu, abundance)
        z0r, result = vs.fit()

        print('Teff = {:g}, tau = {:g}, z0r = {:g}'.format(Teff, vs.tau(), z0r))

        if vs.tau() < 1 and tau_key:
            tau_index = i
            tau_key = False
            if tau_break:
                print('Note: tau<1, tau_break=True. Cycle ends, when tau<1.')
                break

        varkappa_C, rho_C, T_C, P_C, Sigma0 = vs.parameters_C()
        Sigma_plot.append(Sigma0)
        varkappa_c_plot.append(varkappa_C)
        T_c_plot.append(T_C)
        rho_c_plot.append(rho_C)
        z0r_plot.append(z0r)
        tau_plot.append(vs.tau())
        P_c_plot.append(P_C)
        Mdot_plot.append(Mdot)
        Teff_plot.append(Teff)
        F_plot.append(F)

        if i == 0:
            sigma_temp = Sigma0
        else:
            delta_Sigma_plus = Sigma0 - sigma_temp
            sigma_temp = Sigma0

        if make_pic:
            if output == 'Teff':
                Output_Plot.append(Teff)
            elif output == 'Mdot':
                Output_Plot.append(Mdot)
            elif output == 'Mdot_Mdot_edd':
                Mdot_edd = 1.39e18 * M / M_sun
                Output_Plot.append(Mdot / Mdot_edd)
            elif output == 'F':
                Output_Plot.append(F)
            elif output == 'z0r':
                Output_Plot.append(z0r)
            elif output == 'T_C':
                Output_Plot.append(T_C)
            else:
                print('Incorrect output, try Teff, Mdot, Mdot_Mdot_edd, F, z0r or T_C')
                raise Exception

        delta = (4 * sigmaSB) / (3 * c) * T_C ** 4 / P_C
        PradPgas_Plot.append(delta)

        if add_Pi_values:
            Pi_plot.append(vs.Pi_finder())

        if delta < 1.0 and key:
            PradPgas10_index = i
            key = False
        if delta_Sigma_plus > 0.0 and Sigma_plus_key:
            Sigma_plus_index = i
            Sigma_plus_key = False

        rho, eos = vs.law_of_rho(P_C, T_C, full_output=True)
        try:
            _ = eos.grad_ad
            free_e = np.exp(eos.lnfree_e)
            free_e_plot.append(free_e)
            conv_param_z, conv_param_sigma = Convective_parameter(vs)
            conv_param_z_plot.append(conv_param_z)
            conv_param_sigma_plot.append(conv_param_sigma)
            if free_e < (1 + vs.mesaop.X) / 4 and Sigma_minus_key:
                Sigma_minus_index = i
                Sigma_minus_key = False
        except AttributeError:
            pass
        print(i + 1)

    if savedots:
        rg = 2 * G * M / c ** 2
        header = 'Sigma0 \tTeff \tMdot \tF \tz0r \trho_c \tT_c \tP_c \ttau \tPradPgas \tvarkappa_c'
        header_end = '\nM = {:e} Msun, alpha = {}, r = {:e} cm, r = {} rg, structure = {}'.format(
            M / M_sun, alpha, r, r / rg, structure)
        if structure in ['Kramers', 'BellLin', 'MesaIdeal']:
            header_end += ', mu = {}'.format(mu)
        else:
            header_end += ', abundance = {}'.format(abundance)
        header_end += '\nSigma_plus_index = {:d} \tSigma_minus_index = {:d}'.format(Sigma_plus_index, Sigma_minus_index)
        dots_table = np.c_[Sigma_plot, Teff_plot, Mdot_plot, F_plot, z0r_plot, rho_c_plot,
                           T_c_plot, P_c_plot, tau_plot, PradPgas_Plot, varkappa_c_plot]
        if len(free_e_plot) != 0:
            header += ' \tfree_e \tconv_param_z \tconv_param_sigma'
            dots_table = np.c_[dots_table, free_e_plot, conv_param_z_plot, conv_param_sigma_plot]
        if add_Pi_values:
            header += ' \tPi1 \tPi2 \tPi3 \tPi4'
            dots_table = np.c_[dots_table, Pi_plot]
        header = header + '\nAll values are in CGS units.' + header_end
        np.savetxt(path_dots, dots_table, header=header)

    if not make_pic:
        return

    xlabel = r'$\Sigma_0, \, \rm g/cm^2$'

    if xscale == 'parlog':
        Sigma_plot = np.log10(Sigma_plot)
        xlabel = r'$\log \,$' + xlabel
    if yscale == 'parlog':
        Output_Plot = np.log10(Output_Plot)

    pl, = plt.plot(Sigma_plot[:tau_index + 1], Output_Plot[:tau_index + 1],
                   label=r'$P_{{\rm gas}} > P_{{\rm rad}}, \alpha = {:g}$'.format(alpha))
    if tau_index != n:
        plt.plot(Sigma_plot[tau_index:], Output_Plot[tau_index:], color=pl.get_c(), alpha=0.5, label=r'$\tau<1$')
    if PradPgas10_index != 0:
        plt.plot(Sigma_plot[:PradPgas10_index + 1], Output_Plot[:PradPgas10_index + 1],
                 label=r'$P_{\rm gas} < P_{\rm rad}$')

    if xscale != 'parlog':
        plt.xscale(xscale)
    if yscale != 'parlog':
        plt.yscale(yscale)

    if output == 'Teff':
        ylabel = r'$T_{\rm eff}, \, \rm K$'
    elif output == 'Mdot':
        ylabel = r'$\dot{M}, \, \rm g/s$'
    elif output == 'Mdot_Mdot_edd':
        ylabel = r'$\dot{M}/\dot{M}_{\rm edd} $'
    elif output == 'F':
        ylabel = r'$F, \, \rm g~cm^2 / s^2$'
    elif output == 'z0r':
        ylabel = r'$z_0/r$'
    elif output == 'T_C':
        ylabel = r'$T_{\rm c}, \rm K$'
    if yscale == 'parlog':
        ylabel = r'$\log \,$' + ylabel
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.grid(True, which='both', ls='-')
    plt.legend()
    if set_title:
        plt.title(title)
    plt.tight_layout()
    if save_plot:
        plt.savefig(path_plot)
        plt.close()


def Radial_Plot(M, alpha, r_start, r_end, Par, input='Mdot', structure='BellLin', mu=0.6, abundance='solar',
                n=100, tau_break=True, savedots=True, path_dots='radial_struct.dat'):
    """
    Calculates radial structure of disc. Return table, which contains input parameters of the system,
    surface density Sigma0, viscous torque F, accretion rate Mdot, effective temperature Teff,
    geometrical half-thickness of the disc z0r and parameters in the symmetry plane of disc
    as functions of radius.

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
    Par : double
        Par can be accretion rate in g/s or in eddington limits. Choice depends on 'input' parameter.
    input : str
        Define the choice of 'Par' parameter.
        Can be 'Mdot' (accretion rate) or 'Mdot_Mdot_edd' (Mdot in eddington limits).
    structure : str
        Type of vertical structure. Can be 'Kramers', 'BellLin',
        'Mesa', 'MesaIdeal', 'MesaAd', 'MesaFirst' or 'MesaRadConv'.
    mu : double
        Molecular weight. Use in case of ideal gas EOS.
    abundance : dict or str
        Chemical composition of disc. Use in case of Mesa EOS.
        Format: {'isotope_name': abundance}. For example: {'h1': 0.7, 'he4': 0.3}.
        Use 'solar' str in case of solar composition.
    n : int
        Number of dots to calculate.
    tau_break : bool
        Whether to end calculation, when tau<1.
    savedots : bool
        Whether to save data in table.
    path_dots : str
        Where to save data table.

    """
    Sigma_plot, Teff_plot, F_plot = [], [], []
    varkappa_c_plot, T_c_plot, P_c_plot, rho_c_plot = [], [], [], []
    z0r_plot, tau_plot, PradPgas_Plot = [], [], []
    conv_param_z_plot, conv_param_sigma_plot = [], []
    free_e_plot = []
    r_plot = np.geomspace(r_start, r_end, n)
    tau_key = True
    tau_index = n
    for i, r in enumerate(r_plot):
        vs, F, Teff, Mdot = StructureChoice(M, alpha, r, Par, input, structure, mu, abundance)
        z0r, result = vs.fit()
        print('Teff = {:g}, tau = {:g}, z0r = {:g}'.format(Teff, vs.tau(), z0r))

        if vs.tau() < 1 and tau_key:
            tau_index = i
            tau_key = False
            if tau_break:
                print('Note: tau<1, tau_break=True. Cycle ends, when tau<1.')
                break

        varkappa_C, rho_C, T_C, P_C, Sigma0 = vs.parameters_C()
        Sigma_plot.append(Sigma0)
        varkappa_c_plot.append(varkappa_C)
        T_c_plot.append(T_C)
        rho_c_plot.append(rho_C)
        z0r_plot.append(z0r)
        tau_plot.append(vs.tau())
        P_c_plot.append(P_C)
        Teff_plot.append(Teff)
        F_plot.append(F)

        delta = (4 * sigmaSB) / (3 * c) * T_C ** 4 / P_C
        PradPgas_Plot.append(delta)

        rho, eos = vs.law_of_rho(P_C, T_C, full_output=True)
        try:
            _ = eos.grad_ad
            free_e = np.exp(eos.lnfree_e)
            free_e_plot.append(free_e)
            conv_param_z, conv_param_sigma = Convective_parameter(vs)
            conv_param_z_plot.append(conv_param_z)
            conv_param_sigma_plot.append(conv_param_sigma)
        except AttributeError:
            pass
        print(i + 1)

    if savedots:
        rg = 2 * G * M / c ** 2
        header_start = 'r \tr/rg \tSigma0 \tTeff \tF \tz0r \trho_c \tT_c \tP_c \ttau \tPradPgas \tvarkappa_c'
        header_end = '\nM = {:e} Msun, alpha = {}, Mdot = {} g/s, structure = {}'.format(
            M / M_sun, alpha, Mdot, structure)
        if structure in ['Kramers', 'BellLin', 'MesaIdeal']:
            header_end += ', mu = {}'.format(mu)
        else:
            header_end += ', abundance = {}'.format(abundance)
        dots_table = np.c_[r_plot[:tau_index], r_plot[:tau_index] / rg, Sigma_plot, Teff_plot, F_plot,
                           z0r_plot, rho_c_plot, T_c_plot, P_c_plot, tau_plot, PradPgas_Plot, varkappa_c_plot]
        if len(free_e_plot) != 0:
            header = header_start + ' \tfree_e \tconv_param_z \tconv_param_sigma' + \
                     '\nAll values a in CGS units.' + header_end
            dots_table = np.c_[dots_table, free_e_plot, conv_param_z_plot, conv_param_sigma_plot]
        else:
            header = header_start + '\nAll values a in CGS units.' + header_end
        np.savetxt(path_dots, dots_table, header=header)


def main():
    M = 1.5 * M_sun
    alpha = 0.2
    r = 1e10
    Teff = 1e4

    print('Calculation of vertical structure. Return structure table and plot.')
    print('M = {:g} M_sun \nr = {:g} cm \nalpha = {:g} \nTeff = {:g} K'.format(M / M_sun, r, alpha, Teff))

    Structure_Plot(M, alpha, r, Teff, input='Teff', mu=0.62, structure='BellLin', n=100, add_Pi_values=True,
                   savedots=True, path_dots='vs.dat', make_pic=True, save_plot=True, path_plot='vs.pdf',
                   set_title=True,
                   title=r'$M = {:g} \, M_{{\odot}}, r = {:g} \, {{\rm cm}}, \alpha = {:g}, T_{{\rm eff}} = {:g} \, '
                         r'{{\rm K}}$'.format(M / M_sun, r, alpha, Teff))
    print('Structure is calculated successfully. \n')

    print('Calculation of S-curve for Teff from 4e3 K to 1e4 K. Return S-curve table and Sigma0-Mdot plot.\n')

    S_curve(4e3, 1e4, M, alpha, r, input='Teff', structure='BellLin', mu=0.62, n=200, tau_break=False, savedots=True,
            path_dots='S-curve.dat', add_Pi_values=True, make_pic=True, output='Mdot',
            xscale='parlog', yscale='parlog', save_plot=True, path_plot='S-curve.pdf', set_title=True,
            title=r'$M = {:g} \, M_{{\odot}}, r = {:g} \, {{\rm cm}}, \alpha = {:g}$'.format(M / M_sun, r, alpha))
    print('S-curve is calculated successfully.')

    print('Calculation of radial structure of disc for radius from 3.1*rg to 1e3*rg and Mdot = Mdot_edd. '
          'Return radial structure table.\n')

    rg = 2 * G * M / c ** 2
    Radial_Plot(M, alpha, 3.1 * rg, 1e3 * rg, 1, input='Mdot_Mdot_edd', structure='BellLin', mu=0.62, n=200,
                tau_break=True, savedots=True, path_dots='radial_struct.dat')
    print('Radial structure is calculated successfully.')

    return


if __name__ == '__main__':
    main()
