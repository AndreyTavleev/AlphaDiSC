# DiscVerSt — accretion disc vertical structure calculation

This code calculates vertical structure of accretion discs around neutron stars and black holes.

## Contents

   * [Installation](#Installation)
      * [Only analytical opacities and EoS](#Only-analytical-opacities-and-EoS)
      * [Tabular opacities and EoS](#Tabular-opacities-and-EoS)
   * [Calculate structure](#Calculate-structure)
   * [Irradiated discs](#Irradiated-discs)
   * [Structure Choice](#Structure-Choice)
   * [Vertical and radial profile calculation, S-curves](#Vertical-and-radial-profile-calculation-S-curves)
   * [Physical background](#Physical-background)
   * [Calculation tips and tricks](#Calculation-tips-and-tricks)

## Installation

### Only analytical opacities and EoS

If you want to use only analytical formulas for opacity and EoS to calculate structures, choose this installation way.

1. Create and activate virtual environment

``` shell
$ python3 -m venv ~/.venv/vs
$ source ~/.venv/vs/bin/activate
```

2. Update pip and setuptools

``` shell
$ pip3 install -U pip setuptools
```

3. Install 'disc_verst' package

``` shell
$ pip3 install .
```

4. Run python and try to calculate simple structure

``` shell
$ python3 -m disc_verst.vs
```

	Finding Pi parameters of structure and making a structure plot. 
	Structure with Kramers opacity and ideal gas EoS.
	M = 1.98841e+34 grams
	r = 1.1813e+09 cm = 400 rg
	alpha = 0.01
	Mdot = 1e+17 g/s
	The vertical structure has been calculated successfully.
	Pi parameters = [7.10271534 0.4859551  1.13097882 0.3985615 ]
	z0/r =  0.028869666211114635
	Plot of structure is successfully saved to fig/vs.pdf.

### Tabular opacities and EoS

['mesa2py'](https://github.com/hombit/mesa2py) is used to bind tabular opacities and EoS 
from [MESA code](http://docs.mesastar.org) with Python3.
If you want to use tabular values of opacity and EoS to calculate the structure, you should use Docker.

You can use the latest pre-build Docker image:

``` shell
$ docker pull ghcr.io/andreytavleev/discverst:latest
$ docker tag ghcr.io/andreytavleev/discverst discverst
```

Or build a Docker image by yourself

``` shell
$ git clone https://github.com/AndreyTavleev/DiscVerSt.git
$ cd DiscVerSt
$ docker build -t discverst .
```

Then run 'discverst' image as a container and try `mesa_vs.main()`

``` shell
$ docker run -v$(pwd)/fig:/app/fig --rm -ti discverst python3 -m disc_verst.mesa_vs
```

	Calculating structure and making a structure plot.
	Structure with tabular MESA opacity and EoS.
    Chemical composition is solar.
	M = 1.98841e+34 grams
	r = 1.1813e+09 cm = 400 rg
	alpha = 0.01
	Mdot = 1e+17 g/s
	The vertical structure has been calculated successfully.
	z0/r =  0.029812574021917705
	Plot of structure is successfully saved to fig/vs_mesa.pdf.

As the result, the plot `vs_mesa.pdf` is saved to the `/app/fig/` (`/app/` is a Docker WORKDIR) 
in the container and to the `$(pwd)/fig/` in the host machine. 

## Calculate structure

Module `vs` contains several classes that represent the vertical 
structure of accretion discs in X-ray binaries for different analytical assumptions 
of opacity law. For given parameters the vertical structure of 
disc is calculated and can be used for research of disc stability.

Module `mesa_vs` contains some additional classes, that represent 
the vertical structure for tabular opacities and EoS and convective energy transport.

Both `vs` and `mesa_vs` modules have help with description of available structure classes.
``` python3
help(disc_verst.vs)
help(disc_verst.mesa_vs)
```

### Usage with analytical opacities and EoS::

``` python3
from disc_verst import vs

# Input parameters:
M = 2e33  # Mass of central object in grams
alpha = 0.01  # alpha parameter
r = 2e9  # radius in cm
F = 2e34  # viscous torque
mu = 0.6  # molecular weight

# Structure with (Bell & Lin, 1994) power-law opacities and ideal gas EoS
# with molecular weight mu and radiative temperature gradient.

vertstr = vs.IdealBellLin1994VerticalStructure(Mx=M, alpha=alpha, r=r, F=F, mu=mu)  # create the structure object
# the estimation of z0r free parameter is calculated automatically (default)
z0r, result = vertstr.fit(z0r_estimation=None)  # calculate structure
varkappa_C, rho_C, T_C, P_C, Sigma0 = vertstr.parameters_C()
print(z0r)  # semi-thickness z0/r of disc
print(varkappa_C, rho_C, T_C, P_C)  # Opacity, bulk density, temperature and gas pressure in the symmetry plane of disc
print(Sigma0)  # Surface density of disc
print(vertstr.tau())  # optical thickness of disc
```

### Usage with tabular opacities and EoS:

``` shell
$ docker run --rm -ti discverst python3
```

``` python3
from disc_verst import mesa_vs

# Input parameters:
M = 2e33  # Mass of central object in grams
alpha = 0.01  # alpha parameter
r = 2e9  # radius in cm
F = 2e34  # viscous torque

# Structure with tabular MESA opacities and EoS 
# and both radiative and convective energy transport,
# default chemical composition is solar.

vertstr = mesa_vs.MesaVerticalStructureRadConv(Mx=M, alpha=alpha, r=r, F=F)  # create the structure object
z0r, result = vertstr.fit()  # calculate structure
varkappa_C, rho_C, T_C, P_C, Sigma0 = vertstr.parameters_C()
print(z0r)  # semi-thickness z0/r of disc
print(varkappa_C, rho_C, T_C, P_C)  # Opacity, bulk density, temperature and gas pressure in the symmetry plane of disc
print(Sigma0)  # Surface density of disc
print(vertstr.tau())  # optical thickness of disc
```

You can run your own files, that use this code, inside Docker

``` shell
$ docker run -v/path_to/your_file/file.py:/app/file.py --rm -ti discverst python3 file.py
```

## Irradiated discs

Module `mesa_vs` also contains classes that represent 
the vertical structure of self-irradiated discs. Description of structure classes 
with irradiation is available in `mesa_vs` help.
``` python3
help(disc_verst.mesa_vs)
```

Irradiation can be taken into account in two ways:

1. Via either `T_irr` or `C_irr` parameters, that is, the irradiation temperature $T_{\rm irr}$ and irradiation 
   parameter $C_{\rm irr}$. It is a simple approach for irradiation, when the external flux doesn't penetrate 
   into the disc and only heats the disc surface. Relation between $T_{\rm irr}$ and $C_{\rm irr}$:
```math
\sigma_{\rm SB} T^4_{\rm irr} = C_{\rm irr} \frac{\eta \dot{M}c^2}{4\pi r^2},
```
   where $\eta=0.1$ is accretion efficiency.  

2. In the second approximation the external flux is penetrated into the disc and affect the energy flux
   and disc temperature. In this case there are more additional parameters are required, that describe
   the incident spectral flux:
```math
F^{\nu}_{\rm irr} = \frac{L_{\rm X}}{4\pi r^2} \, S(\nu).
```
   Such parameters are: frequency range `nu_irr`, units of frequency range 
   `spectrum_irr_par`, spectrum `spectrum_irr`, luminosity of irradiation source `L_X_irr` 
   and the cosine of incident angle `cos_theta_irr`.

   1. Frequency range `nu_irr` is array-like and can be either in Hz or in energy units (keV), this is determined by 
      `spectrum_irr_par` in `['nu', 'E_in_keV']`.
   2. Spectrum `spectrum_irr` can be either an array-like or a Python function. 
      1. If `spectrum_irr` is array-like, it must be in 1/Hz or in 1/keV depending on 'spectrum_irr_par',
         must be normalised to unity, and its size must be equal to `nu_irr.size`.
      2. If `spectrum_irr` is a Python function, the spectrum is calculated 
         for the frequency range `nu_irr` and automatically normalised to unity over `nu_irr`. 
         Note, that units of `nu_irr` and units of `spectrum_irr` arguments must be consistent. 
         There are two optional parameters `args_spectrum_irr` and `kwargs_spectrum_irr` 
         for arguments (keyword arguments) of spectrum function.
   3. Cosine of incident angle `cos_theta_irr` can be either exact value or `None`. In the latter case
      cosine is calculated self-consistently as `cos_theta_irr_exp * z0 / r`, where `cos_theta_irr_exp` is
      additional parameter, namely the $({\rm d}\ln z_0/{\rm d}\ln r - 1)$ derivative.

### Usage of simple irradiation scheme:

``` python3
from disc_verst import mesa_vs

# Input parameters:
M = 2e33  # Mass of central object in grams
alpha = 0.01  # alpha parameter
r = 2e9  # radius in cm
F = 2e34  # viscous torque
Tirr = 1.2e4  # irradiation temperature

# Structure with tabular MESA opacities and EoS,
# both radiative and convective energy transport,
# default chemical composition is solar, 
# external irradiation is taken into account
# through the simple scheme.

vertstr = mesa_vs.MesaVerticalStructureRadConvExternalIrradiationZeroAssumption(
                     Mx=M, alpha=alpha, r=r, F=F, T_irr=Tirr)  # create the structure object
z0r, result = vertstr.fit()  # calculate structure
varkappa_C, rho_C, T_C, P_C, Sigma0 = vertstr.parameters_C()
print(z0r)  # semi-thickness z0/r of disc
print(varkappa_C, rho_C, T_C, P_C)  # Opacity, bulk density, temperature and gas pressure in the symmetry plane of disc
print(Sigma0)  # Surface density of disc
print(vertstr.tau())  # optical thickness of disc
print(vertstr.C_irr)  # corresponding irradiation parameter
```

### Usage of advanced irradiation scheme:

Define the incident X-ray spectrum as a function of energy in keV:
```math
S_\nu \sim (E/E_0)^n \, \exp(-E/E_0)
```

``` python3
def power_law_exp_spectrum(E, n, scale):
   return (E / scale) ** n * np.exp(-E / scale)
```

Then calculate structure with this spectrum:

``` python3
from disc_verst import mesa_vs
import numpy as np

# Input parameters:
M = 2e33  # Mass of central object in grams
alpha = 0.01  # alpha parameter
r = 4e9  # radius in cm
F = 2e34  # viscous torque
nu_irr = np.geomspace(1, 10, 30)  # energy range from 1 to 10 keV
spectrum_par = 'E_in_keV'  # units of nu_irr if energy in keV
kwargs={'n': -0.4, 'scale': 8}  # spectrum function parameters
cos_theta_irr = None  # incident angle is calculated self-consistently
cos_theta_irr_exp = 1 / 12  # as cos_theta_irr_exp * z0 / r
L_X_irr = 1.0 * 1.25e38 * M / 2e33  #  luminosity of X-ray source = 1.0 * L_eddington

# Structure with tabular MESA opacities and EoS,
# both radiative and convective energy transport,
# default chemical composition is solar, 
# external irradiation is taken into account
# through the advanced scheme.

vertstr = mesa_vs.MesaVerticalStructureRadConvExternalIrradiation(
                     Mx=M, alpha=alpha, r=r, F=F, nu_irr=nu_irr, 
                     spectrum_irr=power_law_exp_spectrum, L_X_irr=L_X_irr,
                     spectrum_irr_par=spectrum_par, 
                     kwargs_spectrum_irr=kwargs,
                     cos_theta_irr=cos_theta_irr, 
                     cos_theta_irr_exp=cos_theta_irr_exp)  # create the structure object
# let us set the free parameters estimations
result = vertstr.fit(z0r_estimation=0.068, Sigma0_estimation=1032)  # calculate structure
# if structure is fitted successfully, cost function must be less than 1e-16
print(result.cost)  # cost function
z0r, Sigma0 = result.x  # Sigma0 is additional free parameter to find
varkappa_C, rho_C, T_C, P_C, Sigma0 = vertstr.parameters_C()
print(z0r)  # semi-thickness z0/r of disc
print(varkappa_C, rho_C, T_C, P_C)  # Opacity, bulk density, temperature and gas pressure in the symmetry plane of disc
print(Sigma0)  # Surface density of disc
print(vertstr.tau())  # optical thickness of disc
print(vertstr.C_irr, vertstr.T_irr)  # irradiation parameter and temperature
```

## Structure Choice
Module `profiles` contains `StructureChoice()` function, serves as interface for creating 
the right structure object in a simpler way. One can use other input parameters instead viscous torque `F`
(such as effective temperature $T_{\rm eff}$ and accretion rate $\dot{M}$) using `input` parameter,
and choose the structure type using `structure` parameter.
``` python3
from disc_verst.profiles import StructureChoice

# Input parameters:
M = 2e33  # Mass of central object in grams
alpha = 0.01  # alpha parameter
r = 4e9  # radius in cm
input = 'Teff'  # the input parameter will be effective temperature
Par = 1e4  # instead of viscous torque F
structure = 'BellLin'  # type of structure
mu = 0.6  # mean molecular weight for this type

# Returns the instance of chosen structure type.
# Also returns viscous torque F, effective temperature Teff and accretion rate Mdot.
vertstr, F, Teff, Mdot = StructureChoice(M=M, alpha=alpha, r=r, Par=Par, 
                                         input=input, structure=structure, mu=mu)
z0r, result = vertstr.fit()  # calculate the structure
```
The `StructureChoice` function has the detailed documentation
``` python3
from disc_verst import profiles

help(profiles.StructureChoice)
```

## Vertical and radial profile calculation, S-curves

Module `profiles` contains functions, that calculate vertical and radial disc profiles and S-curves, and return tables 
with disc parameters. With `profiles.main()` the vertical structure, S-curve and radial structure 
can be calculated for default parameters, stored as tables and plots:
``` shell
$ python3 -m disc_verst.profiles
```

`profiles` contains three functions: `Vertical_Profile`, `S_curve` and `Radial_Profile`. 

`Vertical_Profile` returns table with parameters of disc as functions of vertical coordinate at specific radius.

`S_curve` calculates S-curve and return table with disc parameters on the curve.

`Radial_Profile` returns table with parameters of disc as functions of radius for a given radius range.

### Usage:
``` python3
from disc_verst import profiles

# Input parameters:
M = 1.5 * 2e33  # 1.5 * M_sun
alpha = 0.2
r = 1e10
Teff = 1e4

profiles.Vertical_Profile(M, alpha, r, Teff, input='Teff', structure='BellLin', mu=0.62,
                          n=100, add_Pi_values=True, path_dots='vs.dat')

profiles.S_curve(4e3, 1e4, M, alpha, r, input='Teff', structure='BellLin', mu=0.62, 
                 n=200, tau_break=False, path_dots='S-curve.dat', add_Pi_values=True)

rg = 3e5 * (M / 2e33)  # Schwarzschild radius
profiles.Radial_Profile(M, alpha, 3.1 * rg, 1e3 * rg, 1, input='Mdot_Mdot_edd', structure='BellLin', mu=0.62, 
                        n=200, tau_break=True, path_dots='radial_struct.dat', add_Pi_values=True)
```
Both `profiles` module and functions in it have help
``` python3
from disc_verst import profiles

help(profiles)
help(profiles.Vertical_Profile)
help(profiles.S_curve)
help(profiles.Radial_Profile)
```

## Physical background
### Main equations
The vertical structure of accretion disc is described by system of four ordinary differential equations with four 
boundary conditions plus one additional boundary condition for flux:
```math
\begin{split}
\frac{{\rm d}P}{{\rm d}z} &= -\rho\,\omega^2_{K} z \qquad\quad\,\, P_{\rm gas}(z_0) = P'; \\
\frac{{\rm d}\Sigma}{{\rm d}z} &= -2\rho \qquad\qquad\quad \Sigma(z_0) = 0; \\
\frac{{\rm d} T}{{\rm d} z} &= \nabla \frac{T}{P} \frac{{\rm d} P}{{\rm d} z} \qquad\quad T(z_0) = T_{\rm eff} = (Q_0/\sigma_{\rm SB})^{1/4}; \\
\frac{{\rm d}Q}{{\rm d}z} &= \frac32\omega_K \alpha P \qquad\quad Q(z_0) = \frac{3}{8\pi}\dot{M}\omega_K^2 \left(1 - \sqrt{\frac{r_{\rm in}}{r}}\right) = \frac{3}{8\pi} \frac{F\omega_K}{r^2} = Q_0, \quad Q(0) = 0; \\
&z \in [z_0, 0].
\end{split}
```
Here $P = P_{\rm tot} = P_{\rm gas} + P_{\rm rad} = P_{\rm gas} + aT^4/3, \Sigma, T$ and $Q$ are total pressure, 
column density, temperature and energy flux in the disc, $\nabla\equiv\frac{{\rm d}\ln T}{{\rm d}\ln P}$ is the temperature 
gradient (radiative or convective, according to Schwarzschild criterion), and $\alpha$ is Shakura-Sunyaev turbulent 
parameter ([Shakura & Sunyaev
1973](https://ui.adsabs.harvard.edu/abs/1973A&A....24..337S)). After the normalizing $P_{\rm gas}, Q, T, \Sigma$ on their characteristic values $P_0, Q_0, T_0, \Sigma_{00}$, 
and replacing $z$ on $\hat{z} = 1 - z/z_0$ (in code it is the `t` variable), one has:
```math
\begin{split}
\frac{{\rm d}\hat{P}}{{\rm d}\hat{z}} &= \frac{z_0^2}{P_0}\,\omega^2_{\rm K} \,\rho (1-\hat{z}) - \frac{4aT_0^3}{3 P_0} \frac{{\rm d}\hat{T}}{{\rm d}\hat{z}} \qquad\qquad \hat{P}(0) = P'/P_0; \\ 
\frac{{\rm d}\hat{\Sigma}}{{\rm d}\hat{z}} &= 2\,\frac{z_0}{\Sigma_{00}}\,\rho \qquad\qquad\qquad\qquad\qquad\qquad\qquad  \hat{\Sigma}(0) = 0; \\
\frac{{\rm d}\hat{T}}{{\rm d}\hat{z}} &= \nabla \frac{\hat{T}}{P_{\rm tot}} z_0^2\,\omega^2_{\rm K} \,\rho (1-\hat{z}) \quad\qquad\qquad\qquad \hat{T}(0) = T_{\rm eff}/T_0; \\
\frac{{\rm d}\hat{Q}}{{\rm d}\hat{z}} &= -\frac32\,\frac{z_0}{Q_0}\,\omega_{\rm K} \alpha P_{\rm tot} \qquad\qquad\qquad \hat{Q}(0) = 1, \quad \hat{Q}(1) = 0; \\
P_{\rm tot} &= P_0\hat{P} + aT_0^4\hat{T}^4/3 \qquad\qquad\qquad\qquad\qquad \hat{z} \in [0, 1].
\end{split}
```

The initial boundary condition for gas pressure $P'$ is found through the integral:
```math
P_{\rm gas}(z_0) + P_{\rm rad}(z_0) = P' + P_{\rm rad}(z_0) = \int_0^{2/3} \frac{\omega_{\rm K}^2 z_0}{\varkappa_{\rm R}(P_{\rm gas}, T(\tau))}\, {\rm d}\tau.
```

Characteristic values of pressure, temperature and mass coordinate are as follows:
```math
T_0 = \frac{\mu}{\mathcal{R}} \omega_{\rm K}^2 z_0^2, \quad
P_0 = \frac{4}{3}  \frac{Q_0}{\alpha z_0 \omega_{\rm K}}, \quad
\Sigma_{00} = \frac{28}{3} \frac{Q_0}{\alpha z_0^2 \omega_{\rm K}^3}.
```

### External irradiation
In case of external irradiation $T_{\rm vis}\equiv T_{\rm eff}$. Irradiation can be taken into account in two ways:

(i). Via either irradiation temperature $T_{\rm irr}$ or irradiation parameter $C_{\rm irr}$: It is a simple 
   approach for irradiation, when the external flux doesn't penetrate into the disc and only heats the 
   disc surface. In this case only the boundary condition for temperature $\hat{T}$ changes:
```math
\hat{T}(0) = \frac1{T_0} \left(T_{\rm vis}^4 + T_{\rm irr}^4 \right)^{1/4}.
```

(ii). In the second approximation the external flux is penetrated into the disc and affect the energy flux
   and disc temperature. In this case following equations and boundary conditions change their view:
```math
\begin{split}
\frac{{\rm d}\hat{Q}}{{\rm d}\hat{z}} &= -\frac32\,\frac{z_0}{Q_0}\,\omega_{\rm K} \alpha P_{\rm tot} - \varepsilon\frac{z_0}{Q_0} \qquad\qquad \hat{Q}(0) = 1 + \frac{Q_{\rm irr}}{Q_0}; \\
\hat{T}(0) &= \frac1{T_0} \left(T_{\rm vis}^4 + Q_{\rm irr}/\sigma_{\rm SB} \right)^{1/4} \qquad\qquad\qquad\;\; \hat{\Sigma}(1) = \frac{\Sigma_0}{\Sigma_{00}}; \\
\end{split}
```
   where $\varepsilon, Q_{\rm irr}$ are additional heating rate and surface flux (see [Mescheryakov et al. 2011](https://ui.adsabs.harvard.edu/abs/2011AstL...37..311M})).
   
Due to unknown amtosphere model in case of irradiation the boundary condition $P'$ is found from the algebraic equation:
```math
P_{\rm gas}(z_0) + P_{\rm rad}(z_0) = P' + P_{\rm rad}(z_0) = \frac23\,\frac{\omega_{\rm K}^2 z_0}{\varkappa_{\rm R}(P', T(z_0))}.
```


### Equation of state and opacity law
Equation of state (EoS) and opacity law:
```math
    \rho = \rho(P, T), \quad \varkappa_{\rm R} = \varkappa_{\rm R}(\rho, T)
```
can be set both analytically or as tabular values. For analytical description, the ideal gas equation is adopted:
```math
    \rho = \frac{\mu\,P}{\mathcal{R}\,T}\,
```
where $\mu$ is mean molecular weight, it's an input parameter `mu` of Structure class. 

An analytic opacity coefficient is approximated by a power-law function:
```math
    \varkappa_{\rm R} = \varkappa_0 \rho^{\zeta} T^{\gamma}.
```
There are following analytic opacity options: 
1. Kramers law for solar composition: $(\zeta = 1, \gamma = -7/2, \varkappa_0 = 5\cdot10^{24})$ and Thomson electron 
   scattering $(\varkappa_{\rm R} = 0.34)$.
2. Analytic approximations by [Bell & Lin (1994)](http://adsabs.harvard.edu/abs/1994ApJ...427..987B) to opacity: 
   approximation obtained by the OPAL project $(\varkappa_0 = 1.5\cdot10^{20}, \zeta = 1, \gamma = -5/2)$, opacity 
   from scattering off hydrogen atoms $(\varkappa_0 = 1\cdot10^{-36}, \zeta = 1/3, \gamma = 10)$ and Thomson electron
   scattering $(\varkappa_{\rm R} = 0.34)$.

Tabular values of opacity and EoS are obtained by interpolation using `eos` and `kappa` modules from 
the [MESA code](http://mesa.sourceforge.net). In this case the additional input parameter is `abundance` - the chemical composition of the disc 
matter. It should be a dictionary with format {'isotope_name': abundance}, e.g. `{'h1': 0.7, 'he4': 0.3}`, look for full
list of available isotopes in the MESA source code. Also, you can use `'solar'` string to set the solar composition.


### Calculation
System has one free parameter $z_0$ - the semi-thickness of the disc, which is found using so-called shooting method. 
Code integrates system over $\hat{z}$ from 0 to 1 with initial approximation of free parameter $z_0$, then changes 
its value and integrates the system in order to fulfill the additional condition for flux $\hat{Q}(1)$ at the symmetry 
plane of the disc. 

In the presence of external irradiation in scheme (i), the only change is the boundary condition 
for temperature. If irradiation is calculated through the advanced scheme (ii), the second free parameter is $\Sigma_0$ - the 
surface density of the disc. In this case code integrates system with all changes above and solve two-parameter 
$(z_0, \Sigma_0)$ optimization problem in order to fulfill both $\hat{Q}(1)$ and $\hat{\Sigma}(1)$ additional boundary 
conditions. Namely, code minimises function:
```math
\begin{cases}
    f(z_0)= \hat{Q}(1) &\text{without irradiation / with irradiation scheme (i);} \\
    f(z_0, \Sigma_0)= \hat{Q}(1)^2 + \left( \frac{\hat{\Sigma}(1) \Sigma_{00}}{\Sigma_0} - 1\right)^2 &\text{with irradiation scheme (ii).}
\end{cases}
```


## Calculation tips and tricks
Code was texted for disc $T_{\rm eff}\sim (10^3-10^6) \rm K$. However, there can be some convergence problems, especially when $P_{\rm rad}/P_{\rm gas}\gg0.1$.

### Without irradiation
Calculation can be failed, if during the fitting process $P_{\rm gas}$ become less than zero, the corresponding `PgasPradNotConvergeError` is raised. In this case one recommend to set manually the estimation of 'z0r' free parameter (usually smaller estimation). Also you can get additional information about the value of free parameter during the fitting process through `verbose` parameter:
``` python3
verstr = ...  # definition the structure class
# let us set the free parameter estimation
# and print value of z0r parameter at each fitting iteration
z0r, result = vertstr.fit(z0r_estimation=0.05, verbose=True)
```
Note, that the higher $P_{\rm rad}/P_{\rm gas}$ value the more sensitive calculation convergence to `z0r` estimation.

### With irradiation scheme (i)
Calculation can be failed, if during the fitting process $P_{\rm gas}$ become less than zero, the corresponding `PgasPradNotConvergeError` is raised. In this case again try to set manually the estimation of 'z0r' free parameter.

Another reason of calculation failure concerns the calculation of $P'$ pressure initial condition. In contrast to the 'no-irradiation' case, value of $P'$ is found as the root of algebraic equation. If the default initial estimation of this root is poor, the root finding can be failed (usually it means that during the root finding the $P_{\rm gas}$ become less than zero), the corresponding `PphNotConvergeError` is raised. Then you can try to set that estimation manually (usually higher) through `P_ph_0` parameter:
``` python3
verstr = ...  # definition the structure class
# let us set the free parameter estimation
# and set the estimation of pressure boundary condition
# and print value of z0r parameter at each fitting iteration
result = vertstr.fit(z0r_estimation=0.05, verbose=True, P_ph_0=1e5)
```

### With irradiation scheme (ii)
This case if similar to one for scheme (i), and the ways to solve the convergence problem are almost the same. The main difference is the presence of second free parameter of system - `Sigma0_par`, the surface density $\Sigma_0$ of the disc. The additional convergence problem may occur during the finding for $Q_{\rm irr}$ and $\varepsilon$ irradiation terms. In this case the corresponding `IrrNotConvergeError` is raised. One recommends to change the estimations of free parameters (usually smaller `z0r` and higher `Sigma0_par`) and, perhaps, the `P_ph_0` parameter. 

One specific case is unstable disc region, where code in principle may converge to both hot and cold disc state. Here, even for right `z0r` and `Sigma0_par` values, the wrong $P'$ root can be found due to wrong root estimation, which leads to code non-convergence. Then one recommends to set another `P_ph_0` estimation (much higher or smaller).
``` python3
verstr = ...  # definition the structure class
# let us set the free parameters estimations
# and set the estimation of pressure boundary condition
# and print value of free parameters at each fitting iteration
result = vertstr.fit(z0r_estimation=0.07, Sigma0_estimation=5000, verbose=True, P_ph_0=1e5)
```

### Radial profile and S-curves
Radial profile calculates from `r_start` to `r_end`, and `r_start` should be less than `r_end`. Similarly, S-curve calculates from `Par_max` to `Par_min` (obviously, `Par_max` > `Par_min`). The estimations of all necessary parameters at the next point are taken from the previous points. Additionally, estimations of the parameters at the first point can be set manually. Usually this works well, but regions of small radii (big effective temperature, or accretion rate etc. for S-curve) may not converge due to high $P_{\rm rad}/P_{\rm gas}$. These points of corresponding profiles will be missed and marked as 'Non-converged_fits' in the code output (in output tables there wll be only the number of such 'Non-converged_fits'). 

In this case one can to calculate such 'bad' regions again, but vise versa from higher radius (effective temperature etc.), where code has converged, to smaller one. Then the estimations of free parameters may become more suitable for convergence. Such vise versa calculation is easy to do, just swap `r_start` and `r_end` (`Par_max` and `Par_min`) so that `r_start` > `r_end` (`Par_max` < `Par_min`).

In case of calculation the profile and S-curve of disc with irradiation scheme (ii) there can be a discontinuity (gap) in the instability region, i.e. no smooth transition between 'hot' and 'cold' stable disc regions. This manifests itself in the fact that after the 'hot' stable disc region the code stops converging, and all corresponding points will be marked as 'Non-converged_fits'. Then just calculate the 'cold' region with another `z0r` and `Sigma0_par` (and, perhaps, `P_ph_0`) start estimations. 
