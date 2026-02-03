import numpy as np
from astropy import constants as const
# from scipy.optimize import fsolve
# from scipy import optimize
# import scipy.integrate as integrate
# from scipy.integrate import integrate.simpson
# from scipy.misc import derivative
from scipy.optimize import brentq
from numpy import sin, cos, pi
import inspect

# import matplotlib.pyplot as plt
# import pandas as pd
# import time
# from joblib import Parallel, delayed
# import multiprocessing
# from scipy.interpolate import splev, splrep
# from utils import unpack_params, KNOWN_NAMES, default_params
# from opacity import Opac

#np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 
#np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 
G = const.G.cgs.value
R_GAS = const.R.cgs.value
K_BOLTZ = const.k_B.cgs.value
H_PLANCK = const.h.cgs.value
C_LIGHT = const.c.cgs.value
SIGMA_STEFBOLTZ = const.sigma_sb.cgs.value
R_SOLAR = const.R_sun.cgs.value
M_SOLAR = const.M_sun.cgs.value
# importopac = True

# KAPPA_TH = 0.335 #томпсоновская непрозрачность
# KAPPA_0_FREEFREE = 5e24

ALLOWED_R0_MODELS = ['alfven', 'kuzinlipunova', 'kluzniakrappaport07_i',
                  'kluzniakrappaport07_ii', 'wang97', 'standard']
ALLOWED_R_KEYWORDS = ['rlight', 'rcor']


def _call_with_filtered_dict(func, data_dict, explicit_args=None):
    sig = inspect.signature(func).parameters.keys()
    if explicit_args is not None:
        combined_args = data_dict | explicit_args
    else:
        combined_args = data_dict
    final_kwargs = {k: v for k, v in combined_args.items() if k in sig}
    return func(**final_kwargs)
    

def fn(dist0, dist, n = 1):
    return 1 - (dist0 / dist)**(n/2)

def r_cor(Mx, freq):
    """ Corotation radius."""
    GM = G * Mx
    return np.cbrt(GM / 4 / pi**2 / freq**2)

def r_alfven(Mx, mu_magn, Mdot):
    """Alfven radius (without any numerical coefs)"""
    GM = G * Mx
    return (mu_magn**4 / Mdot**2 / GM)**(1./7.)

def r_isco(Mx):
    """innermost stable circular orbit radius"""
    return 6 * G * Mx / C_LIGHT**2
        
def r_light(freq):
    """ Light cylinder radius"""
    return C_LIGHT / 2 / pi / freq

def f_p(eta, delta0_in, chi_deg):
    """F_P function in Kuzin Lipunova model"""
    chi = np.deg2rad(chi_deg)
    aux_= (1 - eta)**2 + pi * np.sqrt(delta0_in*0.5) * eta * (1 - eta)
    return cos(chi)**2 * aux_ + pi**2/16 * delta0_in * sin(chi)**2

def _require_not_none(**kwargs):
    missing = [k for k, v in kwargs.items() if v is None]
    if missing:
        raise ValueError(f"Required params must not be None: {', '.join(missing)}")


def radius_inner(Mx, model, mu_magn=None, Mdot=None,r_star=None, chi_deg=None, freq=None,
                 r_lower_limit=None, r_upper_limit=None,
                 ra_coef=None, eta=None, kappa=None, delta0_in=None,
                 ):
    """
    Compute the inner disk radius r0 for a magnetized central pbject under several
    literature prescriptions, with optional limiting by lower/upper bounds.
    
    This routine evaluates one of the supported models for the truncation radius
    (Alfvenic or torque-balance based), solves the corresponding algebraic equation
    for the dimensionless fastness parameter w when needed, and returns
    r0 = r_c * w**(2/3) or r0 = r_Alfven as appropriate, then enforces physically
    motivated limits (ISCO and stellar surface) and any user-provided bounds.
    
    Parameters
    ----------
    mu_magn : float
        Magnetic dipole moment of the star [G cm^3].
    Mx : float
        Stellar Mx [g].
    Mdot : float
        Mx accretion rate [g s^-1].
    model : {'Alfven', 'KuzinLipunova', 'KluzniakRappaport07_I',
             'KluzniakRappaport07_II', 'Wang97', 'Standard'}
        Choice of inner-radius prescription:
        - 'Standard': maximum of r_star and R_ISCO. 
        - 'Alfven' : r0 = ra_coef * r_A (Alfven radius defined without any numerical
                               coefficients, multiplied by ra_coef).
        - 'kuzinlipunova' : Kuzin & Lipunova torque-balance form.
        - 'KluzniakRappaport07_I' : Kluźniak & Rappaport (2007) case I 
            (ArXiv 0709.2361, eq. 17).
        - 'KluzniakRappaport07_II' : Kluźniak & Rappaport (2007) case II
            (ArXiv 0709.2361, eq. B7).
        - 'wang97' : Wang (1997) prescription 
            (https://ui.adsabs.harvard.edu/abs/1997ApJ...475L.135W/abstract, 
             eq. (7) with \Gamma === 1)
    r_star : float or None, optional
        Stellar radius (cm). Used as a hard lower limit together with ISCO. 
        If None, set to R_ISCO.
    chi_deg : float, optional
        Angle chi between the magnetic dipole axis and the disc symmetry axis,
        in degrees. Default is 0.
    freq : float, optional
        Stellar spin frequency (Hz). Used for the corotation radius and light cylinder.
        Default is 1.0.
    r_lower_limit : {float, 'rcor', 'rlight', None}, optional
        Additional lower bound to apply to r0. If a string:
        - 'rcor', use r_c (corotation radius).
        - 'rlight', use r_L (light-cylinder radius).
        If None, no extra lower bound is applied. Default is None.
    r_upper_limit : {float, 'rcor', 'rlight', None}, optional
        Additional upper bound to apply to r0. Same string semantics as
        r_lower_limit. If None, no extra upper bound is applied. Default is None.
    ra_coef : float, optional
        Coefficient for the Alfven radius. Default is 1.0.
    eta : float, optional
        Coupling/efficiency/penetration factor appearing in Kuzin-Lipunova and
        Wang models. Default is 1.0.
    kappa : float, optional
        Parameter entering the Kuzin–Lipunova forms. Default is 0.5.
    delta0_in : float, optional
        Boundary layer thickness parameter (dimensionless) used in the models
        of Kuzin-Lipunova and Wang97.
        Default is 0.01.
    
    Returns
    -------
    float
        The inner radius r0 in cm, after applying:
        (1) the model formula,
        (2) the hard lower limit max(r_star, r_ISCO),
        (3) the optional r_lower_limit and r_upper_limit if provided.
    
      
    """
    
    if model.lower() not in ALLOWED_R0_MODELS:
        raise ValueError(f"model of the inner radius should be one of: {ALLOWED_R0_MODELS}.")
        
    risco = r_isco(Mx=Mx)
    if r_star is None:
        r_star =  risco
    rmax = max(r_star, risco)
    if model.lower() == 'standard':
        return rmax
    _require_not_none( mu_magn=mu_magn, Mdot=Mdot, r_star=r_star)
    _ra = r_alfven(Mdot=Mdot, Mx=Mx, mu_magn=mu_magn)
    if model.lower() != 'alfven':
        _require_not_none(freq=freq)
        _rc = r_cor(Mx=Mx, freq=freq)
        _ra_rc = _ra / _rc # the same as \xi in Kluzniak, Rappaport
            
    if model.lower() == 'alfven':
        _require_not_none(ra_coef=ra_coef)
        _r0 = _ra * ra_coef
        
    if model.lower() == 'kuzinlipunova':
        _require_not_none(eta=eta, chi_deg=chi_deg, delta0_in=delta0_in,
                          kappa=kappa)
        _chi = np.deg2rad(chi_deg)
        _s2 = sin(_chi)**2
        _c2 = 1 - _s2
        _f_p = f_p(eta=eta, delta0_in=delta0_in, chi_deg=chi_deg)

        ### special case eta=0
        if eta < 1e-3:
            w = _ra_rc**1.5 * (2 * kappa * _f_p)**(3/7)
        else:
            one = lambda _w: (1-_w) * eta**2 
            two = lambda _w: (1-_w) * eta**2 * (1 - 3 * delta0_in) * _s2
            three = lambda _w: kappa * _f_p
            to_solve = lambda _w: 0.5 -  _ra_rc**3.5 * _w**(-7/3) * (one(_w) + two(_w) + three(_w)) 
            try:
                w = brentq(to_solve, 1e-10, 10000)
            except:
                # print("Equation for r0 was not solved, returning rmax...")
                w = 0
        _r0 = _rc * w**(2/3)

    if model.lower() == 'kluzniakrappaport07_i':
        to_solve = lambda _w: 0.5 - _ra_rc**3.5 * _w**(-10/3) * (1 - _w)
        try:
            w = brentq(to_solve, 1e-10, 10000)
        except:
            # print("Equation for r0 was not solved, returning rmax...")
            w = 0
        _r0 = _rc * w**(2/3)
        
    if model.lower() == 'kluzniakrappaport07_ii':
        to_solve = lambda _w: 0.5 - _ra_rc**3.5 * _w**(-7/3) * (1 - _w)
        try:
            w = brentq(to_solve, 1e-10, 10000)
        except:
            # print("Equation for r0 was not solved, returning rmax...")
            w = 0
        _r0 = _rc * w**(2/3)
        
    if model.lower() == 'wang97':
        _require_not_none(eta=eta, chi_deg=chi_deg,
                          delta0_in=delta0_in)
        _chi = np.deg2rad(chi_deg)
        _s2 = sin(_chi)**2
        _c2 = 1 - _s2
        one = lambda _w: (1 - _w) * _c2
        two = lambda _w: (8 - 5 * _w) * delta0_in * _s2
        
        to_solve = lambda _w: 0.5 - eta**2 * _ra_rc**3.5 * _w**(-7/3) * (one(_w) + two(_w))
        try:
            w = brentq(to_solve, 1e-10, 10000)
        except:
            # print("Equation for r0 was not solved, returning rmax...")
            w = 0
        _r0 = _rc * w**(2/3)
        
    _r0 = np.maximum(rmax, _r0)
    
    if r_lower_limit is not None:
        if isinstance(r_lower_limit, str) and r_lower_limit not in ALLOWED_R_KEYWORDS:
            raise ValueError(f"If r_lower_limit is a string, it should be one of {ALLOWED_R_KEYWORDS}")
        if isinstance(r_lower_limit, str) and r_lower_limit == 'rcor':
            r_lower_limit = _rc
        if isinstance(r_lower_limit, str) and r_lower_limit == 'rlight':
            r_lower_limit = r_light(freq=freq)
        _r0 = np.maximum(r_lower_limit, _r0)
        
    if r_upper_limit is not None:
        if isinstance(r_upper_limit, str) and r_upper_limit not in ALLOWED_R_KEYWORDS:
            raise ValueError(f"If r_upper_limit is a string, it should be one of {ALLOWED_R_KEYWORDS}")
        if isinstance(r_upper_limit, str) and r_upper_limit == 'rcor':
            r_upper_limit = _rc
        if isinstance(r_upper_limit, str) and r_upper_limit == 'rlight':
            r_upper_limit = r_light(freq=freq)
        _r0 = np.minimum(r_upper_limit, _r0)
        
    return _r0

def magnetic_torque(r, mu_magn=None, Mx=None, r0=None, model=None, freq=None,
           r_out_magn=None, eta=None, kappa_prime=None,
           delta0_in=None, chi_deg=None, kappa_alfven=None):
    """
    Compute the magnetic torque (or magnetic stress term) as a function of radius
    for several commonly used magnetosphere–disk coupling prescriptions.
    
    The function returns a radial profile :math:`F_\\mathrm{magn}(r)` evaluated for
    the chosen `model`. For the `'standard'` model the magnetic torque is set to
    zero. For all other models the input radius (or radii) must satisfy
    r >= r_0.
    
    Parameters
    ----------
    r : float or array_like
        Radius (or radii) at which to evaluate the magnetic torque. Must be
        greater than or equal to `r0`. Scalars return a scalar, arrays return an
        array of matching shape.
    mu_magn : float
        Magnetic dipole moment of the central object.
    Mx : float
        Mass of the central object (passed to the corotation-radius helper
        `r_cor` when needed).
    r0 : float
        The inner disk radius.
    model : {'Standard', 'Alfven', 'KuzinLipunova', 'KluzniakRappaport07_I', \
    'KluzniakRappaport07_II', 'Wang97'}
        Choice of magnetic-torque prescription. Matching is case-insensitive.
    
        - 'standard': returns identically zero torque.
        - 'alfven': constant torque level set by `kappa_alfven`.
        - 'kuzinlipunova': Kuzin & Lipunova-type expression using
          `eta`, `kappa_prime`, `delta0_in`, and `chi_deg`.
        - 'kluzniakrappaport07_i': Kluzniak & Rappaport (2007), case I.
        (ArXiv 0709.2361, eq. 24).
        - 'kluzniakrappaport07_ii': Kluzniak & Rappaport (2007), case II
          (piecewise below/above corotation). (ArXiv 0709.2361, eq. B5, B6).
        - 'wang97': Wang (1997)-type expression using `eta`, `delta0_in`,
          and `chi_deg`.
          (https://ui.adsabs.harvard.edu/abs/1997ApJ...475L.135W/abstract, 
           eqs. 8, 9.)
    freq : float, optional
        Spin frequency of the central object. Required for all models except
        'standard' and 'alfven'. Used to compute the corotation radius
        and the fastness parameter.
    r_out_magn : None or float or {'rcor', 'rlight'}, optional
        Optional outer radius for the magnetic interaction region.
        If provided, the torque is evaluated with an effective distance
        dist_ = min(r, r_out_magn). If a string keyword is given:
    
        - 'rcor': uses the corotation radius computed from `freq`.
        - 'rlight': uses the light-cylinder radius computed from `freq`.
    
        If not provided, ``dist_ = r``.
    eta : float, optional
        Dimensionless parameter used in some prescriptions (
        ``'kuzinlipunova'``, ``'wang97'``). Required for those models.
    kappa_prime : float, optional
        Dimensionless coupling parameter used in the 'kuzinlipunova' model.
    delta0_in : float, optional
        The effective z0/r at the inner edge f=of the disc.
        Required for 'kuzinlipunova' and 'wang97'.
    chi_deg : float, optional
        Magnetic obliquity angle in degrees. Required for models that include an
        inclination dependence ('kuzinlipunova' and 'wang97').
    kappa_alfven : float, optional
        Dimensionless normalization for the 'alfven' model. Required if
        model='alfven'.
    
    Returns
    -------
    F_magn : float or numpy.ndarray
        Magnetic torque (or magnetic stress term) evaluated at `r`. If `r` is a
        scalar, a Python float is returned. If `r` is array-like, a NumPy array
        with the same shape as `r` is returned.
    
    """
    if model.lower() not in ALLOWED_R0_MODELS:
        raise ValueError(f"model of the magnetic torque should be one of: {ALLOWED_R0_MODELS}.")
    if model.lower() == 'standard':
        return 0 * r
    _require_not_none(Mx=Mx, mu_magn=mu_magn, r0=r0, model=model)
    r = np.asarray(r)
    if np.ndim(r) == 0:
        if r < r0:
            raise ValueError(f"r = {r} in F_magn should be > r0 = {r0}.")
    else:
        if any([r_i < r0 for r_i in r]):
            raise ValueError("all r in F_magn should be > r0.")
        
    if model.lower() != 'alfven' or isinstance(r_out_magn, str):
        _require_not_none(freq=freq)
        _rc = r_cor(Mx=Mx, freq=freq)
        w_fastness = (r0 / _rc)**1.5
    
    dist_ = r
    
    if r_out_magn is not None:
        if isinstance(r_out_magn, str) and r_out_magn not in ALLOWED_R_KEYWORDS:
            raise ValueError(f"If r_out_magn is a string, it should be one of: {ALLOWED_R_KEYWORDS}")
        if isinstance(r_out_magn, str) and r_out_magn == 'rcor':
            r_out_magn = _rc
        if isinstance(r_out_magn, str) and r_out_magn == 'rlight':
            r_out_magn = r_light(freq=freq)
        # print('r out magn = ', r_out_magn)
        dist_ = np.minimum(r, r_out_magn)   
    

    if model.lower() == 'alfven':
        # print('alf')
        _require_not_none(kappa_alfven = kappa_alfven)
        _F_magn = kappa_alfven * mu_magn**2 / r0**3 * np.ones(r.size)
        
    if model.lower() == 'kuzinlipunova':
        # print('kl')
        _require_not_none(eta=eta, kappa_prime=kappa_prime, chi=chi_deg, 
                          delta0_in=delta0_in)
        chi = np.deg2rad(chi_deg)
        f3_ = fn(r0, dist_, 3)
        f6_ = fn(r0, dist_, 6)
        _g_func = (eta**2 / 3 * (2 * w_fastness * f3_ - f6_) * cos(chi)**2 +
                  2 * kappa_prime * f_p(eta=eta, delta0_in=delta0_in, chi_deg=chi_deg))
    
        _F_magn = mu_magn**2 / r0**3 * _g_func
        
    if model.lower() == 'kluzniakrappaport07_i':
        _F_magn = mu_magn**2/9 * (3 / r0**3 
                                  - 3 / dist_**3 
                                  - 2 * _rc**1.5 / r0**4.5
                                  + 2 * _rc**1.5 / dist_**4.5)
    
    if model.lower() == 'kluzniakrappaport07_ii':
        def _F_magn_under_cor(_r):
            return  mu_magn**2 / 3 / _r**3 * (2 * (_r / _rc)**1.5 * (
                (_r / r0)**1.5 - 1) 
                + 1
                - (_r / r0)**3)
        def _F_magn_over_cor(_r):
            return mu_magn**2 / 9 / _r**3 * ( - 2 * (_r / _rc)**3 
                                             + 2 * (_rc / _r)**1.5 
                                             + 6 * (_r**2 / r0 / _rc)**1.5
                                             - 3 * (_r / r0)**3
                                             - 3
                )
        _F_magn = np.where(dist_ < _rc, _F_magn_under_cor(dist_), _F_magn_over_cor(dist_))
        
    if model.lower() == 'wang97':
        _require_not_none(chi_deg=chi_deg, eta=eta, delta0_in=delta0_in)
        _s2 = sin(np.deg2rad(chi_deg))**2
        _F_magn = mu_magn**2 / r0**3 * ( (1 - _s2) * 1/3 * (
            2 * w_fastness * fn(r0, dist_, 3) - fn(r0, dist_, 6))
            + 2 * _s2 * delta0_in * (w_fastness - 1) 
            )
    
    return _F_magn.item() if np.ndim(r)==0 else _F_magn

        
class MagneticField: #!!!
    """
        A mixin for the magnetic field treatment in the vertical structure 
        calculation.
        
        Takes the dictionary `magn_args` as an argument. If it's None or if
        it contains an entry ['model'] == 'standard', the behavior falls to a
        fully non-magnetic case. Essentially, the behavior is non-magnetic 
        in all cases except model == 'KuzinLipunova'. The dictionary, if provided,
        should always contain 'model' and 'r_star'.
        
        It overwrites the two BaseVerticalStructure functions:
            - additional_photospheric_pressure() --- for the aditional
                pressure the discontinuous magnetic field can exert on a
                photosphere. Set == 0 in all cases except model == 'KuzinLipunova'.
                
            - dP_magn_induced_dz --- for the additional source in the equation
            of vertical hydtrostatics. It should be a d/dz(b_phi^2/8pi), 
            where b_phi is a phi-component of the induced magnetic field inside
            the disc. In case model=='KuzinLipunova', set using the approximation
            b_phi(r, z) = b_phi(r, z=z0) * z/z0. 
            
        Inside, it calls the ViscousTorque class to calculate the inner radius
        of the disk given F. I guess this should be fixed in future when we have the 
        AccretionDisc class containing all info...
            
    """
    def __init__(self, *args, **kwargs):
        magn_args = kwargs.pop("magn_args", None)
        super().__init__(*args, **kwargs)
        
        if magn_args is None:
            self.model = 'standard'
            magn_args = {'model': 'standard'}
        else:
            if magn_args['model'].lower() not in ALLOWED_R0_MODELS:
                raise ValueError(f"""The model in magn_args in the class MagneticField
                                 \nshould be one of:{ALLOWED_R0_MODELS}.""")
            self.model = magn_args['model']
            self.r_star = magn_args['r_star']
        self.magn_args = magn_args
            
        if self.model != 'standard':
            self._set_magn_pars(magn_args=magn_args)
            
    def _set_magn_pars(self, magn_args):
        self.mu_magn = magn_args['mu_magn']
        self.model = magn_args['model']
        
        if self.model.lower() == 'alfven':
            self.ra_coef = magn_args['ra_coef']
        if self.model.lower() == 'kuzinlipunova':            
            self.freq = magn_args['freq']
            self.chi_deg = magn_args['chi_deg']
            self.chi = np.deg2rad(self.chi_deg)
            self.eta = magn_args['eta']
            self.kappa = magn_args['kappa']
            self.kappa_prime = magn_args['kappa_prime']
            self.delta0_in = magn_args['delta0_in']

        
                        
    def _f_p(self):
        return f_p(eta=self.eta, delta0_in=self.delta0_in, chi_deg=self.chi_deg)
    
    def _r_out_magn(self, Mx):
        """An analogue of r_out_magn, but since it depends on Mx which is 
        not known at the init time of this mixin, I made it a funciton of Mx."""
        _r_out = self.magn_args['r_out_magn']
        if _r_out is not None:
            if isinstance(_r_out, str) and _r_out not in ALLOWED_R_KEYWORDS:
                raise ValueError(f"If _r_out is a string, it should be one of {ALLOWED_R_KEYWORDS}")
            if isinstance(_r_out, str) and _r_out == 'rcor':
                _r_out = r_cor(Mx=Mx, freq=self.freq)
            if isinstance(_r_out, str) and _r_out == 'rlight':
                _r_out = r_light(freq=self.freq)
        return _r_out
    
    @property
    def _r0(self):
        _Torques = ViscousTorque(Mx=self.Mx, magn_args=self.magn_args)
        _Mdot = _Torques.Mdot_from_F(F=self.F, r=self.r)
        return _Torques.r0(_Mdot)
    
    def additional_photospheric_pressure(self):
        if self.model.lower() == 'kuzinlipunova':
            _s2 = sin(self.chi)**2 
            _c2 = 1 - _s2 # cos^2(chi)
            D = max(( (2 * self.delta0_in)**0.5, 
                     (self.r**2 / self._r0**2 - 1)**0.5 )) # Lai1999
            P_aly = self.mu_magn ** 2 / self.r ** 6 / 8 / pi * (
                2.5 * _s2 + (4 / pi / D *(1 - self.eta)) ** 2 * _c2
                )
            return P_aly
        else:
            return 0
        
    def _b_induced_surface(self, _r):
        _r = np.asarray(_r)
        if self.model.lower() == 'kuzinlipunova':
            _rc = r_cor(Mx=self.Mx, freq=self.freq)
            R = np.where(
                (_r < self._r_out_magn(Mx=self.Mx)),
                (self.mu_magn / _r**3) * (1.0 - (_r / _rc)**1.5),
                0.0
            ) * self.eta * cos(self.chi)
            return R.item() if np.isscalar(_r) else R
        else:
            return _r * 0
        
    def dP_magn_induced_dz(self, z):
        if self.model.lower() == 'kuzinlipunova':
            ### uses approximation: b(r, z) = b_surface(r) * z/z0,
            ### so d(b^2)/dz = b_surface^2 * 2z/z0^2
            db2_dz = self._b_induced_surface(_r=self.r)**2 * 2 * z / self.z0**2
            return db2_dz / 8 / pi
        else:
            return z*0
        
    
class ViscousTorque: #!!!
    """
    A class sumarizing the information about the visocus and magnetic torques
    distribution over radius, all that can be found prior to soling for the 
    vertical structure. All quantities are in CGS.
    
    All methods are vectorized over r and not vectorized over Mdot. 
    
    Attributes
    ----------
    Mx : double
        Mass of the central object.
    F_in : double
        Inner torque. Default 0. Used only if it was impossible to solve for
        Mdot given F or T_eff, so the simplified standard disc expressions are
        used.        
    GM : double
        Yeah.
    rg : double
        Grav radius 2GM/c^2.
    magn_args : dict or None
        A dictionary of arguments for finding the inenr radius and the magnetic
        distributed torque. 
        
    Methods
    -------
    r0(Mdot):
        Finds the inner radius of the disc by calling the function radius_inner.
        See the documentation for this function. 
    F_standard(Mdot, r):
        The Shalura-Sunyaev viscous torque: Mdot sqrt(GMr) (1 - sqrt(r0/r)).
    F_magn(Mdot, r):
        The magnetic torque. The function magnetic_torque if called, see its
        documentation.
    F_vis(Mdot, r):
        The total torque, the sum of F_standard, F_magn, and F_in.
    Q_vis(Mdot, r):
        Viscous heat flux from the unit of one surface. 
        Q_vis = 3/4 W_rphi w_kepl.
    T_eff(Mdot, r):
        The effective temperature of a disc.
            sigma_sb T_eff^4 = Q_vis.
    Mdot_from_F(F, r):
        Finds such an Mdot as to give the F at the radius r. Tries to solve the
        equation for Mdot, and it fails, returns the analytical standard expression.
    Mdot_from_Teff(Teff, r):
        Finds such an Mdot as to give the Teff at the radius r. Tries to solve the
        equation for Mdot, and it fails, returns the analytical standard expression.
    
    """
    def __init__(self, Mx, F_in=0, magn_args=None):
        self.Mx = Mx
        self.GM = G * Mx
        self.rg = 2 * self.GM / C_LIGHT**2
        self.F_in = F_in
        if magn_args is None:
            self.model = 'standard'
            magn_args = {'model': 'standard'}
        else:
            self.model = magn_args['model']
            if self.model.lower() not in ALLOWED_R0_MODELS:
                raise ValueError(f"""The model in magn_args in the class ViscousTorque
                                 \nshould be one of: {ALLOWED_R0_MODELS}.""")
        self.magn_args = magn_args

    def r0(self, Mdot):
        return _call_with_filtered_dict(func=radius_inner, 
                explicit_args = {'Mx': self.Mx, 'Mdot':Mdot}, 
                data_dict=self.magn_args)
        
    def _fn(self, Mdot, r, n=1):
        # return 1. - (self.r0(Mdot) / r)**(n/2)
        return fn(dist0 = self.r0(Mdot), dist=r, n=n)
    
    def _w_kepl(self, r):
        return np.sqrt(self.GM / r**3)
    
    def _h(self, r):
        return np.sqrt(self.GM * r)
    
    def F_standard(self, Mdot, r):
        return self._h(r) * self._fn(r=r, Mdot=Mdot, n=1) * Mdot
    
    def F_magn(self, Mdot, r):
        return _call_with_filtered_dict(func=magnetic_torque,
                explicit_args = {'r':r,'Mx':self.Mx, 'r0':self.r0(Mdot)}, 
                data_dict=self.magn_args
                )
        
    def F_vis(self, Mdot, r):
        return self.F_standard(Mdot, r) + self.F_magn(Mdot, r) + self.F_in
    
    def Q_vis(self, Mdot, r):
        ### 3/4 W_rphi w_kepl = 3/4 (F / 2 pi r^2) w_kepl
        return 0.75 * (self.F_vis(Mdot, r) / 2 / pi / r**2) * self._w_kepl(r)
    
    def T_eff(self, Mdot, r):
        ### No irradiation...
        return (self.Q_vis(Mdot, r) / SIGMA_STEFBOLTZ)**(1 / 4)
    
    def Mdot_from_F(self, F, r):
        try:
            return brentq(f = lambda _mdot: F - self.F_standard(_mdot, r),
                          a=1.0, b=1e22)
        except ValueError:
            print("Could not solve for Mdot given F; returning Mdot from simple standard expression.")
            return (F - self.F_in) / self.F_standard(Mdot=1.0, r=r)
        
    def Mdot_from_Teff(self, Teff, r):
        try:
            return brentq(f = lambda _mdot: Teff - self.T_eff(_mdot, r),
                          a=1.0, b=1e22)
        except ValueError:
            print("Could not solve for Mdot given Teff; returning Mdot from simple standard expression.")
            # return (F - self.F_in) / self.F_standard(Mdot=1.0, r=r)
            _F = 8 * pi / 3 * self._h(r) ** 7 / self.GM ** 4 * SIGMA_STEFBOLTZ * Teff ** 4
            return (_F - self.F_in) / self.F_standard(Mdot=1.0, r=r)
        
    
                
    
    
        