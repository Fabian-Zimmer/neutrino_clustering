# Default libraries
import os
import sys
import pathlib
import yaml
import time
from datetime import datetime, date as datetime_date, timedelta
import argparse
import random
import math
import glob
from concurrent.futures import ProcessPoolExecutor

# Conda install libraries
import numpy as np
import astropy.units as apu
import astropy.constants as apc
from astropy.coordinates import SkyCoord, GCRS, ICRS, Galactic, get_sun
from astropy.time import Time
from astropy.coordinates import get_body_barycentric, CartesianRepresentation, get_body_barycentric_posvel
import astropy.cosmology as apcosmo
cosmo = apcosmo.Planck18
from scipy.signal import savgol_filter
from scipy.special import zeta
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.optimize import root_scalar
from scipy.spatial import cKDTree
from scipy.integrate import quad
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.animation import FuncAnimation
import io
from icecream import ic
import healpy as hp
import pandas as pd

from functools import partial

# Pip install libraries
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import jax
import jax.scipy as jsp
from jax.scipy.integrate import trapezoid as trap
import jax.numpy as jnp
import chex
import diffrax
import natpy as nat
import colorcet as cc

# Switch jax to use float64 as default instead of float32
from jax.config import config
config.update("jax_enable_x64", True)

# Ignore UserWarnings issued by healpy
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="healpy")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


@chex.dataclass
class Params:

    #############
    ### Units ###
    #############
    
    MB_UNIT: float = 1000**2
    GB_UNIT: float = 1000**3
    Binary_MB_unit: float = 1024**2
    Binary_GB_unit: float = 1024**3
    Pi: float      = jnp.pi
    hc_val: float  = (apc.h/(2*Pi)*apc.c).convert(apu.eV*apu.cm).value

    eV: float     = 1.                         # Unit of energy: eV
    meV: float    = 1.0e-3*eV
    keV: float    = 1.0e3*eV
    MeV: float    = 1.0e3*keV
    GeV: float    = 1.0e3*MeV
    TeV: float    = 1.0e3*GeV
    erg: float    = TeV/1.602                  # erg
    J: float      = 1.0e7*erg                  # joule
    K: float      = 8.61732814974493e-5*eV     # Kelvin
    muK: float    = 1e-6*K                     # micro Kelvin
    kB: float     = 1.380649e-23*J/K

    cm: float     = (1/hc_val)/eV              # centi-meter
    m: float      = 1.0e2*cm
    km: float     = 1.0e3*m
    pc: float     = 3.08567758128e18*cm        # parsec
    kpc: float    = 1.0e3*pc
    Mpc: float    = 1.0e3*kpc
    AU: float     = (((1*apu.AU).convert(apu.m)).value)*m

    # note: 
    # Natural units defined via c=1, i.e. s/m = 299792458
    s: float      = 2.99792458e10*cm           # second
    day: float    = 24*60*60*s
    yr: float     = 365*24*60*60*s
    Gyr: float    = 1e9*yr
    t0: float     = 13.787*Gyr
    Hz: float     = 1.0/s
    c: float      = 299792458*m/s

    kg: float     = J/m**2*s**2
    gram: float   = kg/1000.
    Msun: float   = 1.988409870698051e30*kg    # Mass of the Sun
    G: float      = 6.6743e-11*m**3/kg/s**2    # Gravitational constant
    Da: float     = 1.66053906660e-27*kg       # Dalton or atomic mass unit (u)

    deg: float    = Pi/180.0                   # Degree
    arcmin: float = deg/60.                    # Arcminute
    arcsec: float = arcmin/60.                 # Arcsecond
    sr: float     = 1.                         # Steradian


    #################
    ### Constants ###
    #################

    # Cosmological parameters
    h: float       = cosmo.h
    H0: float      = h * 100 * km/s/Mpc
    Omega_C: float = cosmo.Odm0     # cold dark matter
    Omega_B: float = cosmo.Ob0      # baryonic matter
    Omega_R: float = cosmo.Ogamma0  # radiation
    Omega_M: float = cosmo.Om0      # total matter
    Omega_L: float = cosmo.Ode0     # dark energy

    # Temperature of CMB
    T_CMB: float = cosmo.Tcmb0.value * K

    # N_eff parameter from https://doi.org/10.1103/PhysRevD.108.L121301
    N_eff: float = 3.043

    # Temperature of CNB
    T_CNB = jnp.power(4/11, 1/3)*T_CMB * jnp.power(N_eff/3, 1/4)

    # Function for temperature of CNB with N_eff as input
    def T_CNB_fct(N_eff=N_eff, T_CMB=T_CMB):
        return jnp.power(4/11, 1/3)*T_CMB * jnp.power(N_eff/3, 1/4)

    # Redshift to last-scattering surfaces of CMB and CNB
    Z_LSS_CMB: float = 1100.0
    Z_LSS_CNB: float = 6.e9

    # Scale factor at neutrino decoupling
    a_ndec: float = 1e-10

    # Neutrino degrees of freedom (d.o.f.)
    g_nu: float = 1.

    # Cosmological (i.e. LambdaCDM) relic neutrino number density, per d.o.f.
    N0 = zeta(3.)/Pi**2 * T_CNB**3 * (3./4.)
    N0_cm3 = N0/cm**-3

    # Cosmolofical relic neutrino energy density, per d.o.g.
    rho_0 = Pi**2/30 * T_CNB**4 * (7./8.)

    # Key to fix random numbers of code
    key = jax.random.PRNGKey(0)

    #############################
    ### Simulation Parameters ###
    #############################
    p_start: float = 0.01
    p_stop: float  = 400.0
    p_num: int     = 1000
    phis: int      = 20
    thetas: int    = 20

    ##########################################
    ### Parameters - Mertsch et al. (2020) ###
    ##########################################

    # NFW parameters for Milky Way.
    Mvir_MW  = 2.03e12*Msun                            # Virial mass
    rho0_MW  = 1.06e7*(Msun/kpc**3)                    # density normalization
    Rs_MW    = 19.9*kpc                                # scale radius 
    Rvir_MW  = 333.5*kpc                               # virial radius

    # NFW parameters for Virgo Cluster.
    Mvir_VC  = 6.9e14*Msun                             # Virial mass
    rho0_VC  = 8.08e5*(Msun/kpc**3)                    # density normalization
    Rs_VC    = 399.1*kpc                               # scale radius 
    Rvir_VC  = 2328.8*kpc                              # virial radius

    # Coordinates.
    GLAT_VC = 74.44                                    # Galactic latitude
    GLON_VC = 283.81                                   # Galactic longitude
    DIST_VC = 16.5e3*kpc                               # Distance

    # Translated to cartesian coordinates [kpc] in our setup (from fct.halo_pos).
    X_VC    = np.array([-4289.63477282, 1056.51861602, 15895.27621304])



    ##########################
    ### PTOLEMY Parameters ###
    ##########################
    """From https://arxiv.org/abs/1902.05508."""
    sigma_avg: float = 3.834e-45*cm**2   # velocity-averaged cross-section
    M_T: float       = 100*gram          # target total Tritium mass in setup
    M_3H: float      = 2809.432*MeV      # atomic mass of Helium-3 (3He)
    M_3He: float     = 2809.413*MeV      # atomic mass of 3H (Tritium)
    m_3H: float      = 2808.921*MeV      # nuclear mass of 3H (Tritium)
    m_3He: float     = 2808.391*MeV      # nuclear mass of Helium-3 (3He)
    N_T: float       = M_T / m_3H        # number of Tritium atoms

    m_e: float       = 0.510998950*MeV   # electron mass
    m_n: float       = 939.56542052*MeV  # neutron mass
    m_p: float       = 938.27208816*MeV  # proton mass

    # Kinetic energy of electron at beta-decay endpoint, with a massless neutrino.
    """Lower eqn. of (5.5) in http://arxiv.org/abs/2109.02900."""
    K0_end: float = ((m_3H - m_e)**2 - m_3He**2)/(2*m_3H)


    ###########################
    ### Neutrino Parameters ###
    ###########################
    # Taken from https://arxiv.org/abs/2007.14792

    # PMNS matrix elements of electron flavor (absolute value squared), i.e.
    # |U_e1|^2, |U_e2|^2, |U_e3|^2
    # Taken as average values between upper and lower limits from eqn. (4.2)

    # For Normal Ordering
    U_ei_AbsSq_NO = jnp.array([0.681, 0.297, 0.0222])

    # For Inverted Ordering
    U_ei_AbsSq_IO = jnp.array([0.0222, 0.681, 0.297])

    # Measured neutrino mass squared differences from Table 3 (witout SK atm. data)
    Del_m21_Sq: float = 7.42e-5 * eV**2
    Del_m31_Sq_NO: float = 2.514e-3 * eV**2
    Del_m32_Sq_IO: float = 2.497e-3 * eV**2


    ###############################
    ### Primordial Fluctuations ###
    ###############################






    #############
    ### Plots ###
    #############

    # For paper:
    # SMALL_SIZE = 14
    # MEDIUM_SIZE = 18
    # BIGGER_SIZE = 22
    
    # For testing
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    mpl.rc('figure', figsize=(8, 8))
    mpl.rc('font', size=SMALL_SIZE)          # controls default text sizes
    mpl.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
    mpl.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    # mpl.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    # mpl.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    mpl.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    mpl.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    mpl.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    mpl.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    mpl.rc('xtick', direction='in', top=True)  # x ticks inside, top enabled
    mpl.rc('ytick', direction='in', right=True) # y ticks inside, right enabled
    mpl.rc('font', **{'family': 'serif', 'serif': ['cmr10']})
    mpl.rc('text', usetex=True)
    mpl.rc('axes.formatter', use_mathtext=True)
    mpl.rcParams['xtick.minor.visible'] = True

    healpy_dict = dict(
        coord=['G'],
        graticule=True,
        graticule_labels=True,
        xlabel="longitude",
        ylabel="latitude",
        cb_orientation="horizontal",
        projection_type="mollweide",
        flip='astro',
        latitude_grid_spacing=45,
        longitude_grid_spacing=45,
        phi_convention='counterclockwise',
    )

    grid_dict = dict(
        visible=True, 
        which="both", 
        ls="dashed", 
        alpha=0.5
    )


@chex.dataclass
class Primordial:

    ########################
    ### Class parameters ###
    ########################
    
    # Momenta of the Cl maps
    Cl_qs = np.geomspace(0.01, 100, 50)

    # Select index closest to desired q value
    q_select = 3.15
    # q_select = 0.01
    Cl_qi = np.abs(Cl_qs - q_select).argmin()

    # Define cutoff below which neutrinos are considered massless in CLASS
    massless_cutoff = 2e-5

    # Temperature perturbations, in micro Kelvin
    uK = 1e-6*Params.K    # micro Kelvin

    # Scaling factor to get correct unit/magnitude for the Cl's later
    Cl_scaling = 1e12

    # Multipole and wavemode related parameters
    l_min = 1
    l_max = 50
    ls = np.arange(l_min, l_max+1, 1)
    k_min, k_max = 1e-4, 1e-1  # 1/Mpc
    n_kmodes = 100
    k_magnitudes_full = np.logspace(np.log(k_min), np.log(k_max), n_kmodes, base=np.e)
    l_max_ncdm = 50

    # Momentum related parameters
    n_qbins = 50
    quad_strategy = 4
    q_max = 100

    # Integration stepsize
    int_step_size = 0.05

    # Resolution of healpix in our CNB simulations
    Nside = 8

    # Seed for np.random.seed() used in Tully & Zhang
    seed = 5

    z_cuts = np.round(np.geomspace(1e-1, 31+1e-1, 15)-0.1, 2)


    ###################################
    ### Parameters in certain units ###
    ###################################
    # note: T&Z functions need H0/(km/s/Mpc), T_CNB/K, kB/(eV/K), c/(km/s)

    H0 = Params.H0/(Params.km/Params.s/Params.Mpc)
    T_CNB = Params.T_CNB/Params.K
    kB = Params.kB/(Params.eV/Params.K)
    c = Params.c/(Params.km/Params.s)
    

    ############################
    ### Inflation parameters ###
    ############################

    As_default: float    = 2.215e-9
    pivot_default: float = 5e-2
    pivot: float         = 5e-4 
    ns: float            = 0.9619  # spectral tilt
    As: float            = As_default*(pivot/pivot_default)**(ns-1)  # primordial amplitude


@chex.dataclass
class Data:

    @staticmethod
    def check_array_size_GB(array):

        # Get the size of each element in bytes
        element_size_bytes = array.itemsize

        # Get the total number of elements
        total_elements = array.size

        # Calculate the total size in bytes
        total_size_bytes = element_size_bytes * total_elements

        # Convert the total size to gigabytes (GB)
        total_size_gb = total_size_bytes / (1024 ** 3)

        print(f'Memory size is {total_size_gb:.3f} GB')


    @staticmethod
    def load_neutrino_velocities(halo_ID:str):

        file_name = f'neutrino_vectors_numerical_{halo_ID}'

        nu_vels = jnp.array([
            jnp.load(f'Data/{file_name}_batch{i+1}.npy')[...,3:6]
            for i in range(8)
        ]).reshape(-1,100,3)

        return nu_vels


@chex.dataclass
class Physics:

    """
    Cosmological functions.
    """

    @staticmethod
    @jax.jit
    def Hubble_z(z, args):
        """Evolution of Hubble parameter valid at low redshift."""
        return args.H0*jnp.sqrt(args.Omega_M*(1.+z)**3 + args.Omega_L)
    

    @staticmethod
    @jax.jit
    def time_z(z, args):
        """
        Time of the universe at redshift z, when dark matter and dark energy are 
        dominating and rest is negligible. Eqn. (4.9).
        """
        r_z = args.Omega_L/args.Omega_M * 1/(1.+z)**3
        return 2/(3*args.H0*jnp.sqrt(args.Omega_L)) * jnp.log(jnp.sqrt(r_z) + jnp.sqrt(1+r_z))
    

    @staticmethod
    def neutrino_masses(m_lightest, ordering, args):
        """Returns the 3 neutrino masses with given ordering and lightest mass."""

        def normal_ordering(_):
            m1 = m_lightest
            m2 = jnp.sqrt(m1**2 + args.Del_m21_Sq)
            m3 = jnp.sqrt(m1**2 + args.Del_m31_Sq_NO)
            return jnp.array([m1, m2, m3])

        def inverted_ordering(_):
            m3 = m_lightest
            m2 = jnp.sqrt(m3**2 + args.Del_m32_Sq_IO)
            m1 = jnp.sqrt(m2**2 - args.Del_m21_Sq)
            return jnp.array([m3, m1, m2])

        # Map orderings to indices
        ordering_index = {'NO': 0, 'IO': 1}
        if ordering not in ordering_index:
            raise ValueError("Invalid 'ordering' value. It should be 'NO' or 'IO'.")

        # Switch based on the ordering
        branches = [normal_ordering, inverted_ordering]
        return jax.lax.switch(ordering_index[ordering], branches, None)
    

    @jax.jit
    def velocities_to_momenta_ND_halo_0th_axis(v_arr, m_arr, args):
        """
        Converts velocities to magnitude of momentum [eV] and ratio y=p/T_CNB, 
        according to desired target mass (and mass used in simulation).
        """

        # Magnitudes of velocity
        mags_arr = jnp.linalg.norm(v_arr, axis=-1)
        
        # Reshape m_arr to have the right number of singleton dimensions
        new_shape = (1,) + (m_arr.shape[0],) + (1,) * (jnp.ndim(mags_arr)-1)
        m_sync = jnp.reshape(m_arr, new_shape)

        # From velocity (magnitude) in kpc/s to momentum in eV
        v_unit = args.kpc/args.s
        p_dim = mags_arr[:, None, ...]*v_unit * m_sync

        # p/T_CNB ratio
        y = p_dim / args.T_CNB

        return p_dim, y


    @jax.jit
    def velocities_to_momentum_vec_ND_halo_0th_axis(v_arr, m_arr, args):
        
        # Reshape m_arr to have the right number of singleton dimensions
        new_shape = (1,) + (m_arr.shape[0],) + (1,) * (jnp.ndim(v_arr)-1)
        m_sync = jnp.reshape(m_arr, new_shape)

        # From velocity (vector) in kpc/s to momentum in eV
        v_unit = args.kpc/args.s

        # Scaling the velocity vector by mass and unit conversion
        p_vec = v_arr[:, None, ...]*v_unit * m_sync  

        # Compute magnitude of momentum vectors
        p_dim = jnp.linalg.norm(p_vec, axis=-1)

        # p/T_CNB ratio
        y = p_dim / args.T_CNB

        return p_vec, p_dim, y


    @jax.jit
    def velocities_to_momenta(v_arr, m_arr, args):
        """
        Converts velocities to magnitude of momentum [eV] and ratio y=p/T_CNB, 
        according to desired target mass (and mass used in simulation).
        """

        # Magnitudes of velocity.
        mags_arr = jnp.linalg.norm(v_arr, axis=-1)
        
        # From velocity (magnitude) in kpc/s to momentum in eV.
        p_dim = mags_arr[None, :, :]*(args.kpc/args.s) * m_arr[:, None, None]

        # p/T_CNB ratio.
        y = p_dim / args.T_CNB

        return p_dim, y
    

    @jax.jit
    def velocities_to_momenta_all_sky(v_arr, m_arr, args):
        """
        Converts velocities to magnitude of momentum [eV] and ratio y=p/T_CNB, 
        according to desired target mass (and mass used in simulation). But now assuming the input array to have starting dimentions of (Npix, nu_per_pix), instead of just the total number of neutrinos (N,).
        """

        # Magnitudes of velocity.
        mags_arr = jnp.linalg.norm(v_arr, axis=-1)
        
        # From velocity (magnitude) in kpc/s to momentum in eV.
        p_dim = mags_arr[None,...]*(args.kpc/args.s) * m_arr[:,None,None,None]

        # p/T_CNB ratio.
        y = p_dim / args.T_CNB

        return p_dim, y


    @jax.jit
    def velocities_to_momenta_1D(v_arr, m_arr, args):

        # Magnitudes of velocity.
        mags_arr = jnp.linalg.norm(v_arr, axis=-1)
        
        # From velocity (magnitude) in kpc/s to momentum in eV.
        p_dim = mags_arr[None, :]*(args.kpc/args.s) * m_arr[:, None]

        # p/T_CNB ratio.
        y = p_dim / args.T_CNB

        return p_dim, y


    @jax.jit
    def velocities_to_momenta_individual(v_arr, m_arr, args):
        
        # Reshape m_arr to have the right number of singleton dimensions
        new_shape = (1,) + (m_arr.shape[0],) + (1,) * (jnp.ndim(v_arr)-1)
        m_sync = jnp.reshape(m_arr, new_shape)

        # From velocity (magnitude) in kpc/s to momentum in eV
        v_unit = args.kpc/args.s
        p_dim = v_arr[:, None, ...]*v_unit * m_sync

        # p/T_CNB ratio.
        y = p_dim / args.T_CNB

        return p_dim, y


    @staticmethod
    def filter_momenta(p_arr, y_arr, m_len, args):

        # Select necessary sub-arrays
        p_init = p_arr[...,0]
        p_back = p_arr[...,-1]
        y_init = y_arr[...,0]

        # Sort
        sort_indices = p_init.argsort(axis=-1)
        p_back_sort = jnp.take_along_axis(p_back, sort_indices, axis=-1)
        y_init_sort = jnp.take_along_axis(y_init, sort_indices, axis=-1)

        # Select "most likely", i.e. most clustered neutrinos, of each velocity batch
        p_back_blocks = p_back_sort.reshape((m_len, args.p_num, args.phis*args.thetas))
        p_back_select = jnp.min(p_back_blocks, axis=-1)
        y_init_blocks = y_init_sort.reshape((m_len, args.p_num, args.phis*args.thetas))
        y_init_select = y_init_blocks[...,0]

        return p_back_select, y_init_select
    

    @staticmethod
    def SHM_PSD(v_arr, v_esc_MW, v_0, args):

        v_for_f_mag = jnp.linalg.norm(v_arr, axis=-1)

        # Create mask for velocity magnitudes smaller than escape velocity
        mask = v_for_f_mag < v_esc_MW

        # Compute standard halo model (SHM) distribution
        exp_term = jnp.exp(-v_for_f_mag**2/v_0**2)
        f_v = jnp.where(
            mask, 
            jnp.power(jnp.pi*v_0**2, -3/2) * exp_term, 
            jnp.zeros_like(exp_term))
        
        # Normalisation constant, and normalised velocity distribution
        z = v_esc_MW/v_0
        N_esc = jsp.special.erf(z) - 2/jnp.sqrt(jnp.pi)*z*jnp.exp(-z**2)
        f_v_normalized = f_v / N_esc

        return args.N0 * f_v_normalized

    @staticmethod
    @jax.jit
    def Fermi_Dirac(p, args):
        return 1/(jnp.exp(p/(args.T_CNB)) + 1)
    

    @jax.jit
    def number_density(p_init, p_back, pix_sr, args):
      
        """Neutrino number density obtained by integration over initial momenta.

        Args:
            p_init (array): neutrino momentum today
            p_back (array): neutrino momentum at z_back (final redshift in sim.)

        Returns:
            array: Value of relic neutrino number density in (1/cm^3).
        """    

        # note: trapz integral method needs sorted (ascending) "x-axis" array.
        ind = p_init.argsort(axis=-1)
        p_init_sort = jnp.take_along_axis(p_init, ind, axis=-1)
        p_back_sort = jnp.take_along_axis(p_back, ind, axis=-1)

        # Fermi-Dirac values with momenta at end of sim
        FD_arr = Physics.Fermi_Dirac(p_back_sort, args)

        # Calculate number density
        y = p_init_sort**3 * FD_arr  # dlog integrand
        x = p_init_sort
        n_raw = trap(y, jnp.log(x), axis=-1)

        # Multiply by constants and/or solid angles and convert to 1/cm**3.
        n_cm3 = pix_sr * args.g_nu/((2*args.Pi)**3) * n_raw / (1/args.cm**3)

        return jnp.array(n_cm3)
    

    @staticmethod
    @jax.jit
    def Fermi_Dirac_boosted(p_vec, boost_vec, args):
        """Fermi-Dirac with direction-dependent effective temperature"""
        v_mag = jnp.sqrt(jnp.sum(boost_vec**2))
        gamma = 1/jnp.sqrt(1 - v_mag**2)
        
        # Compute cos(theta) between momentum and boost vectors
        p_mag = jnp.sqrt(jnp.sum(p_vec**2, axis=-1))
        cos_theta = jnp.sum(p_vec * boost_vec, axis=-1)/(p_mag * v_mag)
        
        # Effective temperature factor
        T_factor = gamma * (1 - v_mag * cos_theta)
        
        return 1/(jnp.exp(p_mag/(args.T_CNB / T_factor)) + 1)


    @jax.jit
    def number_density_boosted(p_init_mag, p_back_vec, boost_vec, pix_sr, args):
      
        """Neutrino number density obtained by integration over initial momenta.

        Args:
            p_init (array): neutrino momentum today
            p_back (array): neutrino momentum at z_back (final redshift in sim.)

        Returns:
            array: Value of relic neutrino number density in (1/cm^3).
        """

        # Fermi-Dirac values with momenta at end of sim
        FD_arr = Physics.Fermi_Dirac_boosted(p_back_vec, boost_vec, args)

        # Calculate number density
        y = p_init_mag**3 * FD_arr  # dlog integrand
        x = p_init_mag
        n_raw = trap(y, jnp.log(x), axis=-1)

        # Multiply by constants and/or solid angles and convert to 1/cm**3.
        n_cm3 = pix_sr * args.g_nu/((2*args.Pi)**3) * n_raw / (1/args.cm**3)

        return jnp.array(n_cm3)


    @jax.jit
    def Fermi_Dirac_Delta(p_arr, Deltas, args):
        temp_Deltas = args.T_CNB * (1 + Deltas)
        return 1 / (jnp.exp(p_arr / temp_Deltas) + 1)


    @jax.jit
    def number_density_Delta(p_init, p_back, Deltas, pix_sr, args):
      
        """Neutrino number density obtained by integration over initial momenta.

        Args:
            p_init (array): neutrino momentum today
            p_back (array): neutrino momentum at z_back (final redshift in sim.)

        Returns:
            array: Value of relic neutrino number density in (1/cm^3).
        """    

        # note: trapz integral method needs sorted (ascending) "x-axis" array.
        ind = p_init.argsort(axis=-1)
        p_init_sort = jnp.take_along_axis(p_init, ind, axis=-1)
        p_back_sort = jnp.take_along_axis(p_back, ind, axis=-1)

        # Also sort temp. perturbation array to match new order of momenta
        Deltas_sort = jnp.take_along_axis(Deltas, ind, axis=-1)

        # Fermi-Dirac values with momenta at end of sim
        FD_arr_Delta = Physics.Fermi_Dirac_Delta(p_back_sort, Deltas_sort, args)

        # Calculate number density
        y = p_init_sort**3 * FD_arr_Delta  # dlog integrand
        x = p_init_sort
        n_raw = trap(y, jnp.log(x), axis=-1)

        # Multiply by constants and/or solid angles and convert to 1/cm**3.
        n_cm3 = pix_sr * args.g_nu/((2*args.Pi)**3) * n_raw / (1/args.cm**3)

        return n_cm3


    @jax.jit
    def transform_momenta_to_orig_frame_OLD(p_vec, boost_vec):
        """
        Transform momentum vectors to original frame using p'_i = γ(p_i - v_i E)
        """
        Lf = 1/jnp.sqrt(1 - jnp.sum(boost_vec**2))  # Lorentz factor
        p_mag = jnp.sqrt(jnp.sum(p_vec**2, axis=-1))
        
        # Transform each component
        p_orig_vec = Lf*(p_vec - boost_vec[None,None,None,None,:]*p_mag[...,None])
        
        # Get magnitude and unit vectors
        p_orig_mag = jnp.sqrt(jnp.sum(p_orig_vec**2, axis=-1))
        p_orig_unit = p_orig_vec / p_orig_mag[...,None]
        
        return p_orig_mag, p_orig_unit


    @jax.jit
    def transform_momenta_to_orig_frame(p_vec, boost_vec, masses=None):
        """
        Transform momentum vectors to original frame using proper Lorentz transformation
        
        Args:
            p_vec: Momentum vectors with shape (..., 3)
            boost_vec: Boost velocity vector (3,)
            masses: Mass values for the second dimension
        """

        # Calculate boost magnitude squared
        boost_mag_squared = jnp.sum(boost_vec**2)
        
        # Check if boost is effectively zero
        # For jax, we can use jax.lax.cond for conditional execution
        def zero_boost_case(_):
            p_vec_mag = jnp.sqrt(jnp.sum(p_vec**2, axis=-1))
            return p_vec_mag, p_vec / p_vec_mag[..., None]
        
        def normal_boost_case(_):
            # Calculate Lorentz factor
            boost_mag = jnp.sqrt(jnp.sum(boost_vec**2))
            Lf = 1/jnp.sqrt(1 - boost_mag**2)  # Lorentz factor
            
            # Calculate momentum magnitude
            p_mag = jnp.sqrt(jnp.sum(p_vec**2, axis=-1))
            
            # Calculate energy
            if masses is not None:
                # Ensure masses is properly dimensioned for this specific p_vec
                # If p_vec has shape (1, 5, 768, 1000, 3) and masses has shape (5,)
                # We want to create energy with shape (1, 5, 768, 1000)
                E = jnp.sqrt(p_mag**2 + jnp.reshape(masses, (1, -1, 1, 1))**2)
            else:
                # Massless case (E = p)
                E = p_mag
            
            # Dot product of momentum and boost velocity
            # p_dot_v = jnp.sum(p_vec * boost_vec, axis=-1)

            # Calculate p⃗' = p⃗ + (γ-1)(p⃗·v̂)v̂/v² - γEv⃗
            # term1 = p_vec
            # term2 = (Lf-1) * (p_dot_v / boost_mag**2)[..., None] * boost_vec
            # term3 = -Lf * E[..., None] * boost_vec
            # p_orig_vec = term1 + term2 + term3

            # Calculate p⃗' = γ(p⃗ - Ev⃗)
            p_orig_vec = Lf * (p_vec - E[..., None]*boost_vec)
            
            # Get magnitude and unit vectors
            p_orig_mag = jnp.sqrt(jnp.sum(p_orig_vec**2, axis=-1))
            p_orig_unit = p_orig_vec / p_orig_mag[..., None]
            
            return p_orig_mag, p_orig_unit
        
        # Use jax.lax.cond to handle the special case
        return jax.lax.cond(
            boost_mag_squared < 1e-10,  # Threshold for "effectively zero"
            zero_boost_case,
            normal_boost_case,
            operand=None
        )


    def get_p_vec_pixels(p_unit, nside):
        """
        Convert momentum vectors to HEALPix pixel indices
        
        Args:
            p_unit: unit momentum vectors, shape (H, M, Npix, p_num, 3)
            nside: HEALPix nside parameter
        
        Returns:
            pixel indices, shape (H, M, Npix, p_num)

        Usage example:
            # Pixels (indices) that 1yr momenta in GC frame point at
            # pixel_indices = Physics.get_p_vec_pixels(
            #     p_unit=p_GC_unit, nside=simdata.Nside)
            # (1, masses, Npix, p_num)
        """
        
        # Convert to spherical coordinates
        theta = np.arccos(p_unit[..., 2])
        phi = np.arctan2(p_unit[..., 1], p_unit[..., 0])
        
        # Get pixels
        pixels = hp.ang2pix(nside, theta, phi)
        
        return jnp.array(pixels)


    @jax.jit
    def interpolate_fd_values(p_GC_mag, pixel_indices, p_grid, fd_vals):
        """
        Args:
            p_GC_mag: shape (1, M, Npix, p_num) - transformed momenta magnitudes
            pixel_indices: shape (1, M, Npix, p_num) - pixel indices these momenta point to
            p_grid: shape (H, M, Npix, p_num) - original momentum grid for FD values
            fd_vals: shape (H, M, Npix, p_num) - FD values on grid
        """
        # Process each halo and mass configuration separately to avoid broadcasting issues
        def halo_mass_fun(h, m, pixel_pos):
            # Get the values for this (halo, mass, pixel position) combination
            p_interp = p_GC_mag[h, m, pixel_pos]  # shape (p_num,)
            pixels = pixel_indices[h, m, pixel_pos]  # shape (p_num,)
            
            # Get the grid values for these pixels
            p0_pixels = p_grid[h, m, pixels]  # shape (p_num, p_num)
            fd_pixels = fd_vals[h, m, pixels]  # shape (p_num, p_num)
            
            # Find indices where each p_interp value sits in its corresponding grid
            idx = jnp.sum(p0_pixels <= p_interp[:, None], axis=1) - 1
            idx = jnp.clip(idx, 0, p_grid.shape[-1] - 2)
            
            # Get interpolation points
            x0 = jnp.take_along_axis(p0_pixels, idx[:, None], axis=1)[:, 0]
            x1 = jnp.take_along_axis(p0_pixels, (idx+1)[:, None], axis=1)[:, 0]
            y0 = jnp.take_along_axis(fd_pixels, idx[:, None], axis=1)[:, 0]
            y1 = jnp.take_along_axis(fd_pixels, (idx+1)[:, None], axis=1)[:, 0]
            
            # Linear interpolation
            slope = (y1 - y0) / (x1 - x0)
            return y0 + slope * (p_interp - x0)
        
        def pixel_fun(i, result):
            H, M = fd_vals.shape[:2]
            for h in range(H):
                for m in range(M):
                    vals = halo_mass_fun(h, m, i)
                    result = result.at[h, m, i].set(vals)
            return result
        
        result = jnp.zeros_like(fd_vals)
        return jax.lax.fori_loop(0, fd_vals.shape[-2], pixel_fun, result)


    # @jax.jit
    def average_velocity(v_init, p_init, p_back, args):

        # Syncronize velocity array to add dimension for masses
        v_sync = v_init[:, None, ...]

        # Magnitudes of momenta
        p_mags_init = jnp.linalg.norm(p_init, axis=-1)
        p_mags_back = jnp.linalg.norm(p_back, axis=-1)

        option = 3


        if option == 1:

            ### ============ ###
            ### With sorting ### #? broken? see plot...
            ### ============ ###
            sort_indices_p = p_mags_init.argsort(axis=-1)
            # sort_indices_v = jnp.repeat(sort_indices_p[..., None], 3, axis=-1)
            sort_indices_v = sort_indices_p[..., None]

            p_init_sort = jnp.take_along_axis(p_mags_init, sort_indices_p, axis=-1)
            p_back_sort = jnp.take_along_axis(p_mags_back, sort_indices_p, axis=-1)
            v_sync_sort = jnp.take_along_axis(v_sync, sort_indices_v, axis=-2)

            # Fermi-Dirac values using the end-of-sim momenta (Liouville's Theorem)
            FD_arr = Physics.Fermi_Dirac(p_back_sort, args)

            # Calculate average velocity integral, log integration
            integrand_v = p_init_sort[..., None] * FD_arr[..., None] * v_sync_sort
            int_array_v = p_init_sort[..., None]
            v_int = trap(integrand_v, x=jnp.log(int_array_v), axis=-2)

            print(int_array_v.shape, integrand_v.shape)

            h_pick = 0
            m_pick = 0

            x_axis = jnp.squeeze(int_array_v)[h_pick,m_pick,:]
            y_axis = jnp.linalg.norm(integrand_v, axis=-1)[h_pick,m_pick,:]
            print(x_axis.shape, y_axis.shape)

            plt.plot(x_axis, y_axis)
            plt.show(); plt.close()

            # Calculate normalization
            integrand_n = p_init_sort[..., None] * FD_arr[..., None]
            int_array_n = p_init_sort[..., None]
            n_int = trap(integrand_n, x=jnp.log(int_array_n), axis=-2)


        elif option == 2:

            ### =============== ###
            ### Without sorting ### #? correct?
            ### =============== ###
            # Fermi-Dirac values using the end-of-sim momenta (Liouville's Theorem)
            FD_arr = Physics.Fermi_Dirac(p_mags_back, args)

            # Calculate average velocity integral, log integration
            integrand_v = p_mags_init[..., None] * FD_arr[..., None] * v_sync
            int_array_v = p_mags_init[..., None]
            v_int = trap(integrand_v, x=jnp.log(int_array_v), axis=-2)

            print(int_array_v.shape, integrand_v.shape)

            x_axis = jnp.squeeze(int_array_v)[0,0,:]
            y_axis = jnp.linalg.norm(integrand_v, axis=-1)[0,0,:]
            print(x_axis.shape, y_axis.shape)

            plt.plot(x_axis, y_axis)
            plt.show(); plt.close()

            # Calculate normalization
            integrand_n = p_mags_init[..., None] * FD_arr[..., None]
            int_array_n = p_mags_init[..., None]
            n_int = trap(integrand_n, x=jnp.log(int_array_n), axis=-2)


        elif option == 3:

            ### =========== ###
            ### Simple sums ###
            ### =========== ###
            # Fermi-Dirac values using the end-of-sim momenta (Liouville's Theorem)
            FD_arr = Physics.Fermi_Dirac(p_mags_back, args)

            # Calculate average velocity integral, log integration
            v_int = jnp.sum(FD_arr[..., None] * v_sync, axis=-2)

            # Calculate normalization
            n_int = jnp.sum(FD_arr[..., None], axis=-2)


        return v_int / n_int


    @jax.jit
    def average_velocity_Delta(v_init, p_init, p_back, Deltas, args):

        # Syncronize velocity array to add dimension for masses
        v_sync = v_init[:, None, ...]

        # Magnitudes of momenta
        p_mags_init = jnp.linalg.norm(p_init, axis=-1)
        p_mags_back = jnp.linalg.norm(p_back, axis=-1)

        option = 3


        if option == 1:

            ### ============ ###
            ### With sorting ### #? broken? see plot...
            ### ============ ###
            sort_indices_p = p_mags_init.argsort(axis=-1)
            # sort_indices_v = jnp.repeat(sort_indices_p[..., None], 3, axis=-1)
            sort_indices_v = sort_indices_p[..., None]

            p_init_sort = jnp.take_along_axis(p_mags_init, sort_indices_p, axis=-1)
            p_back_sort = jnp.take_along_axis(p_mags_back, sort_indices_p, axis=-1)
            v_sync_sort = jnp.take_along_axis(v_sync, sort_indices_v, axis=-2)

            # Fermi-Dirac values using the end-of-sim momenta (Liouville's Theorem)
            FD_arr = Physics.Fermi_Dirac(p_back_sort, args)

            # Calculate average velocity integral, log integration
            integrand_v = p_init_sort[..., None] * FD_arr[..., None] * v_sync_sort
            int_array_v = p_init_sort[..., None]
            v_int = trap(integrand_v, x=jnp.log(int_array_v), axis=-2)

            print(int_array_v.shape, integrand_v.shape)

            x_axis = jnp.squeeze(int_array_v)[0,0,:]
            y_axis = jnp.linalg.norm(integrand_v, axis=-1)[0,0,:]
            print(x_axis.shape, y_axis.shape)

            plt.plot(x_axis, y_axis)
            plt.show(); plt.close()

            # Calculate normalization
            integrand_n = p_init_sort[..., None] * FD_arr[..., None]
            int_array_n = p_init_sort[..., None]
            n_int = trap(integrand_n, x=jnp.log(int_array_n), axis=-2)


        elif option == 2:

            ### =============== ###
            ### Without sorting ### #? correct?
            ### =============== ###
            # Fermi-Dirac values using the end-of-sim momenta (Liouville's Theorem)
            FD_arr = Physics.Fermi_Dirac(p_mags_back, args)

            # Calculate average velocity integral, log integration
            integrand_v = p_mags_init[..., None] * FD_arr[..., None] * v_sync
            int_array_v = p_mags_init[..., None]
            v_int = trap(integrand_v, x=jnp.log(int_array_v), axis=-2)

            print(int_array_v.shape, integrand_v.shape)

            x_axis = jnp.squeeze(int_array_v)[0,0,:]
            y_axis = jnp.linalg.norm(integrand_v, axis=-1)[0,0,:]
            print(x_axis.shape, y_axis.shape)

            plt.plot(x_axis, y_axis)
            plt.show(); plt.close()

            # Calculate normalization
            integrand_n = p_mags_init[..., None] * FD_arr[..., None]
            int_array_n = p_mags_init[..., None]
            n_int = trap(integrand_n, x=jnp.log(int_array_n), axis=-2)


        elif option == 3:

            ### =========== ###
            ### Simple sums ###
            ### =========== ###
            # Fermi-Dirac values using the end-of-sim momenta (Liouville's Theorem)
            FD_arr = Physics.Fermi_Dirac_Delta(p_mags_back, Deltas, args)
            print(FD_arr.shape)

            # Calculate average velocity integral, log integration
            v_int = jnp.sum(FD_arr[..., None] * v_sync, axis=-2)

            # Calculate normalization
            n_int = jnp.sum(FD_arr[..., None], axis=-2)


        return v_int / n_int


    @staticmethod
    def Energy_Momentum_conversion(X_arr, mass_arr, mode='E_to_p'):

        def Energy_to_momentum(_):
            signs = jnp.sign(X_arr[None,:])
            diffs = jnp.sqrt(X_arr[None, :]**2 - mass_arr[:, None]**2)
            return signs * diffs
        
        def Momentum_to_energy(_):
            signs = jnp.sign(X_arr[None,:])
            diffs = jnp.sqrt(X_arr[None, :]**2 + mass_arr[:, None]**2)
            return signs * diffs

        # Map modes to indices
        mode_index = {'E_to_p': 0, 'p_to_E': 1}
        if mode not in mode_index:
            raise ValueError("Invalid 'ordering' value. It should be 'E_to_p' or 'p_to_E'.")

        # Switch based on the mode
        branches = [Energy_to_momentum, Momentum_to_energy]
        return jax.lax.switch(mode_index[mode], branches, None)


    ### ---------------------------------------------- ###
    ### Functions using "base" number_density function ###
    ### ---------------------------------------------- ###

    @staticmethod
    @jax.jit
    def number_densities_all_sky(v_arr, m_arr, pix_sr, args):

        # Convert velocities to momenta.
        p_arr, _ = Physics.velocities_to_momenta_all_sky(v_arr, m_arr, args)
        # (masses, Npix, neutrinos per pixel, 2 (first and last time step))
        
        # Compute number densities
        nu_dens = Physics.number_density(
            p_arr[..., 0], p_arr[..., -1], pix_sr, args)
        # (masses, Npix)

        return nu_dens


    @staticmethod
    @jax.jit
    def number_densities_mass_range(v_arr, m_arr, pix_sr, args):
        
        # Convert velocities to momenta.
        p_arr, _ = Physics.velocities_to_momenta(v_arr, m_arr, args)

        nu_dens = Physics.number_density(
            p_arr[...,0], p_arr[...,-1], pix_sr, args)

        return nu_dens


    @staticmethod
    def number_densities_z_evolution(
        sim_vels, nu_masses, z_arr, out_file=None, pix_sr=4*Params.Pi,
        average=False, m_start=0.01, z_start=0., sim_type='single_halos'
    ):
        
        # Convert velocities to momenta.
        p_arr, _ = Physics.velocities_to_momenta(sim_vels, nu_masses, Params)

        if average:
            inds = np.array(np.where(z_arr >= z_start)).flatten()
            temp = [
                Physics.number_density(p_arr[...,0], p_arr[...,k], pix_sr, Params) for k in inds
            ]
            num_densities = np.mean(np.array(temp.T), axis=-1)
        else:
            num_densities = Physics.number_density(p_arr[...,0], p_arr[...,-1], pix_sr, Params)

        if 'all_sky' in sim_type:
            return num_densities
        else:
            np.save(f'{out_file}', num_densities)


    @staticmethod
    def CDF_Fermi_Dirac_normed(p, T, p_norm):
        CDF = p - T*np.log(np.exp(p/T)+1) + T*np.log(2)
        norm = p_norm - T*np.log(np.exp(p_norm/T)+1) + T*np.log(2)
        return CDF/norm


    @staticmethod
    def sample_Fermi_Dirac_momentum(N, T, p_norm):
        p_samples = np.zeros(N)
        
        for i in range(N):
            u = np.random.uniform(0, 1)
            
            try:
                # Use Brent's method for root finding
                root = root_scalar(
                    lambda p: Physics.CDF_Fermi_Dirac_normed(p, T, p_norm) - u, 
                    bracket=[0.01 * T, p_norm], 
                    method='brentq'
                )
                p_samples[i] = root.root

            except ValueError:
                # print(f"Skipped iteration {i} due to bracketing issues.")
                continue
        
        return p_samples


@chex.dataclass
class Cosmo:


    @staticmethod
    @jax.jit
    def get_Chi_integrand(z, p_0, m_nu, args):
        
        # Comoving distance integral for massive particles
        p_div_H = p_0/Physics.Hubble_z(z, args)
        sqrt_term = jnp.sqrt(p_0**2 + m_nu**2/((1+z)**2))

        return p_div_H / sqrt_term


    @staticmethod
    @jax.jit
    def get_LSS_distance(z_LSS, p_0, m_nu, args):
    # def CALC_LSS_distance(z_LSS, p_0=1.0, m_nu=0.0):

        # Logarithmically spaced redshifts starting at zero, ending at z_LSS
        z_int_range = Utils.make_logz_arr(0, z_LSS, 100)

        # Comoving distance (Chi) to last scattering surface (LSS)
        integrand = Cosmo.get_Chi_integrand(z_int_range[:, None], p_0, m_nu, args)
        Chi_LSS =  trap(integrand, z_int_range, axis=0)

        return Chi_LSS


@chex.dataclass
class Ptolemy:

    """
    The Equations are referring to the Akita et al. (2022) paper:
    http://arxiv.org/abs/2109.02900
    """

    @staticmethod
    @jax.jit
    def Gamma_i_to_j(m_i, lambda_ij, args):
        """Eqn. (3.3)."""
        return 1/args.t0 * m_i/(50*args.meV) * (lambda_ij/6.2e-16)**2


    @staticmethod
    @jax.jit
    def n_i_today(m_i, lambda_ij, f_c, n0_i, args):
        """Eqn. (3.4)"""
        return jnp.exp(- m_i/(50*args.meV) * (lambda_ij/6.2e-16)**2) * f_c * n0_i
    

    @staticmethod
    @jax.jit
    def p_j_z(p_j, z):
        """Upper eqn. of (4.6)."""
        return p_j*(1.+z)
    

    @staticmethod
    @jax.jit
    def E_j_z(p_j, m_j, z):
        """Lower eqn. of (4.6)."""
        return jnp.sqrt(p_j**2*(1.+z)**2 + m_j**2)


    @staticmethod
    @jax.jit
    def p_star(m_i, m_j, m_x):
        """Upper eqn. of (4.11)."""
        return 1/2/m_i*jnp.sqrt((m_i**2 - (m_j + m_x)**2)*(m_i**2 - (m_j - m_x)**2))


    @staticmethod
    @jax.jit
    def E_star(m_i, m_j, m_x):
        """Lower eqn. of (4.11)."""
        return (m_i**2 + m_j**2 - m_x**2)/(2*m_i)


    @staticmethod
    @jax.jit
    def E_j_from_E_e(E_e, args):
        """Upper eqn. of (5.5)."""
        return E_e - args.K0_end - args.m_e


    @staticmethod
    @jax.jit
    def E_e_from_E_j(E_j, args):
        """Inverse of upper eqn. of (5.5)."""
        return E_j + args.K0_end + args.m_e


    @staticmethod
    @jax.jit
    def Ei_end(m_i, args):
        """Eqn. (5.9)"""
        return args.K0_end + args.m_e - m_i


    @staticmethod
    @jax.jit
    def E_end(m_lightest, args):
        """Eqn. (5.10)"""
        return args.K0_end + args.m_e - m_lightest

    @staticmethod
    @jax.jit
    def Q_value(m_nu, args):
        return args.m_3H - args.m_3He - args.m_e - m_nu


    @staticmethod
    @jax.jit
    def K_end(m_nu, args):
        return ((args.m_3H - args.m_e)**2 - (m_nu - args.m_3He)**2)/(2*args.m_3H)


    @staticmethod
    @jax.jit
    def H_factor(E_e, m_i, args):
        """Eqn. (5.8)"""
        
        numer1 = 1 - (args.m_e**2/(E_e*args.m_3H))
        denom1 = (1 - (2*E_e/args.m_3H) + (args.m_e**2/args.m_3H**2))**2
        term1 = numer1/denom1
        
        # set negative values to zero
        y_i = jnp.clip(Ptolemy.Ei_end(m_i, args) - E_e, 0., None)
        
        term2 = jnp.sqrt(y_i * (y_i + (2*m_i*args.m_3He/args.m_3H)))
        term3 = y_i + m_i/args.m_3H*(args.m_3He + m_i)
        
        return term1*term2*term3
        

    @staticmethod
    def Beta_decay_spectrum(E_e, nu_masses, ordering, args):
        """Eqn. (5.7)"""

        # H factors for each neutrino mass.
        H_vals = jnp.array([Ptolemy.H_factor(E_e, m_i, args) for m_i in nu_masses])

        # dGamma_beta/dE_e.
        def normal_ordering(_):
            return jnp.tile(args.U_ei_AbsSq_NO, (len(E_e),1)).T
        
        def inverted_ordering(_):
            return jnp.tile(args.U_ei_AbsSq_IO, (len(E_e),1)).T

        # Map orderings to indices
        ordering_index = {'NO': 0, 'IO': 1}
        if ordering not in ordering_index:
            raise ValueError("Invalid 'ordering' value. It should be 'NO' or 'IO'.")

        branches = [normal_ordering, inverted_ordering]
        U_ei_tiled = jax.lax.switch(ordering_index[ordering], branches, None)

        return args.sigma_avg/args.Pi**2*args.N_T * jnp.sum(U_ei_tiled*H_vals, axis=0)


    @staticmethod
    @jax.jit
    def dndE_decay(E_j, m_j, m_i, m_x, tau_i, n0_i, args):
        """Eqn. (4.13)."""

        def true_branch(_):
            return jnp.array(0.0)

        def false_branch(_):
            zE = (Ptolemy.p_star(m_i, m_j, m_x) / jnp.sqrt(E_j**2 - m_j**2)) - 1
            tE = Physics.time_z(zE, args)
            const_term = n0_i * jnp.exp(-tE / tau_i) / (Physics.Hubble_z(zE, args) * tau_i)
            energy_term = E_j / (E_j**2 - m_j**2)
            return const_term * energy_term

        E_star = Ptolemy.E_star(m_i, m_j, m_x)
        condition = (E_j <= m_j) | (E_j > E_star)

        return jax.lax.cond(condition, true_branch, false_branch, None)


    @staticmethod
    def dGammaTilde_CNB_dE_e(E_e, m_j, m_i, m_x, tau_i, n0_i, ordering, args):
        """Eqn. (5.6)"""

        def normal_ordering(_):
            return 2*args.U_ei_AbsSq_NO[0]*args.sigma_avg*args.N_T
        
        def inverted_ordering(_):
            return 2*args.U_ei_AbsSq_IO[0]*args.sigma_avg*args.N_T
        
        # Map orderings to indices
        ordering_index = {'NO': 0, 'IO': 1}
        if ordering not in ordering_index:
            raise ValueError("Invalid 'ordering' value. It should be 'NO' or 'IO'.")
        
        branches = [normal_ordering, inverted_ordering]
        pre = jax.lax.switch(ordering_index[ordering], branches, None)

        E_j = Ptolemy.E_j_from_E_e(E_e, args)
        dnj_dEj = jnp.array([
            Ptolemy.dndE_decay(E, m_j, m_i, m_x, tau_i, n0_i, args) for E in E_j
        ])

        return pre*dnj_dEj
    

    @staticmethod
    def dndE_Fermi_Dirac(nu_masses, p_start, p_stop, p_num, args):

        # Momentum range determined by input parameters
        p_FD = jnp.geomspace(p_start, p_stop, p_num)
        
        # Compute Fermi-Dirac values
        f_FD = Physics.Fermi_Dirac(p_FD, args)

        # Neutrino number per momentum interval
        dndp_FD = p_FD**2 * f_FD

        # Get corresponding energy range
        E_FD = Physics.Energy_Momentum_conversion(p_FD, nu_masses, 'p_to_E')

        # Get velocities via momentum-energy relation
        v_FD = p_FD / E_FD

        # Neutrino number per energy interval
        E_sqrt = jnp.sqrt(E_FD**2 - nu_masses[:,None]**2)
        dndE_FD = E_FD * E_sqrt * f_FD

        return p_FD, dndp_FD, E_FD, dndE_FD, v_FD


    @staticmethod
    def dGamma_CNB_dE_e_FD(nu_masses, ordering, nature, p_start, p_stop, p_num, args):

        def normal_ordering(_):
            return nature*args.U_ei_AbsSq_NO*args.sigma_avg*args.N_T
        
        def inverted_ordering(_):
            return nature*args.U_ei_AbsSq_IO*args.sigma_avg*args.N_T
        
        # Map orderings to indices
        ordering_index = {'NO': 0, 'IO': 1}
        if ordering not in ordering_index:
            raise ValueError("Invalid 'ordering' value. It should be 'NO' or 'IO'.")
        
        # Choose ordering and determine appropriate prefactors
        branches = [normal_ordering, inverted_ordering]
        pre = jax.lax.switch(ordering_index[ordering], branches, None)
        phase_space_factor = args.g_nu / (2 * args.Pi**2)

        # Get Fermi-Dirac energy spectra
        *_, E_FD, dndE_FD, _ = Ptolemy.dndE_Fermi_Dirac(
            nu_masses, p_start, p_stop, p_num, args)

        return E_FD, phase_space_factor * pre[:,None] * dndE_FD
    

    @staticmethod
    def dndE_simulation(halo_ID, nu_masses, args):

        # Load velocity data, units get taken care of in conversion to momenta
        v_sim = Data.load_neutrino_velocities(halo_ID)

        # Convert velocities to momenta
        p_sim, y_sim = Physics.velocities_to_momenta(v_sim, nu_masses, args)

        # Select neutrino with highest Fermi-Dirac value of each momentum batch
        p_back, y_init = Physics.filter_momenta(p_sim, y_sim, len(nu_masses), args)
        p_init = y_init*args.T_CNB

        # Compute Fermi-Dirac values of today, using p_back and Liouville's theorem
        f_today = Physics.Fermi_Dirac(p_back, args)

        # Neutrino number per momentum interval
        dndp_init = p_init**2 * f_today

        # Get corresponding energy range
        E_init = jnp.sqrt(p_init**2 + nu_masses[:, None]**2)

        # Neutrino number per energy interval
        E_sqrt = jnp.sqrt(E_init**2 - nu_masses[:, None]**2)
        dndE_init = E_init * E_sqrt * f_today

        return p_init, dndp_init, E_init, dndE_init
    

    @staticmethod
    def dGamma_CNB_dE_e_simulation(halo_ID, nu_masses, ordering, nature, args):

        def normal_ordering(_):
            return nature*args.U_ei_AbsSq_NO*args.sigma_avg*args.N_T
        
        def inverted_ordering(_):
            return nature*args.U_ei_AbsSq_IO*args.sigma_avg*args.N_T
        
        # Map orderings to indices
        ordering_index = {'NO': 0, 'IO': 1}
        if ordering not in ordering_index:
            raise ValueError("Invalid 'ordering' value. It should be 'NO' or 'IO'.")
        
        # Choose ordering and determine appropriate prefactors
        branches = [normal_ordering, inverted_ordering]
        pre = jax.lax.switch(ordering_index[ordering], branches, None)
        phase_space_factor = args.g_nu / (2 * args.Pi**2)

        # Get CNB simulation energy spectra
        *_, E_sim, dndE_sim = Ptolemy.dndE_simulation(
            halo_ID, nu_masses, args)

        return E_sim, phase_space_factor * pre[:,None] * dndE_sim


    @staticmethod
    def analyze_peak_for_X_range_resolution(data, points_in_FWHM):
        # Finding the peak
        peak_value = jnp.max(data)
        peak_idx = jnp.argmax(data)

        # Calculating the Full Width at Half Maximum (FWHM)
        half_max = peak_value / 2
        left_idx = jnp.where(data[:peak_idx] < half_max)[0][-1]
        right_idx = jnp.where(data[peak_idx:] < half_max)[0][0] + peak_idx
        FWHM = right_idx - left_idx

        # Determining the resolution for the linear x-axis
        total_range = len(data)
        resolution = int((total_range / FWHM) * points_in_FWHM) + 1

        return resolution, peak_idx, FWHM


    @staticmethod
    def analyze_peak_for_energy_step(data_x, data_y, points_in_FWHM):
        """
        Analyzes the peak in the spectrum to determine the energy step size for a given resolution.

        :param data_x: The original energy values (x-axis).
        :param data_y: The corresponding spectrum values (y-axis).
        :param points_in_FWHM: The number of points desired within the FWHM.
        :return: The necessary energy step size.
        """
        # Finding the peak
        peak_value = jnp.max(data_y)
        peak_idx = jnp.argmax(data_y)

        # Calculating the Full Width at Half Maximum (FWHM)
        half_max = peak_value / 2
        left_idx = jnp.where(data_y[:peak_idx] < half_max)[0][-1]
        right_idx = jnp.where(data_y[peak_idx:] < half_max)[0][0] + peak_idx

        # Energy values at FWHM
        E_FWHM_left = data_x[left_idx]
        E_FWHM_right = data_x[right_idx]

        # Determining the energy step size
        FWHM_energy_range = E_FWHM_right - E_FWHM_left
        energy_step = FWHM_energy_range / points_in_FWHM

        return energy_step, E_FWHM_left, E_FWHM_right


    @staticmethod
    def analyze_peak_and_determine_extended_resolution(data, points_in_FWHM, E_start_orig, E_stop_orig, E_min, E_max):
        """
        Determines the resolution for an extended linear energy range based on the FWHM of the original data.

        :param data: The array containing the data.
        :param points_in_FWHM: The number of points desired within the FWHM.
        :param E_start_orig: The starting energy of the original data range.
        :param E_stop_orig: The stopping energy of the original data range.
        :param E_min: The minimum energy of the extended range.
        :param E_max: The maximum energy of the extended range.
        :return: The number of points for the extended linear energy range.
        """
        # Finding the peak and FWHM in the original data
        peak_value = jnp.max(data)
        peak_idx = jnp.argmax(data)
        half_max = peak_value / 2
        left_idx = jnp.where(data[:peak_idx] < half_max)[0][-1]
        right_idx = jnp.where(data[peak_idx:] < half_max)[0][0] + peak_idx
        FWHM = right_idx - left_idx

        # Original energy range and step size
        original_range = E_stop_orig - E_start_orig
        step_size = original_range / FWHM

        # Number of points in the original range to achieve the desired resolution
        num_points_original = int((original_range / step_size) * points_in_FWHM)

        # Applying the same step size to the extended range
        extended_range = E_max - E_min
        num_points_extended = int(extended_range / step_size) + 1

        return num_points_extended, num_points_original, FWHM


    @staticmethod
    def gpr_interpolation(x_original, y_original, x_interpolated):
        """
        Interpolates irregularly spaced data using Gaussian Process Regression.
        """
        
        # Reshaping for compatibility with GPR
        x_original_reshaped = jnp.array(x_original).reshape(-1, 1)
        x_interpolated_reshaped = jnp.array(x_interpolated).reshape(-1, 1)

        # Matérn kernel with a specified nu
        kernel = Matern(
            nu=1.5, length_scale=1.0, length_scale_bounds=(1e-2, 1e2))

        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

        # Fitting the model
        gp.fit(x_original_reshaped, y_original)

        # Predicting the interpolated data
        y_interpolated, sigma = gp.predict(
            x_interpolated_reshaped, return_std=True)
        
        return y_interpolated, sigma
    

    @staticmethod
    @jax.jit
    def Gaussian_kernel(X_range, mu, sigma, args):

        # Normalization for Gaussian
        norm = jnp.sqrt(2*args.Pi)*sigma  

        return jnp.exp(-(X_range - mu)**2 / (2 * sigma**2)) / norm


    @staticmethod
    @jax.jit
    def Gaussian_convolution_1D(X_range, array_to_convolve, Gaussian_sigma, args):

        # X-range differences for the Gaussian in the convolution integral
        X_diff = X_range[:, None] - X_range[None, :]

        # Functions to be convolved
        f1 = array_to_convolve[:, None]
        g1 = Ptolemy.Gaussian_kernel(X_diff, Gaussian_sigma, args)

        # Convolution by integration
        array_convolved = trap(f1 * g1, x=X_range, axis=0)
        
        # # Find peak values
        # peak_original = jnp.max(array_to_convolve)
        # peak_convolved = jnp.max(array_convolved)

        # # Adjust, i.e. normalize, convolved spectrum based on peak values
        # adjustment_factor = peak_original / peak_convolved
        # array_convolved_normalized = array_convolved * adjustment_factor
        

        # # Find areas under curves
        # area_original = trap(array_to_convolve, x=X_range)
        # area_convolved = trap(array_convolved, x=X_range)

        # # # Adjust, i.e. normalize, convolved spectrum based on areas
        # adjustment_factor = area_original / area_convolved
        # array_convolved_normalized = array_convolved * adjustment_factor

        array_convolved_normalized = array_convolved

        return array_convolved_normalized


    @staticmethod
    @jax.jit
    def fourier_convolution(signal, kernel):
        """
        Convolve a signal with a kernel via Fourier Transforms.
        Assumes kernel is already prepared (centered and optionally padded).
        """
        # Shift the kernel to align the zero-frequency component (representing
        # the average) with the first array element
        shifted_kernel = jnp.fft.ifftshift(kernel)

        # indices = jnp.arange(shifted_kernel.size)
        # middle_index = shifted_kernel.size // 2
        # shifted_kernel = jnp.where(indices >= middle_index, 0.0, shifted_kernel)

        # Fourier Transform of the signal and the shifted kernel
        signal_fft = jnp.fft.fft(signal)
        kernel_fft = jnp.fft.fft(shifted_kernel, n=len(signal))

        # Convolution in the frequency domain
        convolved_fft = signal_fft * kernel_fft

        # Inverse Fourier Transform
        convolved_signal = jnp.fft.ifft(convolved_fft)

        # Taking the real part of the result and normalize
        normalized_signal = jnp.real(convolved_signal) / jnp.sum(kernel)

        return normalized_signal


    @staticmethod
    @jax.jit
    def ZhangZhang2018_overdensity(m_nu):
        return 76.5 * m_nu**2.21
    

@chex.dataclass
class Utils:

    @staticmethod
    def savgol_smooth(data, window_length=51, polyorder=3):
        """
        Apply Savitzky-Golay filter for smoothing.
        
        Parameters:
        -----------
        data : array-like
            The data to smooth
        window_length : int
            Length of the smoothing window. Must be odd and greater than polyorder.
        polyorder : int
            Order of the polynomial used for fitting. Must be less than window_length.
            
        Returns:
        --------
        smoothed_data : array-like
            The smoothed data array
        """
        # Convert from JAX array to NumPy if needed
        try:
            data_np = np.array(data)
        except:
            data_np = data
            
        # Ensure window_length is odd
        if window_length % 2 == 0:
            window_length += 1
            
        # Apply Savitzky-Golay filter
        smoothed_data = savgol_filter(data_np, window_length, polyorder)
        
        # Convert back to JAX array if needed
        try:
            return jnp.array(smoothed_data)
        except:
            return smoothed_data


    @staticmethod
    def make_logz_arr(zA, zB, num_points):
        return jnp.logspace(jnp.log10(zA+1), jnp.log10(zB+1), num_points) - 1


    @staticmethod
    @jax.jit
    def jax_interpolate(x_interp, x_points, y_points):

        # Find indices where x_interp is between x_points
        idx = jnp.argmax(x_points > x_interp) - 1

        # Handle edge case where x_interp is less than all x_points
        idx = jnp.where(idx < 0, 0, idx)  

        # Handle edge case for upper bound
        idx = jnp.where(idx >= len(x_points) - 1, len(x_points) - 2, idx)  

        # Compute the slope (dy/dx) for the interval
        dy_dx = (y_points[idx + 1] - y_points[idx]) / (x_points[idx + 1] - x_points[idx])

        # Calculate the interpolated value
        y_interp = y_points[idx] + dy_dx * (x_interp - x_points[idx])

        return y_interp
    

    @staticmethod
    @jax.jit
    def jax_interpolate_1D(x_interp, x_points, y_points):
        # Ensure x_interp is a 1D array
        x_interp = jnp.atleast_1d(x_interp)

        # Find indices where x_interp is between x_points
        idx = jnp.searchsorted(x_points, x_interp, side='right') - 1

        # Handle edge cases
        idx = jnp.clip(idx, 0, len(x_points) - 2)

        # Compute the slopes (dy/dx) for the intervals
        slopes = (y_points[1:] - y_points[:-1]) / (x_points[1:] - x_points[:-1])

        # Calculate the interpolated values
        y_interp = y_points[idx] + slopes[idx] * (x_interp - x_points[idx])

        return y_interp


    @staticmethod
    @jax.jit
    def vectorized_interpolate_1D(x_interp, x_points, y_points):
        # Ensure inputs are JAX arrays
        x_interp = jnp.asarray(x_interp)
        x_points = jnp.asarray(x_points)
        y_points = jnp.asarray(y_points)

        # Get the shape of the input arrays
        shape = x_points.shape[:-1]
        n_points = x_points.shape[-1]

        # Reshape arrays to 2D: (num_arrays, n_points)
        x_points_2d = x_points.reshape(-1, n_points)
        y_points_2d = y_points.reshape(-1, n_points)
        x_interp_2d = x_interp.reshape(-1, x_interp.shape[-1])

        # Find indices where x_interp is between x_points
        idx = jax.vmap(lambda a, v: jnp.clip(jnp.searchsorted(a, v, side='right') - 1, 0, n_points - 2))(x_points_2d, x_interp_2d)

        # Compute the slopes (dy/dx) for the intervals
        slopes = (y_points_2d[:, 1:] - y_points_2d[:, :-1]) / (x_points_2d[:, 1:] - x_points_2d[:, :-1])

        # Gather the required values
        idx_expanded = idx[:, None, :]
        y_left = jnp.take_along_axis(y_points_2d[:, None, :], idx_expanded, axis=-1)
        x_left = jnp.take_along_axis(x_points_2d[:, None, :], idx_expanded, axis=-1)
        slopes_interp = jnp.take_along_axis(slopes[:, None, :], idx_expanded, axis=-1)

        # Calculate the interpolated values
        y_interp = y_left + slopes_interp * (x_interp_2d[:, None, :] - x_left)

        # Reshape the result back to the original shape
        return y_interp.reshape(shape + x_interp.shape[-1:])


    @staticmethod
    def fill_1d_arrays_with_zeros_jax(array_list):
        # Find the maximum N among all arrays
        max_N = max(arr.shape[0] for arr in array_list)
        
        # Use list comprehension to pad arrays for the (N,) case
        filled_arrays = [jnp.pad(arr, (0, max_N - arr.shape[0]), 
                                mode='constant', constant_values=0)
                        for arr in array_list]
        
        # Stack the padded arrays into a single JAX array
        return jnp.stack(filled_arrays)


    @staticmethod
    def fill_2d_arrays_with_zeros_jax(array_list):

        # Find the maximum N among all arrays
        max_N = max(arr.shape[0] for arr in array_list)
        
        # Use list comprehension to pad arrays
        filled_arrays = [jnp.pad(arr, ((0, max_N - arr.shape[0]), (0, 0)), 
                        mode='constant', constant_values=0)
                        for arr in array_list]
        
        return jnp.array(filled_arrays)


    @staticmethod
    def y_fmt(value, tick_number):
        if value == 1e-6:
            return r'1+$10^{-6}$'
        elif value == 1e-5:
            return r'1+$10^{-5}$'
        elif value == 1e-4:
            return r'1+$10^{-4}$'
        elif value == 1e-3:
            return r'1+$10^{-3}$'
        elif value == 1e-2:
            return r'1+$10^{-2}$'
        elif value == 1e-1:
            return r'1+$10^{-1}$'
        elif value == 1e0:
            return r'1+$10^0$'
        elif value == 1e1:
            return r'1+$10^1$'
        

    @staticmethod
    def int_labels(x, pos):
        if x in [1, 10]:  # Only label these x values
            return f"{int(x)}"
        else:
            return ""
        

    @staticmethod
    @jax.jit
    def v_mag_to_xyz(v_mag, key):
        """
        Generate isotropic velocity components from magnitudes.
        """

        N = v_mag.shape[0]
        
        # Split the key for generating two sets of random numbers
        key1, key2 = jax.random.split(key)
        
        # Generate uniform random numbers
        phis = jax.random.uniform(key1, (N,), minval=0, maxval=2*jnp.pi)
        cos_thetas = 2.0 * jax.random.uniform(key2, (N,)) - 1.0
        
        # Calculate sin_thetas
        sin_thetas = jnp.sqrt(1.0 - cos_thetas**2)
        
        # Calculate x, y, z components
        x = v_mag * jnp.cos(phis) * sin_thetas
        y = v_mag * jnp.sin(phis) * sin_thetas
        z = v_mag * cos_thetas
        
        return jnp.column_stack([x, y, z])


    @staticmethod
    @jax.jit
    def generate_isotropic_vectors(v_mag, key):
        """
        Generate N isotropic 3D velocity vectors with the same magnitude.
        
        Parameters:
        -----------
        v_mag : float
            The magnitude for all generated vectors.
        key : jax.random.PRNGKey
            JAX random key for reproducibility.
        
        Returns:
        --------
        jnp.ndarray
            Array of shape (N, 3) containing randomly oriented vectors all with magnitude v_mag.
        """
        
        # Split the key for generating two sets of random numbers
        key1, key2 = jax.random.split(key)
        
        N = 2_000_001
        # N = 1_001

        # Generate uniform random numbers
        phis = jax.random.uniform(key1, (N,), minval=0, maxval=2*jnp.pi)
        cos_thetas = 2.0 * jax.random.uniform(key2, (N,)) - 1.0
        
        # Calculate sin_thetas
        sin_thetas = jnp.sqrt(1.0 - cos_thetas**2)
        
        # Calculate x, y, z components
        x = v_mag * jnp.cos(phis) * sin_thetas
        y = v_mag * jnp.sin(phis) * sin_thetas
        z = v_mag * cos_thetas
        
        return jnp.column_stack([x, y, z])


    @jax.jit
    def v_mag_to_xyz_biased_1D(v_mag, key, v_direction, cone_angle_rad=jnp.pi/6):
        """
        Generate velocity components biased toward a specific direction.
        
        Parameters:
        - v_mag: Array of velocity magnitudes
        - key: JAX random key
        - v_direction: Direction vector to bias toward (any magnitude)
        - cone_angle_rad: Half-angle of the cone in radians (default: 30 degrees)
        
        Returns:
        - Array of velocity vectors biased in the specified direction
        """
        N = v_mag.shape[0]
        
        # Normalize the direction vector
        v_direction_norm = v_direction / jnp.linalg.norm(v_direction)
        
        # Create a rotation matrix to align [0, 0, 1] with our direction vector
        # First, find the axis of rotation by cross product
        v_z = jnp.array([0.0, 0.0, 1.0])
        
        # Handle the special case where v_direction is parallel to z-axis
        parallel_z = jnp.allclose(jnp.abs(jnp.dot(v_direction_norm, v_z)), 1.0)
        
        # If direction is -z, use [1,0,0] as rotation axis
        rotation_axis = jnp.where(
            parallel_z,
            jnp.array([1.0, 0.0, 0.0]),
            jnp.cross(v_z, v_direction_norm)
        )
        
        # Normalize the rotation axis (if not parallel case)
        rotation_axis = jnp.where(
            parallel_z,
            rotation_axis,
            rotation_axis / jnp.linalg.norm(rotation_axis)
        )
        
        # Find the angle between [0,0,1] and v_direction
        cos_angle = jnp.dot(v_z, v_direction_norm)
        angle = jnp.arccos(jnp.clip(cos_angle, -1.0, 1.0))
        
        # Create the rotation matrix using Rodrigues' rotation formula
        K = jnp.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])
        
        # Rodrigues' rotation formula
        R = jnp.eye(3) + jnp.sin(angle) * K + (1 - jnp.cos(angle)) * jnp.matmul(K, K)
        
        # Special case handling for parallel to z-axis
        is_negative_z = parallel_z & (cos_angle < 0)
        R = jnp.where(
            is_negative_z,
            jnp.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0]
            ]),
            R
        )
        
        # Split the key for generating random angles
        key1, key2 = jax.random.split(key)
        
        # Generate random phi angles (around the cone axis)
        phis = jax.random.uniform(key1, (N,), minval=0, maxval=2*jnp.pi)
        
        # Generate random theta angles (cone spread)
        # Using a distribution that concentrates points within the cone_angle
        thetas = jax.random.uniform(key2, (N,), minval=0, maxval=1.0) * cone_angle_rad
        
        # Calculate points on a cone around the z-axis, then rotate to the target direction
        sin_thetas = jnp.sin(thetas)
        cos_thetas = jnp.cos(thetas)
        
        # Generate points on a unit sphere, biased toward z-axis within cone_angle
        x_local = jnp.cos(phis) * sin_thetas
        y_local = jnp.sin(phis) * sin_thetas
        z_local = cos_thetas
        
        # Stack these coordinates
        local_coords = jnp.column_stack([x_local, y_local, z_local])
        
        # Rotate these coordinates to be centered around our direction vector
        # Apply the rotation matrix to each point
        rotated_coords = jnp.einsum('ij,kj->ki', R, local_coords)
        
        # Scale by velocity magnitude
        velocities = rotated_coords * v_mag[:, jnp.newaxis]
        
        return velocities


    @jax.jit
    def v_mag_to_xyz_biased(
        v_mag, key, v_direction, cone_angle_rad=jnp.pi/6):
        """
        Generate velocity components biased toward a specific direction.
        
        Parameters:
        - v_mag: Either a single velocity magnitude or array of velocity magnitudes
        - key: JAX random key
        - v_direction: Direction vector to bias toward (any magnitude)
        - cone_angle_rad: Half-angle of the cone in radians (default: 30 degrees)
        
        Returns:
        - Array of velocity vectors biased in the specified direction
        """
        # Check if v_mag is a scalar or array
        is_scalar = jnp.isscalar(v_mag) or (hasattr(v_mag, 'shape') and len(v_mag.shape) == 0)
        
        # If v_mag is a scalar and N is provided, create an array of N identical values
        N = 100_001
        v_mag_array = jnp.ones(N) * v_mag if is_scalar else v_mag
        
        # Get the number of velocities to generate
        N_actual = N if is_scalar else v_mag_array.shape[0]
        
        # Normalize the direction vector
        v_direction_norm = v_direction / jnp.linalg.norm(v_direction)
        
        # Create a rotation matrix to align [0, 0, 1] with our direction vector
        # First, find the axis of rotation by cross product
        v_z = jnp.array([0.0, 0.0, 1.0])
        
        # Handle the special case where v_direction is parallel to z-axis
        parallel_z = jnp.allclose(jnp.abs(jnp.dot(v_direction_norm, v_z)), 1.0)
        
        # If direction is -z, use [1,0,0] as rotation axis
        rotation_axis = jnp.where(
            parallel_z,
            jnp.array([1.0, 0.0, 0.0]),
            jnp.cross(v_z, v_direction_norm)
        )
        
        # Normalize the rotation axis (if not parallel case)
        rotation_axis = jnp.where(
            parallel_z,
            rotation_axis,
            rotation_axis / jnp.linalg.norm(rotation_axis)
        )
        
        # Find the angle between [0,0,1] and v_direction
        cos_angle = jnp.dot(v_z, v_direction_norm)
        angle = jnp.arccos(jnp.clip(cos_angle, -1.0, 1.0))
        
        # Create the rotation matrix using Rodrigues' rotation formula
        K = jnp.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])
        
        # Rodrigues' rotation formula
        R = jnp.eye(3) + jnp.sin(angle) * K + (1 - jnp.cos(angle)) * jnp.matmul(K, K)
        
        # Special case handling for parallel to z-axis
        is_negative_z = parallel_z & (cos_angle < 0)
        R = jnp.where(
            is_negative_z,
            jnp.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0]
            ]),
            R
        )
        
        # Split the key for generating random angles
        key1, key2 = jax.random.split(key)
        
        # Generate random phi angles (around the cone axis)
        phis = jax.random.uniform(key1, (N_actual,), minval=0, maxval=2*jnp.pi)
        
        # Generate random theta angles (cone spread)
        # Using a distribution that concentrates points within the cone_angle
        thetas = jax.random.uniform(key2, (N_actual,), minval=0, maxval=1.0) * cone_angle_rad
        
        # Calculate points on a cone around the z-axis, then rotate to the target direction
        sin_thetas = jnp.sin(thetas)
        cos_thetas = jnp.cos(thetas)
        
        # Generate points on a unit sphere, biased toward z-axis within cone_angle
        x_local = jnp.cos(phis) * sin_thetas
        y_local = jnp.sin(phis) * sin_thetas
        z_local = cos_thetas
        
        # Stack these coordinates
        local_coords = jnp.column_stack([x_local, y_local, z_local])
        
        # Rotate these coordinates to be centered around our direction vector
        # Apply the rotation matrix to each point
        rotated_coords = jnp.einsum('ij,kj->ki', R, local_coords)
        
        # Scale by velocity magnitude
        if is_scalar:
            # If v_mag was a scalar, we can broadcast it directly
            velocities = rotated_coords * v_mag
        else:
            # Otherwise use the array version with broadcasting
            velocities = rotated_coords * v_mag_array[:, jnp.newaxis]
        
        return velocities


    @staticmethod
    def sim_vels_to_sorted_z0z4(sim_vels, m_nu_arr, merge_last_axes, args):

        # Convert velocities to momenta
        p_arr, y_arr = Physics.velocities_to_momenta_ND_halo_0th_axis(
        v_arr=sim_vels, 
        m_arr=m_nu_arr,
        args=args)

        # Split into arrays for z0 and z4
        p_z0 = p_arr[...,0]
        p_z4 = p_arr[...,-1]
        y_z0 = y_arr[...,0]
        y_z4 = y_arr[...,-1]

        if merge_last_axes:
            p_z0 = p_z0.reshape(*p_z0.shape[:-2], -1)
            p_z4 = p_z4.reshape(*p_z4.shape[:-2], -1)
            y_z0 = y_z0.reshape(*y_z0.shape[:-2], -1)
            y_z4 = y_z4.reshape(*y_z4.shape[:-2], -1)

        # Sort in ascending order of momentum array today
        ind = p_z0.argsort(axis=-1)
        p_z0_sort = jnp.take_along_axis(p_z0, ind, axis=-1)
        p_z4_sort = jnp.take_along_axis(p_z4, ind, axis=-1)
        y_z0_sort = jnp.take_along_axis(y_z0, ind, axis=-1)
        y_z4_sort = jnp.take_along_axis(y_z4, ind, axis=-1)
        
        return ind, p_z0_sort, p_z4_sort, y_z0_sort, y_z4_sort
    

    @staticmethod
    def sim_vels_to_sorted_z0z4_vec(sim_vels, m_nu_arr, merge_last_axes, args):
        # Convert velocities to momenta
        p_vec, p_mag, y = Physics.velocities_to_momentum_vec_ND_halo_0th_axis(
            v_arr=sim_vels,
            m_arr=m_nu_arr,
            args=args
        )

        # Split into arrays for z0 and z4
        p_vec_z0 = p_vec[..., 0, :]
        p_vec_z4 = p_vec[..., -1, :]
        p_mag_z0 = p_mag[..., 0]
        p_mag_z4 = p_mag[..., -1]
        y_z0 = y[..., 0]
        y_z4 = y[..., -1]

        if merge_last_axes:
            p_vec_z0 = p_vec_z0.reshape(*p_vec_z0.shape[:-3], -1, 3)
            p_vec_z4 = p_vec_z4.reshape(*p_vec_z4.shape[:-3], -1, 3)
            p_mag_z0 = p_mag_z0.reshape(*p_mag_z0.shape[:-2], -1)
            p_mag_z4 = p_mag_z4.reshape(*p_mag_z4.shape[:-2], -1)
            y_z0 = y_z0.reshape(*y_z0.shape[:-2], -1)
            y_z4 = y_z4.reshape(*y_z4.shape[:-2], -1)

        # Sort based on magnitude of momentum today
        ind = p_mag_z0.argsort(axis=-1)
        
        # Apply sorting to all arrays
        p_vec_z0_sort = jnp.take_along_axis(p_vec_z0, ind[..., None], axis=-2)
        p_vec_z4_sort = jnp.take_along_axis(p_vec_z4, ind[..., None], axis=-2)
        p_mag_z0_sort = jnp.take_along_axis(p_mag_z0, ind, axis=-1)
        p_mag_z4_sort = jnp.take_along_axis(p_mag_z4, ind, axis=-1)
        y_z0_sort = jnp.take_along_axis(y_z0, ind, axis=-1)
        y_z4_sort = jnp.take_along_axis(y_z4, ind, axis=-1)
        
        return ind, p_vec_z0_sort, p_vec_z4_sort, p_mag_z0_sort, p_mag_z4_sort, y_z0_sort, y_z4_sort
