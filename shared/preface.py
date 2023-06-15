###############
### Imports ###
###############

import sys, os, shutil
from sys import getsizeof as gso
import psutil
import time
import glob
import random
import string
import yaml
import gc
import argparse
import pathlib
from itertools import chain, repeat, zip_longest
from memory_profiler import profile
from icecream import ic
import traceback
import math
from pynverse import inversefunc

# arrays and data packages
import numpy as np
import re
import h5py
from funcy import chunks
import pandas as pd

# astrophysics
from astropy import units as unit
from astropy import constants as const
import natpy as nat
import healpy as hp

# speed improvement
import numba as nb
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count

# scipy packages
from scipy.integrate import solve_ivp, quad, simpson
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.special import expit, zeta
import scipy.stats as stat

from scipy.stats import pearsonr
from scipy.optimize import brentq

# plotting
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from mpl_toolkits import mplot3d
from mpl_toolkits.axes_grid1 import make_axes_locatable
import colorcet as cc
from mycolorpy import colorlist as mcp
import imageio
from sklearn.neighbors import KernelDensity

import code
def raise_sys_exit():
    raise SystemExit
DONE = {"done": raise_sys_exit}

def debug(local_vars):
    try:
        code.interact(local=local_vars)
    except SystemExit:
        sys.exit()

# note: 1 line of code for debugging
# debug(DONE | locals())  # use done() to exit interactive window


#############
### Plots ###
#############

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('figure', figsize=(8, 8))
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


#############
### Units ###
#############

GB_UNIT = 1000*1024**2
Pi = np.pi
hc_val = (const.h/(2*Pi)*const.c).convert(unit.eV*unit.cm).value

eV     = 1.                         # Unit of energy: eV
meV    = 1.0e-3*eV
keV    = 1.0e3*eV
MeV    = 1.0e3*keV
GeV    = 1.0e3*MeV
TeV    = 1.0e3*GeV
erg    = TeV/1.602                  # erg
J      = 1.0e7*erg                  # joule
K      = 8.61732814974493e-5*eV     # Kelvin

cm     = (1/hc_val)/eV              # centi-meter
m      = 1.0e2*cm
km     = 1.0e3*m
pc     = 3.08567758128e18*cm        # parsec
kpc    = 1.0e3*pc
Mpc    = 1.0e3*kpc

# note: 
# Natural units defined via c=1, i.e. s/m = 299792458
s      = 2.99792458e10*cm           # second
yr     = 365*24*60*60*s
Gyr    = 1e9*yr
t0     = 13.787*Gyr
Hz     = 1.0/s

kg     = J/m**2*s**2
gram   = kg/1000.
Msun   = 1.98847e30*kg              # Mass of the Sun
G      = 6.6743e-11*m**3/kg/s**2    # Gravitational constant
Da     = 1.66053906660e-27*kg       # Dalton or atomic mass unit (u)

deg    = Pi/180.0                   # Degree
arcmin = deg/60.                    # Arcminute
arcsec = arcmin/60.                 # Arcsecond
sr     = 1.                         # Steradian


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

'''
# NFW parameters for Andromeda.
Mvir_AG  = 8.0e11*Msun                             # Virial mass
rho0_AG  = 3.89e6*(Msun/kpc**3)                    # density normalization
Rs_AG    = 21.8*kpc                                # scale radius 
Rvir_AG  = 244.7*kpc                               # virial radius

# Coordinates.
GLAT_AG = -21.573311                               # Galactic latitude
GLON_AG = 121.174322                               # Galactic longitude
DIST_AG = 0.784e3*kpc                              # Distance

# Translated to cartesian coordinates [kpc] in our setup (from fct.halo_pos).
X_AG    = np.array([632.29742673, -377.40315121, -288.27006757])
'''

# CMB temperature from Planck 2018.
T_CMB = 2.72548*K 
T_CNB = np.power(4/11, 1/3)*T_CMB

# Neutrino + antineutrino number density of 1 flavor in [1/cm**3],
# using the analytical expression for Fermions.
N0 = 2*zeta(3.)/Pi**2 *T_CNB**3 *(3./4.) /(1/cm**3)

# PMNS matrix elements for electron flavor.
U_ei_AbsSq = np.array([0.681, 0.297, 0.0222])  # |U_e1|^2, |U_e2|^2, |U_e3|^2

# Mass squared differences.
Del_m21_Sq = (8.6*meV)**2
Del_m3l_Sq = (50*meV)**2