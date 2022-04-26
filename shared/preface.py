###############
### Imports ###
###############
# region
import sys, os
from datetime import datetime
import time
import glob

# arrays and data packages
import numpy as np
import re
import h5py

# astrophysics
from astropy import units as unit
from astropy import constants as const
import natpy as nat

# speed improvement
import numba as nb  # jit, njit, vectorize
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import lru_cache

# symbolic integration
import sympy as sympy

# scipy packages
from scipy.integrate import solve_ivp, quad, simpson
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.special import expit

# plotting
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
# endregion



#############
### Plots ###
#############
# region
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# endregion



#############
### Units ###
#############
# region
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

cm     = (1/hc_val)/eV                  # centi-meter
m      = 1.0e2*cm
km     = 1.0e3*m
pc     = 3.08567758128e18*cm        # parsec
kpc    = 1.0e3*pc
Mpc    = 1.0e3*kpc

#NOTE: Natural units defined via c=1, i.e. s/m = 299792458
s      = 2.99792458e10*cm           # second
yr     = 365*24*60*60*s
Gyr    = 1e9*yr
t0     = 13.787*Gyr
Hz     = 1.0/s

kg     = J/m**2*s**2
gram   = kg/1000.
Msun   = 1.98847e30*kg              # Mass of the Sun
G      = 6.6743e-11*m**3/kg/s**2   # Gravitational constant
Da     = 1.66053906660e-27*kg       # Dalton or atomic mass unit (u)

deg    = Pi/180.0                   # Degree
arcmin = deg/60.                    # Arcminute
arcsec = arcmin/60.                 # Arcsecond
sr     = 1.                         # Steradian
# endregion



#################
### Constants ###
#################
# region
h = 0.674
H0 = h * 100 * km/s/Mpc
Omega_R = 9.23640e-5  # not used in simulation
Omega_M = 0.3111
Omega_L = 1.-Omega_M  # since we don't use Omega_R

T_CMB = 2.725*K
T_CNB = 1.95*K

Pi = np.pi


### NFW parameters today - Mertsch et al. (2020)
Mvir_NFW  = 2.03e12*Msun                           # Virial mass
rho0_NFW  = 1.06e7*(Msun/kpc**3.)                  # density normalization
Rs_NFW   = 19.9*kpc                                # scale radius 
Rvir_NFW = 333.5*kpc                               # virial radius
# endregion



######################
### Control Center ###
######################
# region
NU_MASS = 0.03*eV
NU_MASS_KG = NU_MASS/kg
NU_MASSES = np.array([0.01,0.05,0.1,0.3])*eV
N0 = 112  # standard neutrino number density in [1/cm**3]

PHIs = 10
THETAs = 10
Vs = 200
NR_OF_NEUTRINOS = PHIs*THETAs*Vs

LOWER = 0.01*T_CNB
UPPER = 100.*T_CNB
MOMENTA = np.geomspace(LOWER, UPPER, Vs)


# Redshift integration parameters
#NOTE: Linearly spaced, denser for late times (closer to today)
late_steps = 200
early_steps = 100
Z_START, Z_STOP, Z_AMOUNT = 0., 4., late_steps+early_steps
z_late = np.linspace(0,2,200)
z_early = np.linspace(2.01,4,100)
ZEDS = np.concatenate((z_late, z_early))

# Control if simulation runs forwards (+1) or backwards (-1) in time. 
TIME_FLOW = -1
HALOS = 'OFF'

SOLVER = 'RK23'
# endregion