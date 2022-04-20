
import sys, os
from datetime import datetime
import time

# arrays and data packages
import numpy as np
import re
import h5py

# astrophysics
from astropy import units as unit
from astropy import constants as const

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


SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



#############
### Units ###
#############
# region
eV     = 1.                         # Unit of energy: GeV
keV    = 1.0e3*eV
MeV    = 1.0e3*keV
GeV    = 1.0e3*MeV
TeV    = 1.0e3*GeV
erg    = TeV/1.602                  # erg
J      = 1.0e7*erg                  # joule
K      = 8.6e-5*eV                  # Kelvin

cm     = 5.0678e4/eV                # centi-meter
m      = 1.0e2*cm
km     = 1.0e3*m
pc     = 3.086e18*cm                # parsec
kpc    = 1.0e3*pc
Mpc    = 1.0e3*kpc

s      = 2.9979e10*cm               # second

kg     = J/m**2*s**2
gram   = kg/1000.
Msun   = 1.989e30*kg                # Mass of the Sun
G      = 6.674e-11*m**3/kg/s**2     # Gravitational constant

deg    = np.pi/180.0                # Degree
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
Omega_M = 0.315
Omega_L = 0.685
Omega_R = 2.473e-5 / h**2

T_CMB = 2.725*K
T_CNB = 1.95*K

Pi = np.pi
# endregion