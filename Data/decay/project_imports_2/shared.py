 ###############
### Imports ###
###############
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
from pynverse import inversefunc

import jax
import astropy.units as apu
import astropy.constants as apc
from astropy.cosmology import FlatLambdaCDM
import natpy as nat
import jax.random as random
import os
import glob
import re
import h5py

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import gridspec

# Special Imports for this file
from scipy.special import zeta, expit


#############
### Units ###
#############
GB_UNIT = 1000*1024**2
Pi = np.pi
hc_val = (apc.h/(2*Pi)*apc.c).convert(apu.eV*apu.cm).value

key = jax.random.PRNGKey(0)
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


#################
### Constants ###
#################
# note: should these be read from the simulation box?
h = 0.674
H0 = h * 100 * km/s/Mpc
Omega_R = 9.23640e-5
Omega_M = 0.3111
Omega_L = 1.-Omega_M-Omega_R

T_CMB = 2.72548*K
T_CNB = np.power(4/11, 1/3)*T_CMB

Z_LSS_CMB = 1100
Z_LSS_CNB = 6e9

N0 = 2*zeta(3.)/Pi**2 *T_CNB**3 *(3./4.) /(1/cm**3)

Del_m21_Sq = (8.6*meV)**2
Del_m3l_Sq = (50*meV)**2


#############
### Plots ###
#############
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('figure', figsize=(8, 8))
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('xtick', direction='in', top=True)  # x-axis ticks inside, top enabled
plt.rc('ytick', direction='in', right=True) # y-axis ticks inside, right enabled