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
import pandas as pd
import re
import h5py
from funcy import chunks

# astrophysics
from astropy import units as unit
from astropy import constants as const
import natpy as nat
import healpy as hp

# speed improvement
import numba as nb
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool

# scipy packages
from scipy.integrate import solve_ivp, quad, simpson
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.special import expit, zeta
import scipy.stats as stats

# plotting
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from mpl_toolkits import mplot3d
import colorcet as cc
# import imageio

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
X_VC     = np.array([-4289.63477282, 1056.51861602, 15895.27621304])

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

#? for now temps. here
T_CMB = 2.72548*K
T_CNB = np.power(4/11, 1/3)*T_CMB

# Neutrino + antineutrino number density of 1 flavor in [1/cm**3],
# using the analytical expression for Fermions.
N0 = 2*zeta(3.)/Pi**2 *T_CNB**3 *(3./4.) /(1/cm**3)


######################################
### Discrete simulation parameters ###
######################################

# class PRE:

#     def __init__(
#         self, sim, z0_snap=0, z4_snap=0, 
#         sim_dir=None, sim_ver=None, DM_lim=None,
#         phis=0, thetas=0, vels=0, 
#         pre_CPUs=0, sim_CPUs=0, mem_lim_GB=0,
#         MW_HALO=False, VC_HALO=False, AG_HALO=False
#         ):

#         # File management.
#         self.SIM = sim
#         self.MW_HALO = MW_HALO
#         self.VC_HALO = VC_HALO
#         self.AG_HALO = AG_HALO
#         self.HALOS = 'MW'*MW_HALO + '+VC'*VC_HALO + '+AG'*AG_HALO

#         if sim_ver is None:
#             self.OUT_DIR = f'{os.getcwd()}/{sim}'
#         else:
#             self.SIM_DIR = f'{sim_dir}/{sim}/{sim_ver}'
#             self.OUT_DIR = f'{os.getcwd()}/{sim}/{sim_ver}'

#         # Initial conditions for neutrinos.
#         self.PHIs = phis
#         self.THETAs = thetas
#         self.Vs = vels
#         if isinstance(phis, int):
#             self.NUS = phis*thetas*vels
#         else:
#             self.NUS = len(phis)*len(thetas)*vels
#         self.LOWER = 0.01*T_CNB
#         self.UPPER = 400.*T_CNB
#         self.MOMENTA = np.geomspace(self.LOWER, self.UPPER, vels)

#         # Simulation parameters.
#         self.SIM_CPUs = sim_CPUs

#         if sim_ver is None:
#             ...
#         else:
#             self.PRE_CPUs = pre_CPUs
#             self.MEM_LIM_GB = mem_lim_GB
#             self.DM_LIM = DM_lim
#             self.Z0_INT = int(z0_snap)
#             self.Z4_INT = int(z4_snap)
#             self.Z0_STR = f'{z0_snap:04d}'
#             self.Z4_STR = f'{z4_snap:04d}'
#             snaps = np.arange(z4_snap, z0_snap+1)
#             zeds = np.zeros(len(snaps))
#             nums = []

#             #! No Halo tests.
#             # self.OUT_DIR = f'{os.getcwd()}/{sim}/{sim_ver}_noHalo'

#             # Store parameters unique to each simulation box.
#             for j, i in enumerate(snaps):
#                 snap_zi = f'{i:04d}'
#                 snap_z0 = f'{snaps[-1]:04d}'
#                 nums.append(snap_zi)

#                 with h5py.File(f'{self.SIM_DIR}/snapshot_{snap_zi}.hdf5') as snap:
                    
#                     # Store redshifts.
#                     zeds[j] = snap['Cosmology'].attrs['Redshift'][0]

#                     if snap_zi == snap_z0:

#                         # DM mass.
#                         dm_mass = snap['PartType1/Masses'][:]*1e10*Msun
#                         self.DM_SIM_MASS = np.unique(dm_mass)[0]

#                         #! No Halo tests.
#                         # self.DM_SIM_MASS = 0.

#                         # Gravity smoothening length.
#                         sl = snap['GravityScheme'].attrs[
#                             'Maximal physical DM softening length (Plummer equivalent) [internal units]'
#                         ][0]
#                         self.SMOOTH_L = sl*1e6*pc

#             self.ZEDS_SNAPS = np.asarray(zeds)
#             self.NUMS_SNAPS = np.asarray(nums)