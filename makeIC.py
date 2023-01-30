###############################################################################
 # This file is part of SWIFT.
 # Copyright (c) 2021 Camila Correa (camila.correa@uva.nl)
 #
 # This program is free software: you can redistribute it and/or modify
 # it under the terms of the GNU Lesser General Public License as published
 # by the Free Software Foundation, either version 3 of the License, or
 # (at your option) any later version.
 # 
 # This program is distributed in the hope that it will be useful,
 # but WITHOUT ANY WARRANTY; without even the implied warranty of
 # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 # GNU General Public License for more details.
 # 
 # You should have received a copy of the GNU Lesser General Public License
 # along with this program.  If not, see <http://www.gnu.org/licenses/>.
 # 
 ##############################################################################

import h5py
import sys
import numpy as np
from scipy import integrate, interpolate

def function(x, M, a):
    # x specific energy
    G = 4.30091e-6 # km^2/s^2 kpc Msun^-1
    q = np.sqrt((a * x)/(G * M))
    vg = np.sqrt(G * M / a)
    f = M / (8. * np.sqrt(2.) * np.pi**3 * a**3 * vg**3 * (1.-q**2)**(5./2.))
    f *= (3. * np.arcsin(q) + q * np.sqrt(1.-q**2) * (1. - 2*q**2) * (8. * q**4 - 8. * q**2 - 3.))
    return f

# Generates a swift IC file #
boxsize = 1000 # some size
center = boxsize / 2.
numPart = 64**3
L = numPart**(1./3.)
eta = 1.2348          # 48 ngbs with cubic spline kernel
M = 1e12 #Msun - halo mass
c = 9
rhocrit = 2.7754e11 * 0.677 ** 2 / (1e3)**3  # Msun / kpc^3
R200 = (M / (200. * rhocrit * 4 * np.pi / 3))**(1./3.)
a = R200 / c #kpc - scale radius
part_mass = M/numPart #Msun

# Set units
unit_length_in_cgs = 3.085678e21
unit_mass_in_cgs = 1.98848e43
unit_time_in_cgs = 3.085678e16
unit_current_in_cgs = 1
unit_temperature_in_cgs = 1

# Output File
fileName = "HernquistHalo.hdf5"
file = h5py.File(fileName, 'w')

# Header
grp = file.create_group("/Header")
grp.attrs["BoxSize"] = boxsize
grp.attrs["NumPart_Total"] =  [0, numPart, 0, 0, 0, 0]
grp.attrs["NumPart_Total_HighWord"] = [0, 0, 0, 0, 0, 0]
grp.attrs["NumPart_ThisFile"] = [0, numPart, 0, 0, 0, 0]
grp.attrs["Time"] = 0.0
grp.attrs["NumFilesPerSnapshot"] = 1
grp.attrs["MassTable"] = [0.0, part_mass, 0.0, 0.0, 0.0, 0.0]
grp.attrs["Flag_Entropy_ICs"] = 0
grp.attrs["Dimension"] = 3

#Units
grp = file.create_group("/Units")
grp.attrs["Unit length in cgs (U_L)"] = unit_length_in_cgs
grp.attrs["Unit mass in cgs (U_M)"] = unit_mass_in_cgs
grp.attrs["Unit time in cgs (U_t)"] = unit_time_in_cgs
grp.attrs["Unit current in cgs (U_I)"] = unit_current_in_cgs
grp.attrs["Unit temperature in cgs (U_T)"] = unit_temperature_in_cgs

# Create data

# masses
mass = np.ones(numPart) * part_mass / 1e10

# positions
x = np.random.uniform(0, 0.98, numPart)
r = a / ( np.sqrt(1./x)-1. )
theta = np.random.uniform(0, 1, numPart) * 2. * np.pi # uniform [0,2pi)
u = 2. * np.random.uniform(0, 1, numPart) - 1 #uniform [-1,1)

pos = np.zeros((numPart,3))
pos[:,0] =  r * np.cos(theta) * np.sqrt(1. - u**2)
pos[:,1] =  r * np.sin(theta) * np.sqrt(1. - u**2)
pos[:,2] =  r * u

radius = np.sqrt(pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2)

# centering
pos[:,0] += center
pos[:,1] += center
pos[:,2] += center

# velocities
G = 4.30091e-6 # km^2/s^2 kpc Msun^-1
P = G * M /(r + a) #Potential energy

# Generate distribution function
E = np.linspace( 0, int(np.max(P)), 1000)
f = function(E, M, a)

# Make sure no nans
E = E[~np.isnan(f)]
f = f[~np.isnan(f)]

# Choose E from distribution function
n = numPart
Y = np.random.uniform(0, 1, numPart)
ans = np.ones(n)
for i in range(numPart):
    newE = E[E < P[i]] #newE goes to maxE(R)=P(R)
    F = integrate.cumtrapz(f[E<P[i]] * np.sqrt(P[i]-newE),newE, initial=0) #cummulative distribution
    newE = newE[~np.isnan(F)]
    F = F[~np.isnan(F)]
    ans[i] = np.interp(Y[i]*F[-1], F, newE) #find at which energy F=rand


abs_random_v = np.sqrt( 2. * (P-ans)) #velocity magnitude

theta = np.random.uniform(0, 1, numPart) * 2. * np.pi # uniform [0,2pi)
u = 2. * np.random.uniform(0, 1, numPart) - 1 #uniform [-1,1)

vel = np.zeros((numPart,3))
vel[:,0] =  abs_random_v * np.cos(theta) * np.sqrt(1. - u**2)
vel[:,1] =  abs_random_v * np.sin(theta) * np.sqrt(1. - u**2)
vel[:,2] =  abs_random_v * u

# Particle group
grp = file.create_group("/PartType1")
ds = grp.create_dataset('Coordinates', (numPart, 3), 'd', data=pos)
ds = grp.create_dataset('Velocities', (numPart, 3), 'f', data=vel)
ds = grp.create_dataset('Masses', (numPart,1), 'f', data=mass)

l = (4. * np.pi * radius**2 / numPart)**(1./3.) #local mean inter-particle separation
h = np.full((numPart, ), eta * l)
ds = grp.create_dataset('SmoothingLength', (numPart,1), 'f', data=h)

ids = np.linspace(0, numPart, numPart, endpoint=False).reshape((numPart,1))
ds = grp.create_dataset('ParticleIDs', (numPart, 1), 'L')
ds[()] = ids + 1


file.close()

