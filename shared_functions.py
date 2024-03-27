import shared 

####################
### Calculations ###
####################

def CALC_Hubble_rate_1(z):
    # Hubble rate in "numerical" units.
  #  return H0*np.sqrt(Omega_R*(1+z)**4 + Omega_M*(1+z)**3 + Omega_L)
   return H0*np.sqrt(Omega_M*(1+z)**3 + Omega_L )

def CALC_Hubble_rate(z):
    # Hubble rate in "numerical" units.
   # return H0*np.sqrt(Omega_R*(1+z)**4 + Omega_M*(1+z)**3 + Omega_L)
 #  return H0*np.sqrt(Omega_M*(1+z)**3 + Omega_L)
   return H0*np.sqrt(Omega_R*(1+z)**4 + Omega_M*(1+z)**3)

def CALC_Hubble_rate_2(z):
    # Hubble rate in "numerical" units.
   # return H0*np.sqrt(Omega_R*(1+z)**4 + Omega_M*(1+z)**3 + Omega_L)
 #  return H0*np.sqrt(Omega_M*(1+z)**3 + Omega_L)
   return H0*np.sqrt(Omega_R*(1+z)**6 + Omega_M*(1+z)**5+Omega_L*(1+z)**2)

def CALC_Fermi_Dirac(p):
    """Fermi-Dirac phase-space distribution for CNB neutrinos. 
    Zero chem. potential and temp. T_CNB (CNB temp. today). 

    Args:
        p (array): magnitude of momentum, must be in eV!

    Returns:
        array: Value of Fermi-Dirac distr. at p.
    """

    # Function expit from scipy equivalent to 1/(np.exp(-X)+1).
    # (thus the minus sign)
    return expit(-p/T_CNB) 


############
### Data ###
############

def DATA_delete_files(path_to_wildcard_files):

    temp_files = glob.glob(path_to_wildcard_files, recursive=True)

    for f in temp_files:
        try:
            os.remove(f)
        except OSError:
            print("Error while deleting file (file not found")


'''
def NFW_profile(r, rho_0, r_s):
    """NFW density profile.

    Args:
        r (array): radius from center
        rho_0 (array): normalisation 
        r_s (array): scale radius

    Returns:
        array: density at radius r
    """    

    rho = rho_0 / (r/r_s) / np.power(1.+(r/r_s), 2.)

    return rho


def rho_crit(z):
    """Critical density of the universe as a function of redshift, assuming
    matter domination, only Omega_m and Omega_Lambda in Friedmann equation. See 
    notes for derivation.

    Args:
        z (array): redshift

    Returns:
        array: critical density at redshift z
    """    
    
    H_squared = H0**2 * (Omega_M*(1.+z)**3 + Omega_L) 
    rho_crit = 3.*H_squared / (8.*Pi*G)

    return np.float64(rho_crit)
'''