from shared.preface import *


####################################
### Functions used in simulation ###
####################################
# region

# @nb.njit
def c_vir_avg(z, M_vir):
    """Intermediate/helper function for c_vir function below."""

    # Functions for eqn. (5.5) in Mertsch et al. (2020), from their Ref. [40].
    a_of_z = 0.537 + (0.488)*np.exp(-0.718*np.power(z, 1.08))
    b_of_z = -0.097 + 0.024*z

    # Calculate average c_vir.
    arg_in_log = (M_vir / (1.e12 / h * Msun))
    c_vir_avg = np.power(a_of_z + b_of_z*np.log10(arg_in_log), 10.)

    return c_vir_avg
    

# @nb.njit
def c_vir(z, M_vir):
    """Concentration parameter defined as r_vir/r_s, i.e. the ratio of virial 
    radius to the scale radius of the halo according to eqn. 5.5 of 
    Mertsch et al. (2020). 

    Args:
        z (array): redshift
        M_vir (float): virial mass, treated as fixed in time

    Returns:
        array: concentration parameters at each given redshift [dimensionless]
    """

    # The "beta" in eqn. (5.5) is obtained from c_vir_avg(0, M_vir)
    # and c_vir(0, M_vir) (c0_vir variable below) and the values in Table 1.
    # Also see Methods section of Zhang & Zhang (2018) .
    c0_vir = Rvir_NFW / Rs_NFW 
    beta = c0_vir / c_vir_avg(0, M_vir)
    print(beta)

    c = beta * c_vir_avg(z, M_vir)

    return c


# @nb.njit
def rho_crit(z):
    """Critical density of the universe as a function of redshift, assuming
    matter domination, only Omega_m and Omega_Lambda in Friedmann equation. See 
    notes for derivation.

    Args:
        z (array): redshift

    Returns:
        array: critical density at redshift z [Msun/kpc**3]
    """    
    
    H_squared = H0**2 * (Omega_M*(1.+z)**3 + Omega_L) 
    rho_crit = 3.*H_squared / (8.*Pi*G)

    return rho_crit / (Msun/kpc**3)


# @nb.njit
def Omega_M_z(z):
    """Matter density parameter as a function of redshift, assuming matter
    domination, only Omega_M and Omega_L in Friedmann equation. See notes
    for derivation.

    Args:
        z (array): redshift

    Returns:
        array: matter density parameter at redshift z [dimensionless]
    """    

    Omega_M_of_z = (Omega_M*(1.+z)**3) / (Omega_M*(1.+z)**3 + Omega_L)

    return Omega_M_of_z


# @nb.njit
def Delta_vir(z):
    """Function as needed for their eqn. (5.7).

    Args:
        z (array): redshift

    Returns:
        array: value as specified just beneath eqn. (5.7) [dimensionless]
    """    

    Delta_vir = 18.*Pi**2 + 82.*(Omega_M_z(z)-1.) - 39.*(Omega_M_z(z)-1.)**2

    return Delta_vir


# @nb.njit
def R_vir(z, M_vir):
    """Virial radius according to eqn. 5.7 in Mertsch et al. (2020).

    Args:
        z (array): redshift
        M_vir (float): virial mass

    Returns:
        array: virial radius [kpc]
    """    

    R_vir = np.power(3.*M_vir / (4.*Pi*Delta_vir(z)*rho_crit(z)), 1./3.)

    return R_vir


# @nb.njit
def scale_radius(z, M_vir):
    """Scale radius of NFW halo.

    Args:
        z (array): redshift
        M_vir (float): virial mass

    Returns:
        arrat: scale radius [kpc]
    """    
    
    r_s = R_vir(z, M_vir) / c_vir(z, M_vir)

    return r_s

# endregion



######################
### Main functions ###
######################
# region

def s_of_z(z):
    """Convert redshift to time variable s with eqn. 4.1 in Mertsch et al.
    (2020), keeping only Omega_m0 and Omega_Lambda0 in the Hubble eqn. for H(z).

    Args:
        z (float): redshift

    Returns:
        float: time variable s (in [seconds] if 1/H0 factor is included)
    """    

    def s_integrand(z):        

        # We need value of H0 in units of 1/s.
        H0_val = H0/(1/s)
        a_dot = np.sqrt(Omega_M*(1.+z)**3 + Omega_L)/(1.+z)*H0_val
        s_int = 1./a_dot

        return s_int

    s_of_z, _ = quad(s_integrand, 0., z)

    return np.float64(s_of_z)


def Fermi_Dirac(p):
    """Fermi-Dirac phase-space distribution for CNB neutrinos. 
    Zero chem. potential and temp. T_CNB (CNB temp. today). 

    Args:
        p (array): magnitude of momentum, must be in eV!

    Returns:
        array: Value of Fermi-Dirac distr. at p.
    """

    # Function expit from scipy equivalent to 1/(np.exp(-X)+1).
    # (thus the miunus sign)
    return expit(-p/T_CNB) 


def number_density(p0, p1):
    """Neutrino number density obtained by integration over initial momenta.

    Args:
        p0 (array): neutrino momentum today
        p1 (array): neutrino momentum at z_back (final redshift in sim.)

    Returns:
        array: Value of relic neutrino number density.
    """    

    g = 2.  # 2 d.o.f.: flavour and anti-particle/particle 
    
    #NOTE: trapz integral method needs sorted (ascending) arrays
    ind = p0.argsort()
    p0_sort, p1_sort = p0[ind], p1[ind]

    # Fermi-Dirac value with momentum at end of sim.
    FDvals = Fermi_Dirac(p1_sort)  #! needs p in [eV]

    # Convert initial momentum p0 from [eV] to [kg*m/s]
    p0_SI = (p0_sort.to(unit.J) / const.c).to(unit.kg*unit.m/unit.s)
    p0_SI = (p0_sort.to(unit.J) / const.c).to(unit.kg*unit.m/unit.s)

    # Calculate number density.
    y = p0_SI.value**2. * FDvals
    x = p0_SI.value
    n_raw = np.trapz(y, x)

    # Reintroduce "invisible" planck constant h to get 1/m**3
    #NOTE: leftover constants are g*4*np.pi
    n_m3 = n_raw / const.h.value**3 * g*4*np.pi

    # To 1/cm**3
    n_cm3 = (n_m3/unit.m**3).to(1/unit.cm**3)

    return n_cm3

# endregion