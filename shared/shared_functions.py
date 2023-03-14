from shared.preface import *


def Fermi_Dirac(p):
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


def number_density(p0, p1, pix_sr=4*Pi):
    """Neutrino number density obtained by integration over initial momenta.

    Args:
        p0 (array): neutrino momentum today
        p1 (array): neutrino momentum at z_back (final redshift in sim.)

    Returns:
        array: Value of relic neutrino number density in (1/cm^3).
    """    

    g = 2.  # 2 degrees of freedom per flavour: particle and anti-particle
    
    # note: trapz integral method needs sorted (ascending) "x-axis" array.
    ind = p0.argsort(axis=-1)
    p0_sort = np.take_along_axis(p0, ind, axis=-1)
    p1_sort = np.take_along_axis(p1, ind, axis=-1)

    # Fermi-Dirac value with momentum at end of sim.
    FDvals = Fermi_Dirac(p1_sort)

    # Calculate number density.
    y = p0_sort**3 * FDvals
    x = p0_sort
    n_raw = np.trapz(y, np.log(x), axis=-1)

    # Multiply by constants and/or solid angles and convert to 1/cm**3.
    n_cm3 = pix_sr * g/((2*Pi)**3) * n_raw / (1/cm**3)

    return n_cm3


def velocity_to_momentum(sim_vels, m_arr):
    """
    Converts velocities (x,y,z from simulation) to 
    magnitude of momentum [eV] and ratio y=p/T_nu, according to desired
    target mass (and mass used in simulation).
    """

    # Magnitudes of velocity.
    mags_sim = np.sqrt(np.sum(sim_vels**2, axis=-1))
    mags_dim = np.repeat(np.expand_dims(mags_sim, axis=0), len(m_arr), axis=0)

    # Adjust neutrino target mass array dimensionally.
    m_dim = np.expand_dims(
        np.repeat(
            np.expand_dims(m_arr, axis=1), mags_dim.shape[1], axis=1),
        axis=2
    )

    # From velocity (magnitude) in kpc/s to momentum in eV.
    # p_dim = 1/np.sqrt(1-mags_dim**2) * mags_dim*(kpc/s) * m_dim
    p_dim = mags_dim*(kpc/s) * m_dim

    # p/T_CNB ratio.
    y = p_dim/T_CNB

    return p_dim, y


def y_fmt(value, tick_number):
    if value == 1e-3:
        return r'1+$10^{-3}$'
    elif value == 1e-2:
        return r'1+$10^{-2}$'
    elif value == 1e-1:
        return r'1+$10^{-1}$'
    elif value == 1e0:
        return r'1+$10^0$'
    elif value == 1e1:
        return r'1+$10^1$'


def delete_temp_data(path_to_wildcard_files):

    temp_files = glob.glob(path_to_wildcard_files, recursive=True)

    for f in temp_files:
        try:
            os.remove(f)
        except OSError:
            print("Error while deleting file (file not found")