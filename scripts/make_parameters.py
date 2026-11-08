"""Step 2: simulation parameters and initial conditions of a run.

Writes <run_dir>/sim_parameters.yaml and the arrays the simulation starts from:
neutrino masses and momenta, redshift steps z_int_steps and the matching time
variable s_int_steps, healpix directions (all_sky_angles.npy) and the initial
velocities of all neutrinos (initial_velocities.npy, shape (Npix, p_num, 3)).

Needs <run_dir>/box_parameters.yaml from scripts/make_box.py.

Usage: python scripts/make_parameters.py configs/<run>.yaml
"""

import healpy as hp
import numpy as np
import yaml

from cnb.config import config_from_cli
from cnb.physics import Physics
from cnb.units import Params

# CNB temperature used to define the momentum grid. This is the value all
# existing runs were made with (old shared/preface.py: no N_eff correction,
# T_CMB = 2.72548 K). Params.T_CNB is 0.36% higher (N_eff = 3.043).
T_CNB_SETUP = np.power(4/11, 1/3)*2.72548*Params.K


def make_parameters(
        sim_dir,
        nu_mass_start, nu_mass_stop, nu_mass_num, nu_sim_mass,
        specific_masses,
        p_start, p_stop, p_num, p_spacing,
        phis, thetas,
        init_x_dis,
        z_int_shift, z_int_stop, z_int_num,
        CPUs_precalculations, CPUs_simulations, memory_limit_GB,
        DM_in_cell_limit,
        Nside=None, Npix=None, pix_sr=None
    ):

    # Load simulation box parameters.
    with open(f'{sim_dir}/box_parameters.yaml', 'r') as file:
        box_params = yaml.safe_load(file)

    h = box_params['Cosmology']['h']
    Omega_M = box_params['Cosmology']['Omega_M']
    Omega_L = box_params['Cosmology']['Omega_L']


    ### ============================== ###
    ### Simulation control parameters. ###
    ### ============================== ###

    # Number of simulated neutrinos.
    neutrinos = len(thetas)*p_num

    # Neutrino masses in eV and kg.
    nu_mass_eV = nu_sim_mass*Params.eV
    nu_mass_kg = nu_mass_eV/Params.kg

    # Neutrino mass range
    neutrino_massrange_eV = np.geomspace(
        nu_mass_start, nu_mass_stop, nu_mass_num)*Params.eV

    # Ensure mass range contains specified masses
    for m_val in specific_masses:
        idx = np.abs(neutrino_massrange_eV - m_val).argmin()
        neutrino_massrange_eV[idx] = m_val

    # Neutrino momentum range. The *_1k runs (papers 1 and 2) used log spacing,
    # the SunMod_PSD* runs (paper 3) linear spacing.
    if p_spacing == "log":
        neutrino_momenta = np.geomspace(p_start*T_CNB_SETUP, p_stop*T_CNB_SETUP, p_num)
    else:
        neutrino_momenta = np.linspace(p_start*T_CNB_SETUP, p_stop*T_CNB_SETUP, p_num)

    # Logarithmic redshift spacing, and conversion to integration variable s.
    z_int_steps = np.geomspace(z_int_shift, z_int_stop+z_int_shift, z_int_num)
    z_int_steps -= z_int_shift
    s_int_steps = np.array([Physics.s_of_z(z, h, Omega_M, Omega_L) for z in z_int_steps])

    # For sphere of influence tests: Divide inclusion region into shells.
    DM_shell_edges = np.array([0,5,10,15,20,40,100])*100*Params.kpc

    # Multiplier for DM limit for each shell.
    shell_multipliers = np.array([1,3,6,9,12,15])


    ### ============================================ ###
    ### Create .yaml file for simulation parameters. ###
    ### ============================================ ###

    sim_parameters = {
        "neutrinos": neutrinos,
        "initial_haloGC_distance": init_x_dis,
        "neutrino_mass_start": nu_mass_start,
        "neutrino_mass_stop": nu_mass_stop,
        "neutrino_mass_num": nu_mass_num,
        "neutrino_simulation_mass_eV": nu_mass_eV,
        "neutrino_simulation_mass_kg": float(nu_mass_kg),
        "phis": len(phis),
        "thetas": len(thetas),
        "Nside": Nside,
        "Npix": Npix,
        "pix_sr": pix_sr,
        "momentum_start": p_start,
        "momentum_stop": p_stop,
        "momentum_num": p_num,
        "z_inegration_start": 0,
        "z_inegration_stop": z_int_stop,
        "z_inegration_num": z_int_num,
        "CPUs_precalculations": CPUs_precalculations,
        "CPUs_simulations": CPUs_simulations,
        "memory_limit_GB": memory_limit_GB,
        "DM_in_cell_limit": DM_in_cell_limit
    }

    with open(f'{sim_dir}/sim_parameters.yaml', 'w') as file:
        yaml.dump(sim_parameters, file)

    # Save arrays as .npy files.
    np.save(f'{sim_dir}/neutrino_massrange_eV.npy', neutrino_massrange_eV)
    np.save(f'{sim_dir}/neutrino_momenta.npy', neutrino_momenta)
    np.save(f'{sim_dir}/z_int_steps.npy', z_int_steps)
    np.save(f'{sim_dir}/s_int_steps.npy', s_int_steps)
    np.save(f'{sim_dir}/DM_shell_edges.npy', DM_shell_edges)
    np.save(f'{sim_dir}/shell_multipliers.npy', shell_multipliers)


    ### ================================ ###
    ### Initial velocities of neutrinos. ###
    ### ================================ ###

    # Convert momenta to initial velocity magnitudes, in units of [kpc/s].
    u_i = neutrino_momenta/nu_mass_eV / (Params.kpc/Params.s)

    # Each coord. pair gets whole momentum, i.e. velocity range.
    glat = np.deg2rad(thetas)
    glon = np.deg2rad(phis)
    uxs = np.array([u_i*np.cos(b)*np.cos(l) for b, l in zip(glat, glon)])
    uys = np.array([u_i*np.cos(b)*np.sin(l) for b, l in zip(glat, glon)])
    uzs = np.array([u_i*np.sin(b) for b in glat])
    u_i_array = np.stack((uxs, uys, uzs), axis=2)

    # Save the theta and phi angles as numpy arrays.
    np.save(f'{sim_dir}/all_sky_angles.npy', np.transpose((thetas, phis)))

    # note: sim goes backwards in time, hence minus sign
    np.save(f'{sim_dir}/initial_velocities.npy', -u_i_array)


if __name__ == "__main__":
    cfg = config_from_cli(__doc__)
    par = cfg["parameters"]

    # Determine phi and theta angles with healpy.
    Nside = int(par["healpix_nside"])  # power of 2
    Npix = 12 * Nside**2  # Number of pixels
    pix_sr = (4*np.pi)/Npix  # Pixel size  [sr]

    # Galactic coordinates.
    phi_angles, theta_angles = np.array(
        hp.pixelfunc.pix2ang(Nside, np.arange(Npix), lonlat=True))

    make_parameters(
        sim_dir=cfg["run_dir"],
        nu_mass_start=float(par["nu_mass_start"]),
        nu_mass_stop=float(par["nu_mass_stop"]),
        nu_mass_num=int(par["nu_mass_num"]),
        nu_sim_mass=float(par["nu_sim_mass"]),
        specific_masses=[float(m) for m in par["specific_masses"]],
        p_start=float(par["p_start"]),
        p_stop=float(par["p_stop"]),
        p_num=int(par["p_num"]),
        p_spacing=par.get("p_spacing", "linear"),
        phis=phi_angles,
        thetas=theta_angles,
        init_x_dis=float(par["init_x_dis"]),
        z_int_shift=float(par["z_int_shift"]),
        z_int_stop=float(par["z_int_stop"]),
        z_int_num=int(par["z_int_num"]),
        CPUs_precalculations=int(par["CPUs_precalculations"]),
        CPUs_simulations=int(par["CPUs_simulations"]),
        memory_limit_GB=int(par["memory_limit_GB"]),
        DM_in_cell_limit=int(par["DM_in_cell_limit"]),
        Nside=Nside,
        Npix=Npix,
        pix_sr=pix_sr)
