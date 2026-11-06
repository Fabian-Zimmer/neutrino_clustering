"""Parameters and helpers for primordial temperature fluctuations of the CNB
(Zimmer, Franco Abellan & Ando 2024, arXiv:2407.14582)."""


import numpy as np
import healpy as hp
import jax.numpy as jnp
import chex

from cnb.units import Params


@chex.dataclass
class Primordial:

    ########################
    ### Class parameters ###
    ########################
    
    # Momenta of the Cl maps
    Cl_qs = np.geomspace(0.01, 100, 50)

    # Select index closest to desired q value
    q_select = 3.15
    # q_select = 0.01
    Cl_qi = np.abs(Cl_qs - q_select).argmin()

    # Define cutoff below which neutrinos are considered massless in CLASS
    massless_cutoff = 2e-5

    # Temperature perturbations, in micro Kelvin
    uK = 1e-6*Params.K    # micro Kelvin

    # Scaling factor to get correct unit/magnitude for the Cl's later
    Cl_scaling = 1e12

    # Multipole and wavemode related parameters
    l_min = 1
    l_max = 50
    ls = np.arange(l_min, l_max+1, 1)
    k_min, k_max = 1e-4, 1e-1  # 1/Mpc
    n_kmodes = 100
    k_magnitudes_full = np.logspace(np.log(k_min), np.log(k_max), n_kmodes, base=np.e)
    l_max_ncdm = 50

    # Momentum related parameters
    n_qbins = 50
    quad_strategy = 4
    q_max = 100

    # Integration stepsize
    int_step_size = 0.05

    # Resolution of healpix in our CNB simulations
    Nside = 8

    # Seed for np.random.seed() used in Tully & Zhang
    seed = 5

    z_cuts = np.round(np.geomspace(1e-1, 31+1e-1, 15)-0.1, 2)


    ###################################
    ### Parameters in certain units ###
    ###################################
    # note: T&Z functions need H0/(km/s/Mpc), T_CNB/K, kB/(eV/K), c/(km/s)

    H0 = Params.H0/(Params.km/Params.s/Params.Mpc)
    T_CNB = Params.T_CNB/Params.K
    kB = Params.kB/(Params.eV/Params.K)
    c = Params.c/(Params.km/Params.s)
    

    ############################
    ### Inflation parameters ###
    ############################

    As_default: float    = 2.215e-9
    pivot_default: float = 5e-2
    pivot: float         = 5e-4 
    ns: float            = 0.9619  # spectral tilt
    As: float            = As_default*(pivot/pivot_default)**(ns-1)  # primordial amplitude


    @staticmethod
    def generate_DeltaTs_seeds(
        m_arr, Cl_dir, Delta_dir, seeds, simdata, z_cut, phases, args):

        Deltas_z_cut_s_l = []
        Deltas_z0_s_l = []
        for seed in seeds:

            Deltas_z_cut_m_l = []
            Deltas_z0_m_l = []
            for m_Cl in m_arr:

                # Load Cl's for current neutrino mass
                if z_cut == 0.5:
                    Cls_z_cut_raw = jnp.load(f"{Cl_dir}/Cls_z=0p5_m={m_Cl}eV.npy")
                elif z_cut == 0.15:
                    Cls_z_cut_raw = jnp.load(f"{Cl_dir}/Cls_z=0p15_m={m_Cl}eV.npy")
                else:
                    Cls_z_cut_raw = jnp.load(f"{Cl_dir}/Cls_z={z_cut}_m={m_Cl}eV.npy")
                Cls_z0_raw = jnp.load(f"{Cl_dir}/Cls_z=0_m={m_Cl}eV.npy")
                # (q momentum bins, l multipoles) = (50, 20)

                # Add monopole
                Cl_z_cut = jnp.insert(Cls_z_cut_raw, 0, 0, axis=1)
                Cl_z0 = jnp.insert(Cls_z0_raw, 0, 0, axis=1)
                # (q momentum bins, l multipoles + 1) = (50, 21)

                Deltas_z_cut_q_l = []
                Deltas_z0_q_l = []
                # Create temp. flucts. skymaps from Cls for each q-momentum
                for seed_int, qi in enumerate(range(Primordial.n_qbins)):

                    # np.random.seed(seed)  # original
                    np.random.seed(seed + seed_int)
                    Tmap_z_cut = hp.sphtfunc.synfast(
                        Cl_z_cut[qi], nside=simdata.Nside, lmax=None, pol=False)
                    # np.random.seed(seed)  # original  #note: seed needs to reset
                    np.random.seed(seed + seed_int)
                    Tmap_z0 = hp.sphtfunc.synfast(
                        Cl_z0[qi], nside=simdata.Nside, lmax=None, pol=False)

                    Deltas_z_cut = Tmap_z_cut * args.muK / (args.T_CNB*(1+z_cut))
                    Deltas_z0 = Tmap_z0 * args.muK / (args.T_CNB)
                    # (Npix)

                    Deltas_z_cut_q_l.append(Deltas_z_cut)
                    Deltas_z0_q_l.append(Deltas_z0)

                Deltas_z_cut_m_l.append(Deltas_z_cut_q_l)
                Deltas_z0_m_l.append(Deltas_z0_q_l)

            Deltas_z_cut_s_l.append(Deltas_z_cut_m_l)
            Deltas_z0_s_l.append(Deltas_z0_m_l)

        Deltas_z_cut_matrix = jnp.array(Deltas_z_cut_s_l)
        Deltas_z0_matrix = jnp.array(Deltas_z0_s_l)

        if z_cut == 0.5:
            z_cut = "0p5"
        if z_cut == 0.15:
            z_cut = "0p15"

        if phases:
            if len(seeds) == 1:
                jnp.save(
                    f"{Delta_dir}/Delta_matrix_z{z_cut}_random_phases.npy", 
                    Deltas_z_cut_matrix)
                jnp.save(
                    f"{Delta_dir}/Delta_matrix_z0_random_phases.npy", 
                    Deltas_z0_matrix)
            else:
                jnp.save(
                    f"{Delta_dir}/Delta_matrix_z{z_cut}_seeds_random_phases.npy", 
                    Deltas_z_cut_matrix)
                jnp.save(
                    f"{Delta_dir}/Delta_matrix_z0_seeds_random_phases.npy", 
                    Deltas_z0_matrix)
        else:
            if len(seeds) == 1:
                jnp.save(
                    f"{Delta_dir}/Delta_matrix_z{z_cut}.npy", 
                    Deltas_z_cut_matrix)
                jnp.save(
                    f"{Delta_dir}/Delta_matrix_z0.npy", 
                    Deltas_z0_matrix)
            else:
                jnp.save(
                    f"{Delta_dir}/Delta_matrix_z{z_cut}_seeds.npy", 
                    Deltas_z_cut_matrix)
                jnp.save(
                    f"{Delta_dir}/Delta_matrix_z0_seeds.npy", 
                    Deltas_z0_matrix)

        return Deltas_z0_matrix, Deltas_z_cut_matrix
