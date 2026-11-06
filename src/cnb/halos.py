"""NFW halo parameters and escape momenta."""


import numpy as np
import jax.numpy as jnp


class Halos():
    """NFW profile helpers and escape momenta for halos in a run."""


    @staticmethod
    def get_halo_params(sim_folder):

        # Load halo files
        halo_params = jnp.load(
            f"{sim_folder}/halo_batch_0.6-2.0x1e+12.0_Msun_params.npy")

        # Extract halo parameters
        Rvir = halo_params[:,0]  # in kpc
        Mvir = 10**halo_params[:,1]  # in Msun
        conc = halo_params[:,2]  # unitless

        return Rvir, Mvir, conc


    @staticmethod
    def rho_crit(z, args):

        H_squared = args.H0**2 * (args.Omega_M*(1.+z)**3 + args.Omega_L) 
        rho_crit = 3.*H_squared / (8.*args.Pi*args.G)

        return rho_crit


    @staticmethod
    def scale_density_NFW(c, z, args):
        """Eqn. (2) from arXiv:1302.0288. c=r_200/r_s."""
        numer = 200 * c**3
        denom = 3 * (np.log(1+c) - (c/(1+c)))
        delta_c = numer/denom

        rho_critical = Halos.rho_crit(z, args)

        return rho_critical*delta_c


    @staticmethod
    def compute_p_esc_1D(x_i, R_vir, R_s, rho_0, m_nu, args):
        
        # Calculate gravitational potential at coords. x_i.
        r = np.linalg.norm(x_i, axis=-1)
        m = np.minimum(r, R_vir)
        M = np.maximum(r, R_vir)
        prefactor = -4.*args.Pi*args.G*rho_0*R_s**2
        term1 = np.log(1. + (m/R_s)) / (r/R_s)
        term2 = (R_vir/M) / (1. + (R_vir/R_s))
        potential = prefactor * (term1 - term2)

        # Escape momentum formula from Ringwald & Wong (2004).
        p_esc = np.sqrt(2*np.abs(potential)) * m_nu
        y_esc = p_esc/args.T_CNB

        return p_esc, y_esc


    @staticmethod
    def get_p_esc_1D(sim_folder, x_i, m_nu_arr, args):

        # Extract parameters from halos used in simulation folder
        Rvir, Mvir, conc = Halos.get_halo_params(sim_folder)
        
        # Only select DM halos used in papers
        exclude_nums = jnp.array([
            20,  # halo with missing/broken snapshot info
            23,  # halo with anomalous number densities (~0s on almost all pixels)
            24,  # anomalous "compactified into 1 cell" DM halo
            25,  # anomalous "compactified into 1 cell" DM halo
        ])

        halo_nums = jnp.array(
            [x for x in range(1, 31) if x not in exclude_nums]) - 1  # subtract 1 for 0-based indexing
        
        Rvir = Rvir[halo_nums]
        Mvir = Mvir[halo_nums]
        conc = conc[halo_nums]

        # Using get_halo_params function, so x_200 values.
        Rs = Rvir/conc
        rho0 = Halos.scale_density_NFW(c=conc, z=0., args=args)

        # Compute escape momentum for each halo
        p_escs_l, y_escs_l = [], []
        for m in m_nu_arr:
            p_esc, y_esc = Halos.compute_p_esc_1D(
                x_i, 
                Rvir*args.kpc, 
                Rs*args.kpc, 
                rho0, m, args)
            p_escs_l.append(p_esc)
            y_escs_l.append(y_esc)

        return jnp.array(p_escs_l).flatten(), jnp.array(y_escs_l).flatten()


    @staticmethod
    def compute_p_esc(x_i, R_vir, R_s, rho_0, m_nu, args):
        """
        Parameters:
        x_i: [..., 3] coordinates
        R_vir: [n_halos] virial radii
        R_s: [n_halos] scale radii
        rho_0: [n_halos] scale densities
        m_nu: [n_masses] neutrino masses
        """
        # Add axes for broadcasting
        R_vir = R_vir[:, None]  # [n_halos, 1]
        R_s = R_s[:, None]      # [n_halos, 1]
        rho_0 = rho_0[:, None]  # [n_halos, 1]
        m_nu = m_nu[None, :]    # [1, n_masses]
        
        # Calculate gravitational potential at coords. x_i
        r = jnp.linalg.norm(x_i, axis=-1)  # [...] 
        m = jnp.minimum(r[..., None, None], R_vir)  # [..., n_halos, 1]
        M = jnp.maximum(r[..., None, None], R_vir)  # [..., n_halos, 1]
        
        prefactor = -4.*args.Pi*args.G*rho_0*R_s**2  # [n_halos, 1]
        term1 = jnp.log(1. + (m/R_s)) / (r[..., None, None]/R_s)  # [..., n_halos, 1]
        term2 = (R_vir/M) / (1. + (R_vir/R_s))  # [..., n_halos, 1]
        potential = prefactor * (term1 - term2)  # [..., n_halos, 1]

        # Escape momentum formula from Ringwald & Wong (2004)
        # Broadcasting: [..., n_halos, 1] * [1, n_masses] -> [..., n_halos, n_masses]
        p_esc = jnp.sqrt(2*jnp.abs(potential)) * m_nu
        y_esc = p_esc/args.T_CNB

        return p_esc, y_esc


    @staticmethod
    def get_p_esc(sim_folder, x_i, m_nu_arr, args):
        # Extract parameters from halos used in simulation folder
        Rvir, Mvir, conc = Halos.get_halo_params(sim_folder)
        
        # Only select DM halos used in papers
        exclude_nums = jnp.array([20, 23, 24, 25])
        halo_nums = jnp.array([x for x in range(1, 31) if x not in exclude_nums]) - 1
        
        Rvir = Rvir[halo_nums]  # [n_halos]
        Mvir = Mvir[halo_nums]  # [n_halos]
        conc = conc[halo_nums]  # [n_halos]

        # Using get_halo_params function, so x_200 values
        Rs = Rvir/conc  # [n_halos]
        rho0 = Halos.scale_density_NFW(c=conc, z=0., args=args)  # [n_halos]

        # Compute escape momentum for all halos and masses at once
        p_esc, y_esc = Halos.compute_p_esc(
            x_i, 
            Rvir*args.kpc, 
            Rs*args.kpc, 
            rho0, 
            m_nu_arr, 
            args)
        
        return p_esc, y_esc
