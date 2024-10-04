from Shared.shared import *
from Shared.specific_CNB_sim import *
import gc

# sim_name = f"Dopri5_1k"
# sim_folder = f"sim_output_data/{sim_name}"

# sim_name = f"SunMod_1k"
# sim_folder = f"sim_output/{sim_name}"

sim_name = f"high_res"
sim_folder = f"sim_output/{sim_name}"

fig_folder = f"figures_local/{sim_name}"
Cl_folder = f"Shared/Cls"
nu_m_range = jnp.load(f"{sim_folder}/neutrino_massrange_eV.npy")
nu_m_picks = jnp.array([0.01, 0.05, 0.1, 0.2, 0.3])*Params.eV
simdata = SimData(sim_folder)


# List to store number densities for each day
days_nrs_l = []
n_nu_days_l = []

for day in range(365):

    ### ==================== ###
    ### Solar modulation sim ###
    ### ==================== ###

    # Load velocities for current day
    # Insert dimension at beginning to account for multiple halos later
    file_path = f"{sim_folder}/vectors_day{day+1}.npy"

    if os.path.exists(file_path):
        day_vels = jnp.load(
            f"{sim_folder}/vectors_day{day+1}.npy")[..., 3:][None, ...]
        print(f"Day {day+1}/365")
        days_nrs_l.append(day+1)
            
        # Convert to momentum arrays (p and y)
        p_day, y_day = Physics.velocities_to_momenta_ND_halo_0th_axis(
            v_arr=day_vels,
            m_arr=nu_m_picks,
            args=Params())
        del day_vels

        p_day_z0 = p_day[...,0]
        p_day_z4 = p_day[...,-1]
        y_day_z0 = y_day[...,0]
        y_day_z4 = y_day[...,-1]
        del p_day, y_day

        # Sort in ascending order of momentum array today
        ind = p_day_z0.argsort(axis=-1)
        p_day_z0_sort = jnp.take_along_axis(p_day_z0, ind, axis=-1)
        p_day_z4_sort = jnp.take_along_axis(p_day_z4, ind, axis=-1)
        y_day_z0_sort = jnp.take_along_axis(y_day_z0, ind, axis=-1)
        y_day_z4_sort = jnp.take_along_axis(y_day_z4, ind, axis=-1)
        del p_day_z0, p_day_z4, y_day_z0, y_day_z4, ind
        

        ### ================== ###
        ### DM halo simulation ###
        ### ================== ###

        # Neutrino velocities
        halo_num = 2
        vels = SimData.load_velocities(sim_dir=sim_folder, halo_num=halo_num)

        # Convert velocities to momenta
        p_arr, y_arr = Physics.velocities_to_momenta_ND_halo_0th_axis(
            v_arr=vels, 
            m_arr=nu_m_picks,
            args=Params())

        p_z0 = p_arr[...,0]
        p_z4 = p_arr[...,-1]
        y_z0 = y_arr[...,0]

        # Sort in ascending order of momentum array today
        ind = p_z0.argsort(axis=-1)
        p_z0_sort = jnp.take_along_axis(p_z0, ind, axis=-1)
        p_z4_sort = jnp.take_along_axis(p_z4, ind, axis=-1)
        y_z0_sort = jnp.take_along_axis(y_z0, ind, axis=-1)

        # note: z0 sorted arrays of days and DM-sim match (i.e are equivalent)

        ### ========================= ###
        ### Phase-space interpolation ###
        ### ========================= ###

        # """
        # PSD of z0 using Fermi-Dirac assumption and Liouville's theorem
        FD_vals_z0 = Physics.Fermi_Dirac(p_z4_sort, Params())
        # (halos, masses, Npix, p_num)

        # Linear interpolation (for now) ... to get phase-space for modulation vectors
        PSD_day_halos_l = []
        for h in range(halo_num):

            PSD_day_masses_l = []
            for mi, m in enumerate(nu_m_picks):

                PSD_day_pixels_l = []
                for pix in range(simdata.Npix):

                    PSD_day_pixels_l.append(Utils.jax_interpolate_1D(
                        x_interp=y_day_z4_sort[h,mi,pix], 
                        x_points=y_z0_sort[h,mi,pix], 
                        y_points=FD_vals_z0[h,mi,pix]))

                PSD_day_masses_l.append(PSD_day_pixels_l)

            PSD_day_halos_l.append(PSD_day_masses_l)

        PSDs_day = jnp.array(PSD_day_halos_l)
        del PSD_day_pixels_l, PSD_day_masses_l, PSD_day_halos_l
        # """

        # Fermi-Dirac Phase-space, for testing
        # PSDs_day = Physics.Fermi_Dirac(p_day_z4_sort, Params())
        

        ### ================================= ###
        ### Compute number densities for day1 ###
        ### ================================= ###

        #? some values are above 0.5
        PSDs_day = jnp.clip(PSDs_day, a_min=None, a_max=0.5)

        # Integrand in number density expression
        n_raw = trap(
            p_day_z0_sort**3 * PSDs_day, jnp.log(p_day_z0_sort), axis=-1)

        # Multiply by constants and/or solid angles and convert to 1/cm**3.
        pix_sr = 4*Params.Pi/simdata.Npix
        n_cm3 = pix_sr * Params.g_nu/((2*Params.Pi)**3) * n_raw / (1/Params.cm**3)
        n_cm3_pix = jnp.array(n_cm3)
        n_tot = jnp.sum(n_cm3_pix, axis=-1)

        n_nu_days_l.append(n_tot)

        del p_day_z0_sort, p_day_z4_sort, y_day_z0_sort, y_day_z4_sort
        del vels, p_arr, y_arr, p_z0, p_z4, y_z0, p_z0_sort, p_z4_sort, y_z0_sort
        del PSDs_day, n_raw, n_cm3, n_cm3_pix, n_tot
        gc.collect()

    else:
        pass

jnp.save(f"{sim_folder}/n_nu_days.npy", jnp.array(n_nu_days_l))
jnp.save(f"{sim_folder}/days_nrs.npy", jnp.array(days_nrs_l))