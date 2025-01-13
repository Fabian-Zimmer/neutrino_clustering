from Shared.shared import *
from Shared.specific_CNB_sim import *


@jax.jit
def compute_psd_day(y_z4_days, y_z0_dm_sim, fd_vals_z0):
    """Compute PSD for given day using vectorized interpolation."""
    return Utils.vectorized_interpolate_1D(y_z4_days, y_z0_dm_sim, fd_vals_z0)


def calc_CNB_density_days(
        days_vecs_dir: str,
        day_step: int,
        use_gravity: bool = True,
        interp_grav_psd: bool = True,
        use_boosted_fd: bool = False,
        bound: bool | None = None,
        integrate_pixels: bool = True,
        args = None):

    if args is None:
        args = Params()

    days = []
    densities = []
    percentages = []

    # Earth velocities in GC frame (includes solar system motion)
    _, _, earth_v = SimUtil.SunEarthGC_frame_coords_posvel(2024)
    earth_v *= (args.km/args.s)/(args.kpc/args.s)

    # Initialize DM simulation data if using gravity
    if use_gravity:
        halo_num = 1
        dm_sim_v = SimData.load_velocities(
            sim_dir=sim_folder, halo_num=halo_num)
        
        # Calculate initial momentum arrays
        _, p_z0_dm, p_z4_dm, y_z0_dm, _ = Utils.sim_vels_to_sorted_z0z4(
            dm_sim_v, 
            nu_m_picks, 
            merge_last_axes=False, 
            args=args
        )
        
        fd_vals_z0 = Physics.Fermi_Dirac(p_z4_dm, args)
        
        if bound is not None:
            x_earth = jnp.array([8.127, 0., 0.]) * args.kpc
            p_esc, _ = SimUtil.get_p_esc(sim_folder, x_earth, nu_m_picks, args)

    for day in range(0, 365, day_step):
        fpath = f"{days_vecs_dir}/vectors_day{day+1}.npy"
        
        if not os.path.exists(fpath):
            continue

        print(f"Day {day+1}/365")
        days.append(day+1)

        # Load and transform velocities to Earth frame
        day_v = jnp.load(fpath)[..., 3:][None, ...]
        day_v = SimUtil.CNB_to_Earth_frame_trafo(day_v, earth_v[day])

        if use_gravity:
            # Calculate momentum arrays for gravity simulation
            sort_idx, p_z0, p_z4, _, y_z4 = Utils.sim_vels_to_sorted_z0z4(
                day_v, 
                nu_m_picks, 
                merge_last_axes=False, 
                args=args
            )

            # Compute phase space density
            if interp_grav_psd:
                psd = compute_psd_day(
                    jnp.repeat(y_z4, len(y_z0_dm), axis=0),
                    y_z0_dm,
                    fd_vals_z0
                )
                psd = jnp.take_along_axis(psd, sort_idx, axis=-1)
            else:
                psd = Physics.Fermi_Dirac(p_z4, args)

            # Clip PSD values
            psd = jnp.clip(psd, a_min=None, a_max=0.5)

            # Apply momentum boundary conditions if requested
            if bound is not None:
                if bound:
                    esc_mask = p_z0 < p_esc[:halo_num, :, None, None]
                else:
                    esc_mask = p_z0 >= p_esc[:halo_num, :, None, None]
                
                if integrate_pixels:
                    psd_masked = jnp.where(esc_mask, psd, 0.)
                    n_raw = trap(p_z0**3 * psd_masked, jnp.log(p_z0), axis=-1)
                    n_norm = trap(p_z0**3 * psd, jnp.log(p_z0), axis=-1)
                    perc = (n_raw/n_norm) * 100
                    percentages.append(perc)
                else:
                    n_raw = trap(p_z0**3 * psd_masked, jnp.log(p_z0), axis=-1)
            else:
                n_raw = trap(p_z0**3 * psd, jnp.log(p_z0), axis=-1)

        else:
            # Non-gravitational calculation
            _, p_vec_z0, p_vec_z4, p_z0, p_z4, _, _ = Utils.sim_vels_to_sorted_z0z4_vec(
                day_v, 
                nu_m_picks, 
                merge_last_axes = not integrate_pixels, 
                args=args
            )
            
            if use_boosted_fd:
                psd = Physics.Fermi_Dirac_boosted(
                    p_vec_z4, 
                    earth_v[day]*(args.kpc/args.s), 
                    args
                )
            else:
                psd = Physics.Fermi_Dirac(p_z4, args)

            n_raw = trap(p_z0**3 * psd, jnp.log(p_z0), axis=-1)

        # Compute final density
        if integrate_pixels:
            pix_sr = 4 * args.Pi / simdata.Npix
            n_dens = pix_sr * args.g_nu / ((2 * args.Pi)**3) * n_raw / args.cm**-3
            n_total = jnp.sum(jnp.array(n_dens), axis=-1)
        else:
            pix_sr = 4 * args.Pi
            n_dens = pix_sr * args.g_nu / ((2 * args.Pi)**3) * n_raw / args.cm**-3
            n_total = jnp.array(n_dens)

        densities.append(n_total)

    results = (jnp.array(days), jnp.array(densities))
    if bound is not None and use_gravity:
        results += (jnp.array(percentages),)
    
    return results


# Set preliminaries
sim_name = f"SunNoG"
sim_folder = f"sim_output/{sim_name}"
fig_folder = f"figures_local/{sim_name}"
nu_m_range = jnp.load(f"{sim_folder}/neutrino_massrange_eV.npy")
nu_m_picks = jnp.array([0.01, 0.05, 0.1, 0.2, 0.3])*Params.eV
simdata = SimData(sim_folder)

#! Broken halos: either snapshot info missing or anomalous number densities
exclude_nums = jnp.array([
    20,  # halo with missing/broken snapshot info
    23,  # halo with anomalous number densities (~0s on almost all pixels)
    24,  # anomalous "compactified into 1 cell" DM halo
    25,  # anomalous "compactified into 1 cell" DM halo
])
halo_nums = [x for x in range(1, 31) if x not in exclude_nums]

# Set configuration
days_vecs_dir = f"{sim_folder}/NoSun_vectors"
prefix_str = "NoSun"
use_gravity = False
interp_grav_psd = True
day_step = 24  # Ultimately we want to use 1 to have all days
use_boosted_fd = False
integrate_pixels = True
bound = None
# bound: Momentum boundary condition:
#     None - Use full momentum range
#     True - Use p_z0 < p_esc condition
#     False - Use p_z0 >= p_esc condition

# Calculate densities (extra is percentages, only for some conditions)
days, densities, *extra = calc_CNB_density_days(
    days_vecs_dir=days_vecs_dir, 
    day_step=day_step,
    use_gravity=use_gravity,
    interp_grav_psd=interp_grav_psd,
    use_boosted_fd=use_boosted_fd,
    bound=bound,
    integrate_pixels=integrate_pixels,
    args=Params()
)


# Build filename suffixes based on conditions
suffixes = []
if use_gravity:
    if interp_grav_psd:
        suffixes.append('grav_PSD')
    else:
        suffixes.append('FD_PSD')
if not use_gravity:
    suffixes.append('FD_PSD')
if use_boosted_fd:
    suffixes.append('boosted')
if bound is not None:
    suffixes.append('bound' if bound else 'unbound')
if integrate_pixels:
    suffixes.append('int_pixels')
suffix_str = f"_{'_'.join(suffixes)}" if suffixes else ""

# Save arrays with prefix and suffix
jnp.save(f"{sim_folder}/{prefix_str}_days_nums{suffix_str}.npy", days)
jnp.save(f"{sim_folder}/{prefix_str}_days_dens{suffix_str}.npy", densities)

# Save percentages if bound condition was used
if bound is not None and extra:
    jnp.save(f"{sim_folder}/{prefix_str}_days_perc{suffix_str}.npy", extra[0])