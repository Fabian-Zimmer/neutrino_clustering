from Shared.shared import *
from Shared.specific_CNB_sim import *


@jax.jit
def compute_psd_day(y_z4_days, y_z0_dm_sim, fd_vals_z0):
    """Compute PSD for given day using vectorized interpolation."""
    return Utils.vectorized_interpolate_1D(y_z4_days, y_z0_dm_sim, fd_vals_z0)


def calc_CNB_density_days(
        days_vecs_dir: str,
        day_step: int,
        with_DM_gravity: bool = True,
        interp_grav_psd: bool = True,
        Earth_frame: bool = False,
        rel_vel: str = "CNB",
        bound: bool | None = None,
        integrate_pixels: bool = True,
        args = None):

    if args is None:
        args = Params()

    days = []
    densities = []
    percentages = []

    # Earth velocities in GC frame (includes solar system motion)
    _, _, earth_v = SimUtil.SunEarthGC_frame_coords_posvel(2024, rel_vel)
    earth_v_unit = args.km/args.s
    earth_v_mags = jnp.linalg.norm(earth_v, axis=-1)*earth_v_unit

    # Initialize DM simulation data if using gravity
    if with_DM_gravity:
        halo_num = 1
        dm_sim_v = SimData.load_velocities(
            sim_dir=sim_folder, halo_num=halo_num)
        
        # Calculate initial momentum arrays
        _, _, p_z4_dm, y_z0_dm, _ = Utils.sim_vels_to_sorted_z0z4(
            dm_sim_v, 
            nu_m_picks, 
            merge_last_axes=False, 
            args=args
        )

        fd_vals_z0 = Physics.Fermi_Dirac(p_z4_dm, args)
        
        if bound is not None:
            x_earth = jnp.array([8.127, 0., 0.])*args.kpc
            p_esc, _ = SimUtil.get_p_esc(sim_folder, x_earth, nu_m_picks, args)

    for day in range(0, 365, day_step):
        fpath = f"{days_vecs_dir}/vectors_day{day+1}.npy"
        
        if not os.path.exists(fpath):
            continue

        print(f"Day {day+1}/365")
        days.append(day+1)

        # Load and transform velocities to Earth frame
        v_unit = args.kpc/args.s
        day_v = jnp.load(fpath)[..., 3:][None, ...]*v_unit
        print(day_v.shape)
        # (halos, Npix, p_num, 2, 3)

        if with_DM_gravity:
            # Calculate momentum arrays for gravity simulation
            sort_idx, p_z0, p_z4, _, y_z4 = Utils.sim_vels_to_sorted_z0z4(
                day_v/v_unit,  # functions expects kpc/s units 
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
            # _, _, _, p_z0, p_z4, _, _ = Utils.sim_vels_to_sorted_z0z4_vec(
            _, p_z0, p_z4, _, _ = Utils.sim_vels_to_sorted_z0z4(
                day_v/v_unit,  # functions expects kpc/s units
                nu_m_picks, 
                merge_last_axes = not integrate_pixels, 
                args=args
            )
            # p_z0/z4: (halos, masses, 768000) or (halos, masses, 768, 1000)
            # depending on merge_last_axes True or False
            psd = Physics.Fermi_Dirac(p_z4, args)
            n_raw = trap(p_z0**3 * psd, jnp.log(p_z0), axis=-1)

        # Compute final density
        if integrate_pixels:
            pix_sr = 4*args.Pi / simdata.Npix
            n_dens = pix_sr*args.g_nu / ((2*args.Pi)**3) * n_raw / args.cm**-3
            n_total = jnp.sum(jnp.array(n_dens), axis=-1)
        else:
            pix_sr = 4*args.Pi
            n_dens = pix_sr*args.g_nu / ((2*args.Pi)**3) * n_raw / args.cm**-3
            n_total = jnp.array(n_dens)

        # Number density as seen on Earth
        if Earth_frame:
            n_total *= jnp.sqrt(1 - earth_v_mags[day])

        densities.append(n_total)

    results = (jnp.array(days), jnp.array(densities))
    if bound is not None and with_DM_gravity:
        results += (jnp.array(percentages),)
    
    return results


# Set preliminaries
# sim_name = f"SunNoG"
# sim_name = f"SunMod_1k"
sim_name = f"SunMod_2k"
# sim_name = f"SunMod_5k"
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

# No gravity
# days_vecs_dir = f"{sim_folder}/NoSun_vectors"
# prefix_str = "NoSun"
# with_DM_gravity = False
# interp_grav_psd = False

# With Sun but no DM gravity
# days_vecs_dir = f"{sim_folder}/WithSun_vel_CNB_vectors"
# prefix_str = "SunNoDM"
# with_DM_gravity = False
# interp_grav_psd = False

# With Sun and DM gravity
days_vecs_dir = f"{sim_folder}/WithSun_vel_CNB_vectors"
prefix_str = "Sun+DM"
with_DM_gravity = True
interp_grav_psd = False

day_step = 24  # Ultimately we want to use 1 to have all days
Earth_frame = False
integrate_pixels = False

rel_vel = "CNB"
# rel_vel = "MW"
bound = None
# bound: Momentum boundary condition:
#     None - Use full momentum range
#     True - Use p_z0 < p_esc condition
#     False - Use p_z0 >= p_esc condition

# Calculate densities (extra is percentages, only for some conditions)
days, densities, *extra = calc_CNB_density_days(
    days_vecs_dir=days_vecs_dir, 
    day_step=day_step,
    with_DM_gravity=with_DM_gravity,
    interp_grav_psd=interp_grav_psd,
    Earth_frame=Earth_frame,
    rel_vel=rel_vel,
    bound=bound,
    integrate_pixels=integrate_pixels,
    args=Params()
)


# Build filename suffixes based on conditions
suffixes = []
if with_DM_gravity:
    if interp_grav_psd:
        suffixes.append('grav_PSD')
    else:
        suffixes.append('FD_PSD')
if not with_DM_gravity:
    suffixes.append('FD_PSD')
if Earth_frame:
    if rel_vel == "CNB":
        suffixes.append('Earth_frame_wrtCNB')
    if rel_vel == "MW":
        suffixes.append('Earth_frame_wrtMW')
if bound is not None:
    suffixes.append('bound' if bound else 'unbound')
if integrate_pixels:
    suffixes.append('int_pixels')
suffix_str = f"_{'_'.join(suffixes)}" if suffixes else ""

print(f"Done: {prefix_str}{suffix_str}")

# Save arrays with prefix and suffix
jnp.save(f"{sim_folder}/{prefix_str}_days_nums{suffix_str}.npy", days)
jnp.save(f"{sim_folder}/{prefix_str}_days_dens{suffix_str}.npy", densities)

# Save percentages if bound condition was used
if bound is not None and extra:
    jnp.save(f"{sim_folder}/{prefix_str}_days_perc{suffix_str}.npy", extra[0])