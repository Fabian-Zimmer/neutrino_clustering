from Shared.shared import *
from Shared.specific_CNB_sim import *

# region: old interpolate pixel values
def transform_pixel_indices(p_unit, cell_pos, earth_pos, nside):
    """
    Transform pixel indices from Earth frame to each starting cell's frame
    using Rodrigues' angle rotations.
    
    Args:
        p_unit: Momentum unit vectors in Earth frame (shape: [1, masses, Npix, p_num, 3])
        cell_positions: Starting positions for each halo (shape: [halos, 3])
        nside: HEALPix nside parameter
        
    Returns:
        Pixel indices in each halo's frame (shape: [halos, masses, Npix, p_num])
    """
    num_halos = cell_pos.shape[0]
    result = []
    
    # Process each halo individually
    for h in range(num_halos):
        # Get rotation matrix from Earth to this starting cell
        R = SimUtil.get_direct_rotation(earth_pos, cell_pos[h])
        
        # Apply rotation to all momentum vectors
        # We transform from Earth frame to cell frame
        rotated_p_unit = jnp.einsum('ij,abcj->abci', R, p_unit[0, ...])
        
        # Calculate spherical coordinates in this frame
        # theta = jnp.arccos(jnp.clip(rotated_p_unit[..., 2], -1.0, 1.0))
        theta = jnp.arccos(rotated_p_unit[..., 2])
        phi = jnp.arctan2(rotated_p_unit[..., 1], rotated_p_unit[..., 0])
        
        # Convert to pixel indices using healpy
        pixels = hp.ang2pix(nside, theta, phi)
        result.append(jnp.array(pixels))
    
    return jnp.stack(result)

def rotate_p_and_interpolate_fd_values(
        p_PSD_mag, p_PSD_unit, p_grid, fd_vals, cell_pos, earth_pos, nside):
    """
    Interpolate distribution function values using Rodrigues' angle rotations
    to transform between reference frames.
    
    Args:
        p_GC_mag: Momentum magnitudes (shape: [1, masses, Npix, p_num])
        p_GC_unit: Momentum unit vectors (shape: [1, masses, Npix, p_num, 3])
        p_grid: Momentum grid (shape: [halos, masses, pixels, p_grid_size])
        fd_vals: Distribution function values (shape: [halos, masses, pixels, p_grid_size])
        cell_positions: Starting cell positions (shape: [halos, 3])
        nside: HEALPix nside parameter
        
    Returns:
        Interpolated distribution function values (shape: [halos, masses, Npix, p_num])
    """
    # Transform pixel indices to each halo's frame
    pixel_indices_halos = transform_pixel_indices(
        p_PSD_unit, cell_pos, earth_pos, nside)
    # shape: [halos, masses, Npix, p_num]
    
    @jax.jit
    def process_halo(h):
        result = jnp.zeros_like(p_grid[h])
        
        def pixel_fun(i, val):
            for m in range(p_grid.shape[1]):
                # Use index 0 directly since p_GC_mag is the same for all halos
                p_interp = p_PSD_mag[0, m, i]

                # heal-pixels for current halo-mass permutation
                pixels = pixel_indices_halos[h, m, i]
                
                # z=0 momenta, psd values (x-axis, y-axis for interpolation) 
                # for current halo-mass permutation
                p0_pixels = p_grid[h, m, pixels]
                fd_pixels = fd_vals[h, m, pixels]
                
                # Find indices for interpolation
                idx = jnp.sum(p0_pixels <= p_interp[:, None], axis=1) - 1
                idx = jnp.clip(idx, 0, p_grid.shape[-1] - 2)
                
                # Get interpolation points
                x0 = jnp.take_along_axis(
                    p0_pixels, idx[:, None], axis=1)[:, 0]
                x1 = jnp.take_along_axis(
                    p0_pixels, (idx+1)[:, None], axis=1)[:, 0]
                y0 = jnp.take_along_axis(
                    fd_pixels, idx[:, None], axis=1)[:, 0]
                y1 = jnp.take_along_axis(
                    fd_pixels, (idx+1)[:, None], axis=1)[:, 0]
                
                # Linear interpolation
                slope = (y1 - y0) / (x1 - x0)
                val = val.at[m, i].set(y0 + slope * (p_interp - x0))
            return val
        
        return jax.lax.fori_loop(0, p_grid.shape[2], pixel_fun, result)
    
    # Vectorize over halos
    return jax.vmap(process_halo)(jnp.arange(p_grid.shape[0]))
# endregion


def transform_pixel_indices_v2(p_unit, cell_pos, nside):
    """
    Transform pixel indices from Earth frame to each starting cell's frame
    using Euler angle rotations and healpy's lonlat convention.
    
    Args:
        p_unit: Momentum unit vectors in Earth frame (shape: [1, masses, Npix, p_num, 3])
        cell_pos: Starting positions for each halo (shape: [halos, 3])
        earth_pos: Earth position vector (shape: [3])
        nside: HEALPix nside parameter
        
    Returns:
        Pixel indices in each halo's frame (shape: [halos, masses, Npix, p_num])
    """
    num_halos = cell_pos.shape[0]
    result = []
    
    # Process each halo individually
    for h in range(num_halos):
        # Get rotation matrix from Earth frame to this starting cell's frame
        R = SimUtil.get_rotation_matrix_euler(cell_pos[h])
        
        # Apply rotation to all momentum unit vectors
        rot_p_unit = jnp.einsum('ij,abcj->abci', R, p_unit[0, ...])
        
        # Extract x,y,z components
        px, py, pz = rot_p_unit[..., 0], rot_p_unit[..., 1], rot_p_unit[..., 2]

        # Projected distance on xy-plane
        proj_xy = jnp.sqrt(px**2 + py**2)

        # Get galactic longitude and galactic latitude in healpy convention
        hp_glon = jnp.rad2deg(jnp.arctan2(py, px))
        hp_glat = jnp.rad2deg(jnp.arctan2(pz, proj_xy))

        # Get healpy pixels
        hp_pixels = hp.ang2pix(nside, hp_glon, hp_glat, lonlat=True)
        result.append(hp_pixels)

    return jnp.stack(result)


def transform_pixel_indices_v3(p_unit, nside):

    # Extract x,y,z components
    px, py, pz = p_unit[..., 0], p_unit[..., 1], p_unit[..., 2]

    # Projected distance on xy-plane
    proj_xy = jnp.sqrt(px**2 + py**2)

    # Get galactic longitude and galactic latitude in healpy convention
    hp_glon = jnp.rad2deg(jnp.arctan2(py, px))
    hp_glat = jnp.rad2deg(jnp.arctan2(pz, proj_xy))

    # Get healpy pixels
    hp_pixels = hp.ang2pix(nside, hp_glon, hp_glat, lonlat=True)
    return jnp.array(hp_pixels)


def rotate_p_and_select_closest_fd_values(
        p_PSD_mag, p_PSD_unit, p_grid, fd_vals, cell_pos, nside):
    """
    Select distribution function values using Euler angle rotations
    to transform between reference frames, choosing the closest p_grid value
    instead of interpolating.
    
    Args:
        p_PSD_mag: Momentum magnitudes (shape: [1, masses, Npix, p_num])
        p_PSD_unit: Momentum unit vectors (shape: [1, masses, Npix, p_num, 3])
        p_grid: Momentum grid (shape: [halos, masses, pixels, p_grid_size])
        fd_vals: Distribution function values (shape: [halos, masses, pixels, p_grid_size])
        cell_pos: Starting cell positions (shape: [halos, 3])
        earth_pos: Earth position (shape: [3])
        nside: HEALPix nside parameter
        
    Returns:
        Selected distribution function values (shape: [halos, masses, Npix, p_num])
    """
    # Transform pixel indices to each halo's frame
    # pixel_indices_halos = transform_pixel_indices_v2(
    #     p_PSD_unit, cell_pos, nside)
    # shape: [halos, masses, Npix, p_num]

    # Without rotation
    pixel_indices_halos = transform_pixel_indices_v3(
        p_PSD_unit, nside)
    # shape: [1, masses, Npix, p_num]

    @jax.jit
    def process_halo(h):
        result = jnp.zeros_like(p_grid[h])
        
        def pixel_fun(i, val):
            for m in range(p_grid.shape[1]):
                # Use index 0 directly since p_PSD_mag is the same for all halos
                p_interp = p_PSD_mag[0, m, i]

                # heal-pixels for current halo-mass permutation
                pixels = pixel_indices_halos[h, m, i]
                
                # z=0 momenta, psd values for current halo-mass permutation
                p0_pixels = p_grid[h, m, pixels]
                fd_pixels = fd_vals[h, m, pixels]
                
                # Calculate absolute differences to find closest point
                abs_diff = jnp.abs(p0_pixels - p_interp[:, None])
                
                # Get indices of closest values
                closest_idx = jnp.argmin(abs_diff, axis=1)
                
                # Select fd values at those indices
                closest_fd_values = jnp.take_along_axis(
                    fd_pixels, closest_idx[:, None], axis=1)[:, 0]
                
                # Set the result
                val = val.at[m, i].set(closest_fd_values)
            return val
        
        return jax.lax.fori_loop(0, p_grid.shape[2], pixel_fun, result)
    
    # Vectorize over halos
    return jax.vmap(process_halo)(jnp.arange(p_grid.shape[0]))


def compare_fd_max_values(
        original_fd, interpolated_fd, halo_idx=0, mass_idx=0, p_grid=None, figsize=(10, 6), x_lims=None, y_min=1e-2, y_max=0.5):
    """
    Compare maximum distribution function values across pixels for each momentum value.
    
    Args:
        original_fd: Original distribution function values (shape: [halos, masses, Npix, p_num])
        interpolated_fd: Interpolated distribution function values (shape: [halos, masses, Npix, p_num])
        halo_idx: Index of the halo to plot
        mass_idx: Index of the mass to plot
        p_grid: Momentum grid values (optional, for x-axis labeling)
        figsize: Figure size tuple
    """
    # Extract data for selected halo and mass
    orig = original_fd[halo_idx, mass_idx]    # Shape: [Npix, p_num]
    interp = interpolated_fd[halo_idx, mass_idx]  # Shape: [Npix, p_num]
    
    # Find the maximum value across pixels for each momentum value
    max_orig_per_p = np.max(orig, axis=0)    # Shape: [p_num]
    max_interp_per_p = np.max(interp, axis=0)  # Shape: [p_num]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # X-axis values
    x = p_grid if p_grid is not None else np.arange(orig.shape[1])
    
    # Plot maximum values
    ax.plot(
        x, max_orig_per_p, ls='-.', color='black', alpha=0.6, 
        label='Original PSD')
    ax.plot(
        x, max_interp_per_p*1.1, ls='solid', color='red', alpha=0.8, 
        label='Interpolated PSD')
    
    # Plot relative error
    rel_error = np.abs(max_interp_per_p - max_orig_per_p) / max_orig_per_p
    ax2 = ax.twinx()
    ax2.plot(
        x, rel_error * 100, 'g-', alpha=0.5, label=r'Relative Error ($\%$)')
    ax2.set_ylabel(r'Relative Error ($\%$)', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    
    # Add lines to show mean relative error
    mean_rel_error = np.mean(rel_error) * 100
    ax2.axhline(mean_rel_error, color='g', linestyle='--', alpha=0.5)
    ax2.text(x[-1] * 0.3, mean_rel_error * 1.2, f'Mean: {mean_rel_error:.2f}%', color='g')
    
    ax.set_xlabel('Momentum Value' if p_grid is not None else 'Momentum Index')
    ax.set_ylabel('Maximum Distribution Function Value')
    ax.set_title(f'Maximum PSD Values Across Pixels (Halo {halo_idx}, Mass {mass_idx})')
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='lower left')
    
    ax.grid(True, alpha=0.3)
    

    # Set y-axis to log scale with specified limits
    ax.set_yscale('log')
    ax.set_ylim(y_min, y_max)
    
    # Set x-axis limits
    ax.set_xlim(x_lims[0], x_lims[1])
    
    plt.tight_layout()

    plt.savefig('healpix_interpolation_demo.png', dpi=150, bbox_inches='tight')
    # plt.show()

    return fig



def calc_CNB_density_days(
        days_vecs_dir: str,
        day_step: int,
        halo_num: int,
        with_DM_gravity: bool = True,
        interp_grav_psd: bool = True,
        Earth_frame: bool = False,
        rel_vel: str = "MW",
        Earth_rel_Sun: bool = False,
        bound: bool | None = None,
        integrate_pixels: bool = True,
        args = None):

    if args is None:
        args = Params()

    days = []
    densities = []
    percentages = []
    psds = []

    # region: Boost velocities
    # Units used for frame/boost related quantities
    Ev_unit = args.km/args.s

    # Need Earth velocities relative to both:
    # 1. Galactic Centre (GC) frame (includes solar system motion)
    # (order of 250 km/s)
    _, _, Ev_GC, _ = SimUtil.SunEarthGC_frame_coords_posvel(
        2024, rel_vel, Earth_rel_Sun=False)
    Ev_GC_boost = Ev_GC*Ev_unit
    # 2. Sunlock (SL) frame (i.e. motion of Earth relative to only Sun)
    # (order of 30 km/s)
    _, _, Ev_SL, _ = SimUtil.SunEarthGC_frame_coords_posvel(
        2024, rel_vel, Earth_rel_Sun=True)
    Ev_SL_boost = Ev_SL*Ev_unit
    # endregion

    # Initialize DM simulation data if using gravity
    if with_DM_gravity:
        dm_sim_v = SimData.load_velocities(
            sim_dir=sim_folder, halo_num=halo_num)
        
        # Calculate initial momentum arrays
        _, p_z0_dm, p_z4_dm, _, _ = Utils.sim_vels_to_sorted_z0z4(
            dm_sim_v, 
            nu_m_picks, 
            merge_last_axes=False, 
            args=args
        )
        # (halos, masses, Npix, p_num)

        # Phase-space "today" through z=4 momenta and Liouville's theorem
        fd_vals_z0 = Physics.Fermi_Dirac(p_z4_dm, args)
        
        if bound is not None:
            x_earth = jnp.array([8.127, 0., 0.])*args.kpc
            p_esc, _ = SimUtil.get_p_esc(sim_folder, x_earth, nu_m_picks, args)


    for day in range(0, 365, day_step):
        
        # region: Preamble
        t_start = time.perf_counter()

        fpath = f"{days_vecs_dir}/vectors_day{day+1}.npy"
        if not os.path.exists(fpath):
            continue
        
        print(f"Day {day+1}/365")
        days.append(day+1)

        # Load velocities from day simulations (these are in "SunLock" frame)
        v_unit = args.kpc/args.s
        day_v = jnp.load(fpath)[..., 3:][None, ...]*v_unit
        # (1, Npix, p_num, 2, 3)
        # endregion

        if with_DM_gravity:
            # Calculate momentum arrays using output from daily sims
            _, p_today_vec, p_1yr_vec, *_ = Utils.sim_vels_to_sorted_z0z4_vec(
                day_v/v_unit,  # functions expects kpc/s units 
                nu_m_picks, 
                merge_last_axes=False, 
                args=args
            )
            # (1, masses, Npix, p_num, 3) vec, (1, masses, Npix, p_num) mag
            # output momenta are with numerical units of kpc/s attached

            # Compute phase space density
            if interp_grav_psd:
                if Earth_frame:        
                    # Momentum to use for PSD
                    # (p_1yr from daily sims, transformed into GC frame)
                    p_PSD_mag, p_PSD_unit = Physics.transform_momenta_to_orig_frame(
                        p_vec=p_1yr_vec, 
                        # boost_vec=Ev_GC_boost[day], 
                        boost_vec=jnp.zeros_like(Ev_GC_boost[0]), 
                        masses=nu_m_picks)
                    # (1, masses, Npix, p_num), (1, masses, Npix, p_num, 3)

                    # Momentum to integrate over
                    # (p_today from daily sims, transformed into Earth frame)
                    p_int_mag, _ = Physics.transform_momenta_to_orig_frame(
                        p_vec=p_today_vec, 
                        # boost_vec=Ev_SL_boost[day], 
                        boost_vec=jnp.zeros_like(Ev_SL_boost[0]), 
                        masses=nu_m_picks)
                    # (1, masses, Npix, p_num)

                    psd = rotate_p_and_select_closest_fd_values(
                        p_PSD_mag=p_PSD_mag, 
                        p_PSD_unit=p_PSD_unit,
                        p_grid=p_z0_dm, 
                        fd_vals=fd_vals_z0,
                        cell_pos=init_xyzs[:halo_num],
                        nside=simdata.Nside
                    )
                    # fd_vals_z0 and psd both (halos, masses, Npix, p_num)

                    # region: Plot comparison of interpolated vs. original psd
                    # if (day+1) == 1:
                    #     compare_fd_max_values(
                    #         original_fd=fd_vals_z0, interpolated_fd=psd,
                    #         halo_idx=0, mass_idx=3, p_grid=None, 
                    #         figsize=(10, 6),
                    #         x_lims=(0, 700), y_min=0.01, y_max=0.8)
                    # endregion
                else:                
                    ...
            else:
                psd = Physics.Fermi_Dirac(p_z4, args)

            # Clip PSD values to avoid potential boundary issues from interp.
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
                n_raw = trap(p_int_mag**3 * psd, jnp.log(p_int_mag), axis=-1)

        else:
            # Non-DM-gravitational calculation
            _, p_today_vec, p_1yr_vec, *_ = Utils.sim_vels_to_sorted_z0z4_vec(
                day_v/v_unit,  # functions expects kpc/s units
                nu_m_picks, 
                merge_last_axes=False, 
                args=args
            )
            # (1, masses, Npix, p_num, 3)
            # output momenta are with numerical units of kpc/s attached

            if Earth_frame:        
                # Momentum to use for PSD
                # (p_1yr from daily sims, transformed into GC frame)
                p_PSD_mag, _ = Physics.transform_momenta_to_orig_frame(
                    p_vec=p_1yr_vec, 
                    # boost_vec=Ev_GC_boost[day],
                    boost_vec=jnp.zeros_like(Ev_GC_boost[0]), 
                    masses=nu_m_picks)
                # (1, masses, Npix, p_num), (1, masses, Npix, p_num, 3)

                # Momentum to integrate over
                # (p_today from daily sims, transformed into Earth frame)
                p_int_mag, _ = Physics.transform_momenta_to_orig_frame(
                    p_vec=p_today_vec, 
                    # boost_vec=Ev_SL_boost[day], 
                    boost_vec=jnp.zeros_like(Ev_SL_boost[0]), 
                    masses=nu_m_picks)
                # (1, masses, Npix, p_num)

            # Fermi-Dirac as boundary PSD
            # psd = Physics.Fermi_Dirac(p_PSD_mag, args)

            # SHM as boundary PSD
            v_0 = 220*args.km/args.s  # 220, 400
            v_esc_MW = 550*args.km/args.s
            psd = Physics.SHM_PSD(
                day_v[..., -1, :], v_esc_MW, v_0, args)[:, None, ...]

            n_raw = trap(p_int_mag**3 * psd, jnp.log(p_int_mag), axis=-1)

        # Compute final density
        if integrate_pixels:
            pix_sr = 4*args.Pi / simdata.Npix
            n_dens = pix_sr*args.g_nu / ((2*args.Pi)**3) * n_raw / args.cm**-3
            n_total = jnp.sum(jnp.array(n_dens), axis=-1)
        else:
            pix_sr = 4*args.Pi
            n_dens = pix_sr*args.g_nu / ((2*args.Pi)**3) * n_raw / args.cm**-3
            n_total = jnp.array(n_dens)

        densities.append(n_total)
        psds.append(psd)

        tot_time = time.perf_counter() - t_start
        print(f"Loop time: {tot_time/60.:.2f} min")

    results = (jnp.array(days), jnp.array(densities), jnp.array(psds))
    if bound is not None and with_DM_gravity:
        results += (jnp.array(percentages),)
    
    return results


# sim_name = f"SunNoG"
sim_name = f"SunMod_1k"
# sim_name = f"SunMod_2k"
sim_folder = f"sim_output/{sim_name}"

prefix_str = "SunMoveDop5_fin"
days_vecs_dir = f"{sim_folder}/SunMove_Dopri5_wrtMW"

# With DM gravity, and interpolated PSD from core sim, or FD instead
with_DM_gravity = True
halo_num = 1  #/ 5 is max on laptop with current routine
interp_grav_psd = True

# Earth frame parameters
Earth_frame = True
Earth_rel_Sun = False
# Only relevant if Earth_rel_Sun = False
# rel_vel = "CNB"
rel_vel = "MW"

day_step = 12  # Ultimately we want to use 1 to have all days
integrate_pixels = True
bound = None
# bound: Momentum boundary condition:
#     None - Use full momentum range
#     True - Use p_z0 < p_esc condition
#     False - Use p_z0 >= p_esc condition


print(datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
print(f"Halos: {int(halo_num)}")

# In units of kpc (i.e array already divided by Params.kpc)
init_xyzs = jnp.array(
    [jnp.load(f"{sim_folder}/init_xyz_halo{h+1}.npy") for h in range(10)])

# nu_m_picks = jnp.array([0.01, 0.05, 0.15, 0.2, 0.3])*Params.eV
nu_m_picks = jnp.array([0.15, 0.2, 0.25, 0.3, 0.01])*Params.eV
simdata = SimData(sim_folder)

# Calculate densities (extra is percentages, only for some conditions)
days, densities, psds, *extra = calc_CNB_density_days(
    days_vecs_dir=days_vecs_dir, 
    day_step=day_step,
    with_DM_gravity=with_DM_gravity,
    halo_num=halo_num,
    interp_grav_psd=interp_grav_psd,
    Earth_frame=Earth_frame,
    rel_vel=rel_vel,
    Earth_rel_Sun=Earth_rel_Sun,
    bound=bound,
    integrate_pixels=integrate_pixels,
    args=Params()
)

# region: file saving
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
    if Earth_rel_Sun:
        suffixes.append('Earth_frame_wrtSun')
    else:
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
jnp.save(
    f"{sim_folder}/annual_densities_numerical/{prefix_str}_days_nums{suffix_str}.npy", days)
jnp.save(
    f"{sim_folder}/annual_densities_numerical/{prefix_str}_days_dens{suffix_str}.npy", densities)
jnp.save(
    f"{sim_folder}/annual_densities_numerical/{prefix_str}_days_psds{suffix_str}.npy", psds)

# Save percentages if bound condition was used
if bound is not None and extra:
    jnp.save(
        f"{sim_folder}/annual_densities_numerical/{prefix_str}_days_perc{suffix_str}.npy", extra[0])
# endregion