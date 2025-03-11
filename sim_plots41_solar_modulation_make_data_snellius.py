from Shared.specific_CNB_sim import *

parser = argparse.ArgumentParser()
parser.add_argument('--directory', required=True)
parser.add_argument('-hn', '--halo_num', required=True)
pars = parser.parse_args()

print(datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
print(f"Halos: {int(pars.halo_num)}")


def get_rotation_matrix_euler(cell_position):
    """
    Calculate rotation matrix using Euler angles to transform between reference frames.
    
    Args:
        earth_position: Earth position vector (x,y,z) in consistent units
        cell_position: Cell position vector (x,y,z) in consistent units
        
    Returns:
        3x3 rotation matrix
    """
    # Calculate Euler angles for rotation
    # First angle: rotation around z-axis
    zAngle = jnp.arctan2(cell_position[1], -cell_position[0])
    
    # Second angle: rotation around y-axis
    yAngle = jnp.arctan2(-cell_position[2], jnp.linalg.norm(cell_position[:2]))
    
    # Compute trigonometric values
    cz = jnp.cos(zAngle)
    sz = jnp.sin(zAngle)
    cy = jnp.cos(yAngle)
    sy = jnp.sin(yAngle)
    
    # Rotation matrix around z-axis
    R_z = jnp.array([
        [cz, -sz, 0],
        [sz, cz,  0],
        [0,  0,   1]
    ])
    
    # Rotation matrix around y-axis
    R_y = jnp.array([
        [cy,  0, sy],
        [0,   1,  0],
        [-sy, 0, cy]
    ])
    
    # Combined rotation matrix (y-rotation after z-rotation)
    rot_mat = jnp.matmul(R_y, R_z)
    
    return rot_mat

def transform_pixel_indices_euler(p_unit, cell_positions, nside):
    """
    Transform pixel indices from Earth frame to each starting cell's frame
    using Euler angle rotations.
    
    Args:
        p_unit: Momentum unit vectors in Earth frame (shape: [1, masses, Npix, p_num, 3])
        cell_positions: Starting positions for each halo (shape: [halos, 3])
        nside: HEALPix nside parameter
        
    Returns:
        Pixel indices in each halo's frame (shape: [halos, masses, Npix, p_num])
    """
    num_halos = cell_positions.shape[0]
    result = []
    
    # Process each halo individually
    for h in range(num_halos):
        # Get rotation matrix from Earth to this starting cell
        R = get_rotation_matrix_euler(cell_positions[h])
        
        # Apply rotation to all momentum vectors
        # We transform from Earth frame to cell frame
        rotated_p_unit = jnp.einsum('ij,abcj->abci', R, p_unit[0])
        
        # Calculate spherical coordinates in this frame
        theta = jnp.arccos(jnp.clip(rotated_p_unit[..., 2], -1.0, 1.0))
        phi = jnp.arctan2(rotated_p_unit[..., 1], rotated_p_unit[..., 0])
        
        # Convert to pixel indices using healpy
        pixels = hp.ang2pix(nside, theta, phi)
        result.append(jnp.array(pixels))
    
    return jnp.stack(result)

def interpolate_fd_values_with_euler_rotation(
        p_GC_mag, p_GC_unit, p_grid, fd_vals, cell_positions, nside):
    """
    Interpolate distribution function values using Euler angle rotations
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
    pixel_indices_halos = transform_pixel_indices_euler(
        p_GC_unit, cell_positions, nside)
    # shape: [halos, masses, Npix, p_num]
    
    @jax.jit
    def process_halo(h):
        result = jnp.zeros_like(p_grid[h])
        
        def pixel_fun(i, val):
            for m in range(p_grid.shape[1]):
                # Use index 0 directly since p_GC_mag is the same for all halos
                p_interp = p_GC_mag[0, m, i]
                pixels = pixel_indices_halos[h, m, i]
                
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


@jax.jit
def interpolate_fd_values_parallel(p_GC_mag, pixel_indices, p_grid, fd_vals):
    @jax.jit
    def process_halo(h):
        result = jnp.zeros_like(p_grid[h])
        
        def pixel_fun(i, val):
            for m in range(p_grid.shape[1]):
                p_interp = p_GC_mag[0, m, i]
                pixels = pixel_indices[0, m, i]
                
                p0_pixels = p_grid[h, m, pixels]
                fd_pixels = fd_vals[h, m, pixels]
                
                idx = jnp.sum(p0_pixels <= p_interp[:, None], axis=1) - 1
                idx = jnp.clip(idx, 0, p_grid.shape[-1] - 2)
                
                x0 = jnp.take_along_axis(
                    p0_pixels, idx[:, None], axis=1)[:, 0]
                x1 = jnp.take_along_axis(
                    p0_pixels, (idx+1)[:, None], axis=1)[:, 0]
                y0 = jnp.take_along_axis(
                    fd_pixels, idx[:, None], axis=1)[:, 0]
                y1 = jnp.take_along_axis(
                    fd_pixels, (idx+1)[:, None], axis=1)[:, 0]
                
                slope = (y1 - y0) / (x1 - x0)
                val = val.at[m, i].set(y0 + slope * (p_interp - x0))
            return val
        
        return jax.lax.fori_loop(0, p_grid.shape[2], pixel_fun, result)
    
    # Vectorize over halos
    return jax.vmap(process_halo)(jnp.arange(p_grid.shape[0]))


def calc_CNB_density_days(
        days_vecs_dir: str,
        day_step: int,
        halo_num: int,
        with_DM_gravity: bool = True,
        interp_grav_psd: bool = True,
        Earth_frame: bool = False,
        rel_vel: str = "CNB",
        Earth_rel_Sun: bool = False,
        bound: bool | None = None,
        integrate_pixels: bool = True,
        args = None):

    if args is None:
        args = Params()

    days = []
    densities = []
    percentages = []

    # Units used for frame/boost related quantities
    Ev_unit = args.km/args.s

    # Need Earth velocities relative to both:
    # 1. Galactic Centre (GC) frame (includes solar system motion)
    # (order of 250 km/s)
    _, _, Ev_GC = SimUtil.SunEarthGC_frame_coords_posvel(
        2024, rel_vel, Earth_rel_Sun=False)
    Ev_GC_boost = Ev_GC*Ev_unit
    # 2. Sunlock (SL) frame (i.e. motion of Earth relative to only Sun)
    # (order of 30 km/s)
    _, _, Ev_SL = SimUtil.SunEarthGC_frame_coords_posvel(
        2024, rel_vel, Earth_rel_Sun=True)
    Ev_SL_boost = Ev_SL*Ev_unit

    # Initialize DM simulation data if using gravity
    if with_DM_gravity:
        dm_sim_v = SimData.load_velocities(
            sim_dir=pars.directory, halo_num=halo_num)
        
        # Calculate initial momentum arrays
        _, p_z0_dm, p_z4_dm, _, _ = Utils.sim_vels_to_sorted_z0z4(
            dm_sim_v, 
            nu_m_picks, 
            merge_last_axes=False, 
            args=args
        )

        # Phase-space "today" through z=4 momenta and Liouville's theorem
        fd_vals_z0 = Physics.Fermi_Dirac(p_z4_dm, args)
        
        if bound is not None:
            x_earth = jnp.array([8.127, 0., 0.])*args.kpc
            p_esc, _ = SimUtil.get_p_esc(
                pars.directory, x_earth, nu_m_picks, args)

    for day in range(0, 365, day_step):
        fpath = f"{days_vecs_dir}/vectors_day{day+1}.npy"
        
        if not os.path.exists(fpath):
            continue

        print(f"Day {day+1}/365")
        days.append(day+1)

        # Load velocities from day simulations (these are in "SunLock" frame)
        v_unit = args.kpc/args.s
        day_v = jnp.load(fpath)[..., 3:][None, ...]*v_unit
        # (halos, Npix, p_num, 2, 3)

        # if Earth_frame:
        #     # Transform velocities to GC frame
        #     day_v = SimUtil.S_to_Sprime_frame_trafo(
        #         day_v, earth_v[day]*Ev_unit)


        if with_DM_gravity:
            # Calculate momentum arrays using output from daily sims
            _, p_today_vec, p_1yr_vec, *_ = Utils.sim_vels_to_sorted_z0z4_vec(
                day_v/v_unit,  # functions expects kpc/s units 
                nu_m_picks, 
                merge_last_axes=False, 
                args=args
            )
            # (halos, masses, Npix, p_num, 3)
            # output momenta are with numerical units of kpc/s attached

            # Compute phase space density
            if interp_grav_psd:
                if Earth_frame:
                    # Transform momenta to GC frame
                    #/ for this we use p_1yr from daily sims
                    p_GC_mag, p_GC_unit = Physics.transform_momenta_to_orig_frame(
                        p_vec=p_1yr_vec, boost_vec=Ev_GC_boost[day])
                    # (1, masses, Npix, p_num)

                    # Pixels (indices) that 1yr momenta in GC frame point at
                    # pixel_indices = Physics.get_p_vec_pixels(
                    #     p_unit=p_GC_unit, nside=simdata.Nside)
                    # (1, masses, Npix, p_num)
                    
                    # Transform momenta to Earth frame
                    #/ for this we use p_today from daily sims
                    p_E_mag, _ = Physics.transform_momenta_to_orig_frame(
                        # p_vec=p_today_vec, boost_vec=Ev_SL_boost[day])
                        p_vec=p_today_vec, boost_vec=Ev_GC_boost[day])
                    
                    # psd = interpolate_fd_values_parallel(
                    #     p_GC_mag=p_GC_mag, 
                    #     pixel_indices=pixel_indices, 
                    #     p_grid=p_z0_dm, 
                    #     fd_vals=fd_vals_z0)

                    psd = interpolate_fd_values_with_euler_rotation(
                        p_GC_mag=p_GC_mag, 
                        p_GC_unit=p_GC_unit,
                        p_grid=p_z0_dm, 
                        fd_vals=fd_vals_z0,
                        cell_positions=init_xyzs[:halo_num],
                        nside=simdata.Nside
                    )

                else:                
                    #? unfinished...    
                    psd = Physics.interpolate_fd_values(
                        p_GC_mag=jnp.linalg.norm(p_1yr_vec, axis=-1),
                        pixel_indices=pixel_indices,
                        p_grid=p_z0_dm,
                        fd_vals=fd_vals_z0)
            else:
                psd = Physics.Fermi_Dirac(p_z4, args)

            # Clip PSD values (avoid boundary effects from interpolation)
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
                n_raw = trap(p_E_mag**3 * psd, jnp.log(p_E_mag), axis=-1)

        else:
            # Non-gravitational calculation
            _, p_z0, p_z4, _, _ = Utils.sim_vels_to_sorted_z0z4(
                day_v/v_unit,  # functions expects kpc/s units
                nu_m_picks, 
                merge_last_axes = not integrate_pixels, 
                args=args
            )
            # p_z0/z4: (H, M, 768000) or (H, M, 768, 1000)
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

        densities.append(n_total)

    results = (jnp.array(days), jnp.array(densities))
    if bound is not None and with_DM_gravity:
        results += (jnp.array(percentages),)
    
    return results


# Set preliminaries
sim_output_dir = str(pathlib.Path(pars.directory).parent)
nu_m_range = jnp.load(f"{pars.directory}/neutrino_massrange_eV.npy")
nu_m_picks = jnp.array([0.01, 0.05, 0.1, 0.2, 0.3])*Params.eV
simdata = SimData(pars.directory)

init_xyzs = jnp.array(
    [jnp.load(f"{pars.directory}/init_xyz_halo{h+1}.npy") for h in range(10)])


# Folders and names
# prefix_str = "SunLock"
# days_vecs_dir = f"{pars.directory}/SunLock_frame"

prefix_str = "NoG_Euler"
days_vecs_dir = f"{pars.directory}/NoSun_vectors"

# With DM gravity, and interpolated PSD from core sim, or FD instead
with_DM_gravity = True
halo_num = int(pars.halo_num)
interp_grav_psd = True

# Only relevant for Earth frame
Earth_frame = True
Earth_rel_Sun = False  # irrelevant actually, since we now use both velocities
# rel_vel = "CNB"
rel_vel = "MW"

day_step = 48  # Ultimately we want to use 1 to have all days
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
jnp.save(f"{pars.directory}/{prefix_str}_days_nums{suffix_str}.npy", days)
jnp.save(f"{pars.directory}/{prefix_str}_days_dens{suffix_str}.npy", densities)

# Save percentages if bound condition was used
if bound is not None and extra:
    jnp.save(
        f"{pars.directory}/{prefix_str}_days_perc{suffix_str}.npy", extra[0])