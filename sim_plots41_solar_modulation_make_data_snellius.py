from Shared.specific_CNB_sim import *

parser = argparse.ArgumentParser()
parser.add_argument('--directory', required=True)
parser.add_argument('-hn', '--halo_num', required=True)
pars = parser.parse_args()

print(datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
print(f"Halos: {int(pars.halo_num)}")

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
            # Calculate momentum arrays using output from core DM sims
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
                    
                    # Pixels (indices) that 1yr momenta in GC frame point at
                    pixel_indices = Physics.get_p_vec_pixels(
                        p_unit=p_GC_unit, nside=simdata.Nside)
                    
                    # Transform momenta to Earth frame
                    #/ for this we use p_today from daily sims
                    p_E_mag, _ = Physics.transform_momenta_to_orig_frame(
                        # p_vec=p_today_vec, boost_vec=Ev_SL_boost[day])
                        p_vec=p_today_vec, boost_vec=Ev_GC_boost[day])
                    
                    psd = interpolate_fd_values_parallel(
                        p_GC_mag=p_GC_mag, 
                        pixel_indices=pixel_indices, 
                        p_grid=p_z0_dm, 
                        fd_vals=fd_vals_z0)
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

# Folders and names
days_vecs_dir = f"{pars.directory}/SunLock_frame"
prefix_str = "SunLock"

# With DM gravity, and interpolated PSD from core sim, or FD instead
with_DM_gravity = True
halo_num = int(pars.halo_num)
interp_grav_psd = True

# Only relevant for Earth frame
Earth_frame = True
Earth_rel_Sun = False  # irrelevant actually, since we now use both velocities
# rel_vel = "CNB"
rel_vel = "MW"

day_step = 24  # Ultimately we want to use 1 to have all days
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