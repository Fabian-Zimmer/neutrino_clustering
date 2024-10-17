from Shared.shared import *
from Shared.specific_CNB_sim import *

sim_name = f"SunMod_1k"
sim_folder = f"sim_output/{sim_name}"

fig_folder = f"figures_local/{sim_name}"
Cl_folder = f"Shared/Cls"
nu_m_range = jnp.load(f"{sim_folder}/neutrino_massrange_eV.npy")
nu_m_picks = jnp.array([0.01, 0.05, 0.1, 0.2, 0.3])*Params.eV
simdata = SimData(sim_folder)


import datetime


### --------- ###
### Constants ###
### --------- ###


# Length of semi-major axis of Earth's orbit
a = 1.4960e8*Params.km  # (almost equal to Params.AU)

# Eccentricity of Earth's orbit
ecc = 0.016722

# Ecliptic longitude of the perihelion as in 2013
y_p = 102*Params.deg

# Earth's average orbital speed
v_earth = 29.79*Params.km/Params.s

# Approximate escape velocity for MW
v_esc_MW = 550*Params.km/Params.s

# Angular frequency of Earth's orbit
omega = 2*Params.Pi/Params.yr

# Time of vernal equinox (in fraction of 1 year)
t_ve = 79/365*Params.yr

# Time of perihelion (in fraction of 1 year)
t_p = 4/365*Params.yr

# Sun's velocity in CNB(==CMB) frame
v_CNB = jnp.array([-0.0695, -0.662, 0.747])*369*Params.km/Params.s

# Sun's velocity in Galactic frame
v_sun = jnp.array([11, 232, 7])*Params.km/Params.s

# Analytical expression for uniform CNB energy density, "far away from Sun"
rho_infty = Params.rho_0

# Unit vectors of ecliptic plane in Galactic coordinates as in 2013
eps1 = jnp.array([0.9940, 0.1095, 0.0031])
eps2 = jnp.array([-0.0517, 0.4945, -0.8677])


# Generate time points for one year
times = jnp.linspace(0, Params.yr, 365)
jd_times = Time('2023-01-01').jd + times / Params.yr * 365


@jax.jit
def earth_vector(t):
    """Calculate Earth's position vector relative to the Sun at time t."""
    g_of_t = omega * (t - t_p)
    nu_angle = g_of_t + 2*ecc*jnp.sin(g_of_t) + 5/4*ecc**2*jnp.sin(2*g_of_t)
    r_of_t = a*(1-ecc**2)/(1+ecc*jnp.cos(nu_angle))
    y_of_t = y_p + nu_angle
    return r_of_t * (-eps1*jnp.sin(y_of_t) + eps2*jnp.cos(y_of_t))


@jax.jit
def earth_velocity(t):
    """Calculate Earth's velocity relative to the Sun at time t."""
    phase = omega * (t - t_p)
    return v_earth * (eps1*jnp.cos(phase) + eps2*jnp.sin(phase))


# Using ecliptic functions
earth_positions = jnp.array([earth_vector(t) for t in times])  # in num. AU
earth_velocities = jnp.array([earth_velocity(t) for t in times]) # in num. km/s
# print(earth_positions.shape, earth_velocities.shape)

@jax.jit
def v_infinity(v_s, r_s):
    """
    Calculate the initial Solar-frame velocity for particles, s.t. they have velocity v_s at Earth's location.
    """
    
    v_GM = 2*Params.G*Params.Msun/jnp.linalg.norm(r_s, axis=-1)
    v_inf2 = jnp.linalg.norm(v_s, axis=-1)**2 - v_GM
    # v_inf = jnp.sqrt(jnp.maximum(0, v_inf2))
    v_inf = jnp.sqrt(v_inf2)
    r_s_unit = r_s / jnp.linalg.norm(r_s, axis=-1)
    vr_s_dot = jnp.dot(v_s, r_s_unit)

    numer = v_inf2[..., None]*v_s + v_inf[..., None]*v_GM/2*r_s_unit - v_inf[..., None]*v_s*vr_s_dot[..., None]
    denom = v_inf2 + v_GM/2 - v_inf*vr_s_dot
    
    return numer / denom[..., None]


@jax.jit
def f_distr_3D(mesh_v_range, t_index, m_nu, bound, v_0):

    # Create a meshgrid for v_x, v_y, v_z
    v_x, v_y, v_z = jnp.meshgrid(
        mesh_v_range, mesh_v_range, mesh_v_range, indexing='ij')

    def bound_case(_):

        # Calculate v_s for each point in the meshgrid
        v_nu = jnp.stack([v_x, v_y, v_z], axis=-1)
        r_s = earth_positions[t_index]
        v_s = v_nu + earth_velocities[t_index]
        v_inf = v_infinity(v_s, r_s)
        v_for_f = v_CNB + v_inf

        # Calculate the magnitude of v_for_f
        v_for_f_mag = jnp.linalg.norm(v_for_f, axis=-1)

        # Create the mask
        mask = v_for_f_mag < v_esc_MW

        # Calculate the distribution
        # exp_term = jnp.exp(-v_for_f_mag**2 / v_0**2)
        exp_term = jnp.exp(-(v_x**2 + v_y**2 + v_z**2) / v_0**2)
        f_v = jnp.where(
            mask, 
            jnp.power(jnp.pi * v_0**2, -3/2) * exp_term, 
            jnp.zeros_like(exp_term)
        )

        # Normalize
        z = v_esc_MW / v_0
        N_esc = jsp.special.erf(z) - 2 / jnp.sqrt(jnp.pi) * z * jnp.exp(-z**2)
        f_v_normalized = f_v / N_esc

        return f_v_normalized

    def unbound_case(_):
        # Implementation for unbound case
        return jnp.zeros_like(v_x)

    f_v = jax.lax.cond(bound, bound_case, unbound_case, operand=None)

    return f_v


@jax.jit
def f_distr(v_range, t_index, m_nu, bound, v_0):
    """Phase-space distribution at Earth's location."""

    # Get x,y,z coords.
    v_nu = Utils.v_mag_to_xyz(v_range, Params.key)
    
    r_s = earth_positions[t_index]
    v_s = v_nu + earth_velocities[t_index]
    v_inf = v_infinity(v_s, r_s)
    v_for_f = v_CNB + v_inf

    # Test without Sun
    # v_for_f = v_nu #+ v_CNB + earth_velocities[t_index]

    def bound_case(_):
        v_for_f_mag = jnp.linalg.norm(v_for_f, axis=-1)
        mask = v_for_f_mag < v_esc_MW

        f_v = jnp.where(
            mask, 
            jnp.power(jnp.pi*v_0**2, -3/2) * jnp.exp(-v_for_f_mag**2/v_0**2), 
            0.0)
        z = v_esc_MW/v_0
        N_esc = jsp.special.erf(z) - 2/jnp.sqrt(jnp.pi)*z*jnp.exp(-z**2)
        f_v_normalized = f_v / N_esc

        return f_v_normalized

    def unbound_case(_):
        v_for_f_mag = jnp.linalg.norm(v_for_f, axis=-1)
        f_v = Physics.Fermi_Dirac(v_for_f_mag, Params())
        # f_norm = 
        return f_v

    f_v = jax.lax.cond(bound, bound_case, unbound_case, operand=None)

    return f_v


@jax.jit
def number_density_3D(t_index, m_nu, bound, v_0):
    """Calculate the neutrino number density at time t (a certain day)."""

    p_range = jnp.geomspace(0.01, 100, 1000) * Params.T_CNB
    v_range = p_range / m_nu
    v_max = 100*Params.T_CNB / m_nu
    mesh_v_range = jnp.linspace(-v_max, v_max, 50)

    def bound_case(_):
        
        # Get 3D velocity distribution
        f_v = f_distr_3D(mesh_v_range, t_index, m_nu, bound, v_0)
        
        # Integrate over 3D velocity sphere
        integral_over_z = trap(f_v, mesh_v_range, axis=-1)
        integral_over_yz = trap(integral_over_z, mesh_v_range, axis=-1)
        integral_over_xyz = trap(integral_over_yz, mesh_v_range, axis=-1)
        
        # Compute and return number density        
        return integral_over_xyz / (2*jnp.pi)**3

    def unbound_case(_):
        integrand = p_range**3 * f_distr(v_range, t_index, m_nu, bound, v_0)
        integral = trap(integrand, x=jnp.log(v_range))
        nu_dens = integral/(2*jnp.pi**2)/Params.cm**-3
        return nu_dens
    
    nu_dens = jax.lax.cond(bound, bound_case, unbound_case, operand=None)

    return nu_dens


@jax.jit
def calculate_modulation(m_nu, bound, v_0):
    """Calculate the fractional modulation throughout the year."""

    nu_dens_fctn = lambda t_index: number_density_3D(t_index, m_nu, bound, v_0)
    densities = jax.vmap(nu_dens_fctn)(jnp.arange(len(times)))
    min_density = jnp.min(densities)
    mod = (densities - min_density) / (densities + min_density) * 100

    # return times, mod
    return times, densities


def plot_modulations(which, start_date_str='09-11'):
    """Plot the fractional modulations for different scenarios."""
    fig = plt.figure(figsize=(10, 6))
    fig.tight_layout()
    ax = fig.add_subplot(111)

    # Convert start_date_str to datetime object
    start_date = datetime.datetime.strptime(start_date_str, '%m-%d')
    
    # Calculate the day of the year for the start date
    start_day_of_year = start_date.timetuple().tm_yday

    m_nu_light = 0.15 * Params.eV
    m_nu_heavy = 0.35 * Params.eV
    
    cases = [
        (m_nu_light, False, 220*Params.km/Params.s, 'Unbound, 0.15 eV', 'dashed', 'purple', 0.5),
        (m_nu_heavy, False, 220*Params.km/Params.s, 'Unbound, 0.35 eV', 'dashed', 'blue', 0.5),
        (m_nu_heavy, True, 220*Params.km/Params.s, 'Bound, v_0 = 220 km/s', 'solid', 'black', 0.5),
        (m_nu_heavy, True, 400*Params.km/Params.s, 'Bound, v_0 = 400 km/s', 'solid', 'orange', 0.5)
    ]

    if which == "unbound":
        scenarios = cases[:2]
    elif which == "bound":
        scenarios = cases[2:]
    elif which == "both":
        scenarios = cases

    for m_nu, bound, v_0, label, linestyle, color, alpha in scenarios:
        days, mod = calculate_modulation(m_nu, bound, v_0)

        # Shift days to start from the specified date
        shifted_days = (days/Params.yr*365 - start_day_of_year) % 365 + 1
        sorted_indices = jnp.argsort(shifted_days)
        shifted_days = shifted_days[sorted_indices]
        mod = mod[sorted_indices]

        ax.plot(
            shifted_days, mod, 
            linestyle=linestyle, color=color, label=label, alpha=alpha)

    # Calculate tick positions and labels
    tick_dates = [datetime.date(2000, 11, 1), datetime.date(2000, 2, 1), 
                  datetime.date(2000, 5, 1), datetime.date(2000, 8, 1)]
    tick_days = [
        (date - datetime.date(2000, start_date.month, start_date.day)).days % 365 + 1 for date in tick_dates
    ]
    tick_labels = ['Nov 1', 'Feb 1', 'May 1', 'Aug 1']

    plt.xticks(tick_days, tick_labels)

    # Set y-axis limits
    # plt.ylim(0, 3)

    plt.title('Annual Modulation of Cosmic Relic Neutrino Density')
    plt.ylabel(r'Modulation ($\%$)')
    plt.grid(True, which="major", linestyle="dashed")

    # Add vertical lines for March 12th and September 11th
    march_12 = (datetime.date(2000, 3, 12) - datetime.date(2000, start_date.month, start_date.day)).days % 365 + 1
    sept_11 = (datetime.date(2000, 9, 11) - datetime.date(2000, start_date.month, start_date.day)).days % 365 + 1
    plt.axvline(
        x=march_12, color='dodgerblue', linestyle=':', 
        label=r'$\textbf{Min. unbound}$ (Mar 12th)')
    plt.axvline(
        x=sept_11, color='magenta', linestyle=':', 
        label=r'$\textbf{Max. unbound}$ (Sep 11th)')

    plt.legend(prop={"size":12})

    plt.savefig(
        f"Safdi.pdf",
        bbox_inches="tight")
    

# Run the function with a specific start date
# plot_modulations(which="unbound", start_date_str='09-20')
plot_modulations(which="bound", start_date_str='09-20')
# plot_modulations(which="both", start_date_str='09-20')