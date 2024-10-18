from Shared.shared import *
from Shared.specific_CNB_sim import *


def calculate_fractional_day_numbers(year):
    """
    Calculate the fractional day number n for each day of the given year.
    
    :param year: The year for which to calculate the fractional day numbers
    :return: A list of tuples, each containing (date, fractional day number)
    """
    results = []
    start_date = datetime(year, 1, 1)
    
    for day in range(
        366 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 365):
        current_date = start_date + timedelta(days=day)
        Y, M, D = current_date.year, current_date.month, current_date.day
        
        Y_tilde = Y - 1 if M <= 2 else Y
        M_tilde = M + 12 if M <= 2 else M
        
        n = np.floor(365.25 * Y_tilde) + np.floor(30.61 * (M_tilde + 1)) + D - 730563.5
        
        results.append((current_date.strftime('%Y-%m-%d'), n))
    
    return results


def calculate_average_ecliptic_vectors(year):
    """
    Calculate the average ecliptic unit vectors ϵ_x and ϵ_y for the given year.
    
    :param year: The year for which to calculate the average ecliptic vectors
    :return: A tuple containing (year, average ϵ_x, average ϵ_y)
    """
    # Constants for ϵ_x and ϵ_y calculations
    eps_x_0 = np.array([0.054876, -0.494109, 0.867666])
    eps_x_T = np.array([-0.024232, -0.002689, 1.546e-6])
    eps_y_0 = np.array([0.993821, 0.110992, 0.000352])
    eps_y_T = np.array([0.001316, -0.011851, 0.021267])
    
    eps_x_sum = np.zeros(3)
    eps_y_sum = np.zeros(3)
    
    # Get fractional day numbers for the year
    fractional_days = calculate_fractional_day_numbers(year)
    
    for _, n in fractional_days:
        # Calculate T
        T = n / 36525
        
        # Calculate ϵ_x and ϵ_y for this day
        eps_x = eps_x_0 + eps_x_T * T
        eps_y = eps_y_0 + eps_y_T * T
        
        # Add to the sum
        eps_x_sum += eps_x
        eps_y_sum += eps_y
    
    # Calculate averages
    days_in_year = len(fractional_days)
    eps_x_avg = eps_x_sum / days_in_year
    eps_y_avg = eps_y_sum / days_in_year
    
    # Normalize the average vectors
    eps_x_avg /= np.linalg.norm(eps_x_avg)
    eps_y_avg /= np.linalg.norm(eps_y_avg)
    
    return eps_x_avg, eps_y_avg


def calculate_earth_position(year):
    """
    Calculate Earth's position relative to the Sun in ecliptic coordinates for each day of the year.
    
    :param year: The year for which to calculate Earth's position
    :return: A list of tuples, each containing (date, position vector)
    """
    # Constants
    e = 0.9574  # eccentricity in degrees
    e_rad = np.radians(e)

    # Get fractional day numbers and ecliptic vectors for the year
    fractional_days = calculate_fractional_day_numbers(year)
    eps_x, eps_y = calculate_average_ecliptic_vectors(year)

    results = []

    for date, n in fractional_days:
        # Calculate L, g, and varpi
        L = np.radians(280.460 + 0.9856474 * n)
        g = np.radians(357.528 + 0.9856003 * n)
        varpi = np.radians(282.932 + 0.0000471 * n)

        # Calculate ecliptic longitude
        l = L + 2 * e_rad * np.sin(g) + 5/4 * e_rad**2 * np.sin(2*g)

        # Calculate Earth-Sun distance (in AU)
        r = 1.00014 - 0.01671 * np.cos(g) - 0.00014 * np.cos(2*g)

        # Calculate position vector
        position = r * (np.cos(l) * eps_x + np.sin(l) * eps_y)

        results.append((date, position))

    return results


def calculate_earth_velocity(year):
    """
    Calculate Earth's velocity for each day of the year.
    
    :param year: The year for which to calculate Earth's velocity
    :return: A list of tuples, each containing (date, velocity vector)
    """
    # Constants
    e = 0.9574  # eccentricity in degrees
    e_rad = np.radians(e)
    u_E_avg = 29.79  # Average Earth velocity in km/s

    # Get fractional day numbers and ecliptic vectors for the year
    fractional_days = calculate_fractional_day_numbers(year)
    eps_x, eps_y = calculate_average_ecliptic_vectors(year)

    results = []

    for date, n in fractional_days:
        # Calculate L and varpi
        L = np.radians(280.460 + 0.9856474 * n)
        varpi = np.radians(282.932 + 0.0000471 * n)

        # Calculate velocity components
        v_x = -u_E_avg * (np.sin(L) + e_rad * np.sin(2*L - varpi))
        v_y = u_E_avg * (np.cos(L) + e_rad * np.cos(2*L - varpi))

        # Calculate velocity vector
        velocity = v_x * eps_x + v_y * eps_y

        results.append((date, velocity))

    return results



### ==================================== ###
### Get Earth's positions and velocities ###
### ==================================== ###

year = 2023

earth_positions = jnp.array(
    [pos for _, pos in calculate_earth_position(year)])*Params.AU
earth_velocities = jnp.array(
    [vel for _, vel in calculate_earth_velocity(year)])*Params.km/Params.s
# print(earth_positions.shape, earth_velocities.shape)
# print(earth_positions[0], jnp.linalg.norm(earth_velocities[0]))

# Generate time points for one year
times = jnp.linspace(0, Params.yr, 365)
jd_times = Time('2023-01-01').jd + times / Params.yr * 365

# Sun's velocity in CNB(==CMB) frame
v_CNB = jnp.array([-0.0695, -0.662, 0.747])*369*Params.km/Params.s

# Approximate escape velocity for MW
v_esc_MW = 550*Params.km/Params.s

# Sun's velocity in Galactic frame
v_Sun = jnp.array([11, 232, 7])*Params.km/Params.s


@jax.jit
def v_infinity(v_s, r_s):
    """
    Calculate the initial Solar-frame velocity for particles, s.t. they have velocity v_s at Earth's location.
    """
    
    v_GM = 2*Params.G*Params.Msun/jnp.linalg.norm(r_s, axis=-1)
    v_inf2 = jnp.linalg.norm(v_s, axis=-1)**2 - v_GM
    v_inf = jnp.sqrt(jnp.maximum(0, v_inf2))
    # v_inf = jnp.sqrt(v_inf2)
    r_s_unit = r_s / jnp.linalg.norm(r_s, axis=-1)
    vr_s_dot = jnp.dot(v_s, r_s_unit)

    numer = v_inf2[..., None]*v_s + v_inf[..., None]*v_GM/2*r_s_unit - v_inf[..., None]*v_s*vr_s_dot[..., None]
    denom = v_inf2 + v_GM/2 - v_inf*vr_s_dot
    
    return numer / denom[..., None]


@jax.jit
def f_distr(v_range, t_index, m_nu, bound, v_0):
    """Phase-space distribution at Earth's location."""

    # Get x,y,z coords.
    v_nu = Utils.v_mag_to_xyz(v_range, Params.key)

    # Compute v_inf and v argument as used for f(v)
    r_s = earth_positions[t_index]
    v_s = v_nu + earth_velocities[t_index]
    v_inf = v_infinity(v_s, r_s)

    def bound_case(_):

        v_for_f = v_inf + v_Sun
        v_for_f_mag = jnp.linalg.norm(v_for_f, axis=-1)

        # Create mask for velocity magnitudes smaller than escape velocity
        mask = v_for_f_mag < v_esc_MW

        # Compute standard halo model (SHM) distribution
        exp_term = jnp.exp(-v_for_f_mag**2/v_0**2)
        f_v = jnp.where(
            mask, 
            jnp.power(jnp.pi*v_0**2, -3/2) * exp_term, 
            jnp.zeros_like(exp_term))
        
        # Normalisation constant, and normalised velocity distribution
        z = v_esc_MW/v_0
        N_esc = jsp.special.erf(z) - 2/jnp.sqrt(jnp.pi)*z*jnp.exp(-z**2)
        f_v_normalized = f_v / N_esc

        return Params.N0 * f_v_normalized

    def unbound_case(_):

        v_for_f = v_inf + v_CNB
        v_for_f_mag = jnp.linalg.norm(v_for_f, axis=-1)

        f_v = m_nu**3/(jnp.exp(m_nu*v_for_f_mag/Params.T_CNB)+1)

        return f_v

    f_v = jax.lax.cond(bound, bound_case, unbound_case, operand=None)

    return f_v

@jax.jit
def number_density(t_index, m_nu, bound, v_0):
    """Calculate the neutrino number density at time t (a certain day)."""

    # note: below reso of (0.001, 100, 100_000), curves are wonky
    p_range = jnp.geomspace(0.001, 1000, 10_000_000) * Params.T_CNB
    v_range = p_range / m_nu

    def bound_case(_):
        
        f_v = f_distr(v_range, t_index, m_nu, bound, v_0)
        integrand = v_range**3 * f_v
        integral = trap(integrand, x=jnp.log(v_range), axis=-1)
        nu_dens_cm3 = integral/(2*jnp.pi**2)/Params.cm**-3
        return nu_dens_cm3

    def unbound_case(_):

        f_v = f_distr(v_range, t_index, m_nu, bound, v_0)
        integrand = v_range**3 * f_v
        integral = trap(integrand, x=jnp.log(v_range), axis=-1)
        nu_dens_cm3 = integral/(2*jnp.pi**2)/Params.cm**-3
        return nu_dens_cm3
    
    nu_dens = jax.lax.cond(bound, bound_case, unbound_case, operand=None)

    return nu_dens

@jax.jit
def calculate_modulation(m_nu, bound, v_0):
    """Calculate the fractional modulation throughout the year."""
    
    num_days = len(times)
    densities = jnp.zeros(num_days)
    
    def body_fun(i, densities):
        density = number_density(i, m_nu, bound, v_0)
        return densities.at[i].set(density)
    
    densities = jax.lax.fori_loop(0, num_days, body_fun, densities)

    return times, densities

def plot_modulations(which, start_date_str='09-11'):
    """Plot the fractional modulations for different scenarios."""
    fig = plt.figure(figsize=(10, 6))
    fig.tight_layout()
    ax = fig.add_subplot(111)

    # Convert start_date_str to datetime object
    start_date = datetime.strptime(start_date_str, '%m-%d')
    
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
        days, densities = calculate_modulation(m_nu, bound, v_0)

        if bound == False:
            jnp.save(
                f"annual_densities_{m_nu}eV_unbound.npy", 
                densities)
        if bound == True:
            jnp.save(
                f"annual_densities_{v_0/(Params.km/Params.s)}kms_bound.npy", 
                densities)

        min_density = jnp.min(densities)
        mod = (densities - min_density) / (densities + min_density) * 100

        # Check if all elements are zero
        if jnp.all(mod == 0):
            print("all zeros")

        # Check if all elements are nan
        if jnp.all(jnp.isnan(mod)):
            print("all nan")

        # Shift days to start from the specified date
        shifted_days = (days/Params.yr*365 - start_day_of_year) % 365 + 1
        sorted_indices = jnp.argsort(shifted_days)
        shifted_days = shifted_days[sorted_indices]
        mod = mod[sorted_indices]

        ax.plot(
            shifted_days, mod, 
            linestyle=linestyle, color=color, label=label, alpha=alpha)

    # Calculate tick positions and labels
    tick_dates = [datetime_date(2000, 11, 1), datetime_date(2000, 2, 1), 
                datetime_date(2000, 5, 1), datetime_date(2000, 8, 1)]
    tick_days = [(d - datetime_date(2000, start_date.month, start_date.day)).days % 365 + 1 for d in tick_dates]

    tick_labels = ['Nov 1', 'Feb 1', 'May 1', 'Aug 1']

    plt.xticks(tick_days, tick_labels)

    # Set y-axis limits
    # plt.ylim(0, 3)

    plt.title('Annual Modulation of Cosmic Relic Neutrino Density')
    plt.ylabel(r'Modulation ($\%$)')
    plt.grid(True, which="major", linestyle="dashed")
    
    if which == "unbound" or which == "both":
        # Add vertical lines for ~March 12th and ~September 11th
        march_12 = (datetime_date(2000, 3, 12) - datetime_date(2000, start_date.month, start_date.day)).days % 365 + 1
        sept_11 = (datetime_date(2000, 9, 11) - datetime_date(2000, start_date.month, start_date.day)).days % 365 + 1

        plt.axvline(
            x=march_12, color='dodgerblue', linestyle=':', 
            label=r'$\textbf{Min. Unbound}$ (Mar 12th)')
        plt.axvline(
            x=sept_11, color='magenta', linestyle=':', 
            label=r'$\textbf{Max. Unbound}$ (Sep 11th)')
        
    elif which == "bound" or which == "both":
        # Add vertical lines for ~March 1st and ~September 1st
        march_1 = (datetime_date(2000, 3, 1) - datetime_date(2000, start_date.month, start_date.day)).days % 365 + 1
        sept_1 = (datetime_date(2000, 9, 1) - datetime_date(2000, start_date.month, start_date.day)).days % 365 + 1

        # Plot the vertical lines
        plt.axvline(
            x=march_1, color='red', linestyle=':', 
            label=r'$\textbf{Max. Bound}$ (Mar 1st)')
        plt.axvline(
            x=sept_1, color='purple', linestyle=':', 
            label=r'$\textbf{Min. Bound}$ (Sep 1st)')


    plt.legend(prop={"size":12})

    plt.savefig(
        f"Safdi.pdf",
        bbox_inches="tight")

# Run the function with a specific start date
# plot_modulations(which="unbound", start_date_str='09-20')
# plot_modulations(which="bound", start_date_str='09-20')
plot_modulations(which="both", start_date_str='09-20')