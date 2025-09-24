from Shared.shared import *
from Shared.specific_CNB_sim import *


### ==================================== ###
### Get Earth's positions and velocities ###
### ==================================== ###

year = 2023

earth_positions = jnp.array(
    [pos for _, pos in SimUtil.calculate_earth_position(year)])*Params.AU
earth_velocities = jnp.array(
    [vel for _, vel in SimUtil.calculate_earth_velocity(year)])*Params.km/Params.s
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
def f_distr(v_range, t_index, m_nu, bound, v_0):
    """Phase-space distribution at Earth's location."""

    # Get x,y,z coords.
    # v_nu = Utils.v_mag_to_xyz(v_range, Params.key)

    # # Compute v_inf and v argument as used for f(v)
    # r_s = earth_positions[t_index]
    # v_s = v_nu + earth_velocities[t_index]
    # v_inf = SimPlot.v_infinity(v_s, r_s)

    def bound_case(_):

        # v_nu = Utils.v_mag_to_xyz_biased(
        #     v_range, Params.key, -v_Sun, cone_angle_rad=jnp.pi/12)
        v_nu = Utils.v_mag_to_xyz(v_range, Params.key)

        # Compute v_inf and v argument as used for f(v)
        r_s = earth_positions[t_index]
        v_s = v_nu + earth_velocities[t_index]
        v_inf = SimPlot.v_infinity(v_s, r_s)

        v_for_f = v_inf + v_Sun  # original
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

        # v_nu = Utils.v_mag_to_xyz_biased(
        #     v_range, Params.key, -v_CNB, cone_angle_rad=jnp.pi/12)
        v_nu = Utils.v_mag_to_xyz(v_range, Params.key)

        # Compute v_inf and v argument as used for f(v)
        r_s = earth_positions[t_index]
        v_s = v_nu + earth_velocities[t_index]
        v_inf = SimPlot.v_infinity(v_s, r_s)

        v_for_f = v_inf + v_CNB  # original
        v_for_f_mag = jnp.linalg.norm(v_for_f, axis=-1)

        f_v = m_nu**3/(jnp.exp(m_nu*v_for_f_mag/Params.T_CNB)+1)

        return f_v

    f_v = jax.lax.cond(bound, bound_case, unbound_case, operand=None)

    return f_v


@jax.jit
def number_density(t_index, m_nu, bound, v_0):
    """Calculate the neutrino number density at time t (a certain day)."""

    # note: below reso of (0.001, 100, 100_000), curves are wonky
    p_range = jnp.geomspace(0.001, 1000, 1_000_000) * Params.T_CNB
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


def compute_modulations(m_nu_light, m_nu_heavy, out_dir):
    """Compute the annually modulated densities for different scenarios."""
    
    v_unit = Params.km/Params.s
    cases = [
        (m_nu_light, False, 220*v_unit),
        (m_nu_heavy, False, 220*v_unit),
        (m_nu_heavy, True, 220*v_unit),
        (m_nu_heavy, True, 400*v_unit)
    ]

    for triplet in cases:
        m_nu, bound, v_0 = triplet[0], triplet[1], triplet[2]
        _, densities = calculate_modulation(m_nu, bound, v_0)

        if bound == False:
            jnp.save(
                f"{out_dir}/annual_densities_{m_nu}eV_unbound.npy",  # orig
                densities)
        if bound == True:
            jnp.save(
                f"{out_dir}/annual_densities_{v_0/(Params.km/Params.s)}kms_{m_nu}eV_bound.npy",  # orig
                densities)


m_nu_l = 0.15
m_nu_h = 0.35
# compute_modulations(m_nu_l, m_nu_h)
# compute_modulations(m_nu_l, m_nu_h)
compute_modulations(m_nu_l, m_nu_h, out_dir="sim_output/SunMod_1k/annual_densities_analytical")
