"""Time steps for the solar modulation runs (simulate.mode: solar_modulation).

Redshift, integration variable s and lookback time t for each day of the past
two years (plus one day), with the Planck18 cosmology of Params. Saves
z/s/t_int_steps_2years.npy and the first 365 days as *_1year.npy in run_dir.
Previously done in notebook sim_plots41_solar_modulation_background.

Usage: python scripts/make_solar_time_steps.py configs/<run>.yaml
"""

import astropy.units as apu
import jax.numpy as jnp
import numpy as np
from astropy.time import Time
from scipy.optimize import root_scalar

from cnb.config import config_from_cli
from cnb.physics import Physics
from cnb.units import Params, cosmo


def z_at_age(age):

    # Function to calculate redshift for a given age of the universe
    def age_diff(z):
        return cosmo.age(z).to(apu.year).value - age

    # Use root_scalar to find the redshift that gives the specified age
    sol = root_scalar(
        age_diff, bracket=[0, 10], method='brentq', xtol=1e-12, rtol=1e-12)
    return sol.root


def get_redshift_array(days):

    # Today's date
    today = Time.now()

    # Calculate redshift for each day
    redshifts = np.zeros(days)
    age_today = cosmo.age(0).to(apu.year).value  # Age of the universe today
    for day in range(days):

        # Lookback time of each day of the past, in years
        past_date = today - day*apu.day
        lookback_time = (today - past_date).to(apu.year).value

        # Redshift corresponding to the age of the universe at that time
        redshifts[day] = z_at_age(age_today - lookback_time)

    return jnp.array(redshifts)


def make_solar_time_steps(run_dir):

    # 2 years plus additional day, s.t. integration routine can pick 2nd last
    # (see notes in Simulation.backtrack_1_neutrino)
    days = (365*2)+1

    z_int_steps_2years = get_redshift_array(days)

    # Convert to integration variable s
    s_int_steps_2years = jnp.array([
        Physics.s_of_z(z, Params.h, Params.Omega_M, Params.Omega_L)
        for z in z_int_steps_2years])

    # Convert to "physical" time in seconds (lookback time)
    t_int_steps_2years = jnp.array(
        cosmo.lookback_time(np.array(z_int_steps_2years)).to(apu.s).value)

    for name, arr in (("z", z_int_steps_2years), ("s", s_int_steps_2years), ("t", t_int_steps_2years)):
        jnp.save(f"{run_dir}/{name}_int_steps_2years.npy", arr)
        jnp.save(f"{run_dir}/{name}_int_steps_1year.npy", arr[:365])


if __name__ == "__main__":
    make_solar_time_steps(config_from_cli(__doc__)["run_dir"])
