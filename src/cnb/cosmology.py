"""Cosmological distances travelled by relic neutrinos."""


import jax
import jax.numpy as jnp
from jax.scipy.integrate import trapezoid as trap
import chex

from cnb.physics import Physics
from cnb.utils import Utils


@chex.dataclass
class Cosmo:


    @staticmethod
    @jax.jit
    def get_Chi_integrand(z, p_0, m_nu, args):
        
        # Comoving distance integral for massive particles
        p_div_H = p_0/Physics.Hubble_z(z, args)
        sqrt_term = jnp.sqrt(p_0**2 + m_nu**2/((1+z)**2))

        return p_div_H / sqrt_term


    @staticmethod
    @jax.jit
    def get_LSS_distance(z_LSS, p_0, m_nu, args):
    # def CALC_LSS_distance(z_LSS, p_0=1.0, m_nu=0.0):

        # Logarithmically spaced redshifts starting at zero, ending at z_LSS
        z_int_range = Utils.make_logz_arr(0, z_LSS, 100)

        # Comoving distance (Chi) to last scattering surface (LSS)
        integrand = Cosmo.get_Chi_integrand(z_int_range[:, None], p_0, m_nu, args)
        Chi_LSS =  trap(integrand, z_int_range, axis=0)

        return Chi_LSS
