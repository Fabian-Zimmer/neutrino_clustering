from Shared.specific_CNB_sim import *

# Argparse inputs.
parser = argparse.ArgumentParser()
parser.add_argument('--directory', required=True)
parser.add_argument(
    '--pixel_densities', required=True, action=argparse.BooleanOptionalAction)
parser.add_argument(
    '--total_densities', required=True, action=argparse.BooleanOptionalAction)
parser.add_argument(
    '--testing', required=True, action=argparse.BooleanOptionalAction)
pars = parser.parse_args()


@jax.jit
def EOMs_sun(s_val, y, args):

    # Unpack the input data
    s_int_steps, z_int_steps, kpc, s = args

    # Initialize vector.
    x_i, u_i = y

    # Switch to "numerical reality" here.
    x_i *= kpc
    u_i *= (kpc/s)

    # Find z corresponding to s via interpolation.
    z = Utils.jax_interpolate(s_val, s_int_steps, z_int_steps)


    #?# compute gradient of sun
    grad_tot = SimExec.sun_gravity(x_i)


    # Switch to "physical reality" here.
    grad_tot /= (kpc/s**2)
    x_i /= kpc
    u_i /= (kpc/s)


    # Hamilton eqns. for integration (global minus, s.t. we go back in time).
    dyds = -jnp.array([
        u_i, 1./(1.+z)**2 * grad_tot
    ])

    return dyds



@jax.jit
def backtrack_1_neutrino(
    init_vector, s_int_steps, z_int_steps, zeds_snaps, 
    snaps_GRID_L, snaps_DM_com, snaps_DM_num, snaps_QJ_abs, 
    dPsi_grid_data, cell_grid_data, cell_gens_data, DM_mass, kpc, s):

    """
    Simulate trajectory of 1 neutrino. Input is 6-dim. vector containing starting positions and velocities of neutrino. Solves ODEs given by the EOMs function with an jax-accelerated integration routine, using the diffrax library. Output are the positions and velocities at each timestep, which was specified with diffrax.SaveAt. 
    """

    # Initial vector in correct shape for EOMs function
    y0 = init_vector.reshape(2,3)

    # ODE solver setup
    term = diffrax.ODETerm(EOMs_sun)
    t0 = s_int_steps[0]
    t1 = s_int_steps[-1]
    dt0 = (s_int_steps[0] + s_int_steps[1]) / 1000
    

    ### ------------- ###
    ### Dopri5 Solver ###
    ### ------------- ###
    solver = diffrax.Dopri5()
    stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
    # note: no change for tighter rtol and atol, e.g. rtol=1e-5, atol=1e-9


    ### ------------- ###
    ### Dopri8 Solver ###
    ### ------------- ###
    # solver = diffrax.Dopri8()
    # stepsize_controller = diffrax.PIDController(rtol=1e-7, atol=1e-9)


    # Specify timesteps where solutions should be saved
    saveat = diffrax.SaveAt(ts=jnp.array(s_int_steps))
    
    # Solve the coupled ODEs, i.e. the EOMs of the neutrino
    sol = diffrax.diffeqsolve(
        term, solver, 
        t0=t0, t1=t1, dt0=dt0, y0=y0, max_steps=10000,
        saveat=saveat, stepsize_controller=stepsize_controller,
        args=( 
            s_int_steps, z_int_steps, zeds_snaps, snaps_GRID_L, snaps_DM_com, snaps_DM_num, snaps_QJ_abs, dPsi_grid_data, cell_grid_data, cell_gens_data, DM_mass, kpc, s)
    )
    
    trajectory = sol.ys.reshape(100,6)

    return jnp.stack([trajectory[0], trajectory[-1]])