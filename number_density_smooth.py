from shared.preface import *
import shared.functions as fct


# Initialize parameters and files.
PRE = PRE(
    sim='LinfNinf', phis=10, thetas=10, vels=100,
    sim_CPUs=128, MW_HALO=True
)

# Make temporary folder to store files, s.t. parallel runs don't clash.
rand_code = ''.join(
    random.choices(string.ascii_uppercase + string.digits, k=4)
)
TEMP_DIR = f'{PRE.OUT_DIR}/temp_data_{rand_code}'
os.makedirs(TEMP_DIR)


def EOMs(s_val, y):
    """Equations of motion for all x_i's and u_i's in terms of s."""

    # Initialize vector.
    x_i, u_i = np.reshape(y, (2,3))

    # Switch to "numerical reality" here.
    x_i *= kpc
    u_i *= (kpc/s)

    # Find z corresponding to s via interpolation.
    z = np.interp(s_val, S_STEPS, ZEDS)

    # Sum gradients of each halo. Seperate if statements, for adding any halos.
    grad_tot = np.zeros(len(x_i))
    if MW_HALO:
        grad_tot += fct.dPsi_dxi_NFW(
            x_i, z, rho0_MW, Mvir_MW, Rvir_MW, Rs_MW, 'MW'
            )
    if VC_HALO:
        grad_tot += fct.dPsi_dxi_NFW(
            x_i, z, rho0_VC, Mvir_VC, Rvir_VC, Rs_VC, 'VC'
            )
    if AG_HALO:
        grad_tot += fct.dPsi_dxi_NFW(
            x_i, z, rho0_AG, Mvir_AG, Rvir_AG, Rs_AG, 'AG'
            )

    # Switch to "physical reality" here.
    grad_tot /= (kpc/s**2)
    x_i /= kpc
    u_i /= (kpc/s)

    # Hamilton eqns. for integration.
    dyds = -np.array([
        u_i, 1./(1.+z)**2 * grad_tot
    ])

    return dyds


def backtrack_1_neutrino(y0_Nr):
    """Simulate trajectory of 1 neutrino."""

    # Split input into initial vector and neutrino number.
    y0, Nr = y0_Nr[0:-1], y0_Nr[-1]

    # Solve all 6 EOMs.
    sol = solve_ivp(
        fun=EOMs, t_span=[S_STEPS[0], S_STEPS[-1]], t_eval=S_STEPS,
        y0=y0, method=SOLVER, vectorized=True
        )
    
    np.save(f'{TEMP_DIR}/nu_{int(Nr)}.npy', np.array(sol.y.T))


if __name__ == '__main__':
    start = time.perf_counter()

    # Draw initial velocities.
    ui = fct.init_velocities(PRE.PHIs, PRE.THETAs, PRE.MOMENTA)
    
    # Combine vectors and append neutrino particle number.
    y0_Nr = np.array(
        [np.concatenate((X_SUN, ui[k], [k+1])) for k in range(PRE.NUS)]
        )


    sim_testing = False

    if sim_testing:
        # Test 1 neutrino only.
        backtrack_1_neutrino(y0_Nr[0])
        backtrack_1_neutrino(y0_Nr[1])
        backtrack_1_neutrino(y0_Nr[2])

    else:

        # Run simulation on multiple cores, in batches.
        # (important for other solvers (e.g. Rk45), due to memory increase)
        batch_size = 10000
        ticks = np.arange(0, PRE.NUS/batch_size, dtype=int)
        for i in ticks:

            id_min = (i*batch_size) + 1
            id_max = ((i+1)*batch_size) + 1
            print(f'From {id_min} to and incl. {id_max-1}')

            if i == 0:
                id_min = 0

            with ProcessPoolExecutor(PRE.SIM_CPUs) as ex:
                ex.map(backtrack_1_neutrino, y0_Nr[id_min:id_max])

            print(f'Batch {i+1}/{len(ticks)} done!')


        # Compactify all neutrino vectors into 1 file.
        Ns = np.arange(PRE.NUS, dtype=int)
        nus = np.array([np.load(f'{TEMP_DIR}/nu_{Nr+1}.npy') for Nr in Ns])
        oname = f'{PRE.NUS}nus_smooth_{PRE.HALOS}_{SOLVER}'
        np.save(f'{PRE.OUT_DIR}/{oname}.npy', nus)


    # Remove temporary folder with all individual neutrino files.
    shutil.rmtree(TEMP_DIR)   

    seconds = time.perf_counter()-start
    minutes = seconds/60.
    hours = minutes/60.
    print(f'Time min/h: {minutes} min, {hours} h.')