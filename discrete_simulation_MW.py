from shared.preface import *
import shared.functions as fct


def EOMs(s_val, y):

    # Initialize vector.
    x_i, u_i = np.reshape(y, (2,3))

    # Switch to "numerical reality" here.
    x_i *= kpc
    u_i *= (kpc/s)

    # Find z corresponding to s via interpolation.
    z = np.interp(s_val, S_STEPS, ZEDS)

    # Find which (pre-calculated) derivative grid to use at current z.
    dPsi_grid = fct.load_grid(z, SIM_ID, 'derivatives')
    cell_grid = fct.load_grid(z, SIM_ID, 'positions')

    # Find gradient at neutrino position, i.e. for corresponding cell.
    cell_idx = fct.nu_in_which_cell(x_i, cell_grid)

    # Get derivative of cell.
    grad_tot = dPsi_grid[cell_idx,:]
    
    # Switch to "physical reality" here.
    grad_tot /= (kpc/s**2)
    x_i /= kpc
    u_i /= (kpc/s)

    # Hamilton eqns. for integration.
    dyds = TIME_FLOW * np.array([
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
    
    np.save(f'neutrino_vectors/nu_{int(Nr)}_CubeSpace.npy', np.array(sol.y.T))



if __name__ == '__main__':
    start = time.perf_counter()
    
    # Integration steps.
    S_STEPS = np.array([fct.s_of_z(z) for z in ZEDS])

    # Draw initial velocities.
    ui = fct.draw_ui(
        phi_points   = PHIs,
        theta_points = THETAs
        )
    
    # Combine vectors and append neutrino particle number.
    y0_Nr = np.array(
        [np.concatenate((X_SUN, ui[i], [i+1])) for i in range(NUS)]
        )


    CPUs = 16

    # Display parameters for simulation.
    print(
        '***Running simulation*** \n',
        f'neutrinos={NUS} ; method=CubeSpace ; CPUs={CPUs} ; solver={SOLVER}'
    )

    # Test 1 neutrino only.
    # backtrack_1_neutrino(y0_Nr[0])

    # '''
    # Run simulation on multiple cores.
    with ProcessPoolExecutor(CPUs) as ex:
        ex.map(backtrack_1_neutrino, y0_Nr)  


    # Compactify all neutrino vectors into 1 file.
    Ns = np.arange(NUS, dtype=int)  # Nr. of neutrinos
    nus = np.array([np.load(f'neutrino_vectors/nu_{Nr+1}_CubeSpace.npy') for Nr in Ns])
    
    np.save(
        f'neutrino_vectors/nus_{NUS}_CubeSpace.npy',
        nus
        )
    
    fct.delete_temp_data('neutrino_vectors/nu_*CubeSpace.npy')    
    # '''

    seconds = time.perf_counter()-start
    minutes = seconds/60.
    hours = minutes/60.
    print(f'Time sec/min/h: {seconds} sec, {minutes} min, {hours} h.')