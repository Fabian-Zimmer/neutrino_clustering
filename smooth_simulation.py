from shared.preface import *
import shared.functions as fct


def EOMs(s_val, y):
    """Equations of motion for all x_i's and u_i's in terms of s."""

    # Initialize vector.
    x_i, u_i = np.reshape(y, (2,3))


    #! Switch to numerical reality here.
    x_i *= kpc
    u_i *= (kpc/s)


    # Find z corresponding to s via interpolation.
    z_interp = np.interp(s_val, S_STEPS, ZEDS)

    # For testing and comparing with CubeSpace simulation.
    idx = np.abs(ZEDS_SNAPSHOTS - z_interp).argmin()
    z = ZEDS_SNAPSHOTS[idx]

    # Sum gradients of each halo.
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


    #! Switch to physical reality here.
    grad_tot /= (kpc/s**2)
    x_i /= kpc
    u_i /= (kpc/s)


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
    
    np.save(f'{sim_folder}/nu_{int(Nr)}.npy', np.array(sol.y.T))


if __name__ == '__main__':
    start = time.perf_counter()
    
    sim_folder = 'LinfNinf'

    # Draw initial velocities.
    ui = fct.draw_ui(
        phi_points   = PHIs,
        theta_points = THETAs
        )
    
    # Combine vectors and append neutrino particle number.
    y0_Nr = np.array(
        [np.concatenate((X_SUN, ui[i], [i+1])) for i in range(NUS)]
        )


    halos = 'MW'*MW_HALO + '+VC'*VC_HALO + '+AG'*AG_HALO
    CPUs = 6

    # Print out all relevant parameters for simulation.
    print(
        '***Running simulation*** \n',
        f'neutrinos={NUS} ; solver={SOLVER} ; halos={halos} ; CPUs={CPUs}'
    )


    # Test 1 neutrino only.
    # backtrack_1_neutrino(y0_Nr[0])


    # Run simulation on multiple cores, in batches.
    #note: important for Rk45 solver, where memory of process increases alot
    batch_size = 2000
    ticks = np.arange(0, NUS/batch_size, dtype=int)
    for i in ticks:

        id_min = (i*batch_size) + 1
        id_max = ((i+1)*batch_size) + 1
        print(f'From {id_min} to and incl. {id_max-1}')

        if i == 0:
            id_min = 0

        with ProcessPoolExecutor(CPUs) as ex:
            ex.map(backtrack_1_neutrino, y0_Nr[id_min:id_max])

        print(f'Batch {i+1}/{len(ticks)} done!')


    # Compactify all neutrino vectors into 1 file.
    Ns = np.arange(NUS, dtype=int)  # Nr. of neutrinos
    nus = np.array([np.load(f'{sim_folder}/nu_{Nr+1}.npy') for Nr in Ns])
    
    np.save(
        f'{sim_folder}/nus_{NUS}_halos_{halos}_{SOLVER}.npy',
        nus
        )
    
    fct.delete_temp_data(f'{sim_folder}/nu_*.npy')    

    seconds = time.perf_counter()-start
    minutes = seconds/60.
    hours = minutes/60.
    print(f'Time min/h: {minutes} min, {hours} h.')