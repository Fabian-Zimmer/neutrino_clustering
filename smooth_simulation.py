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
    z = np.interp(s_val, S_STEPS, ZEDS)

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
    
    np.save(f'neutrino_vectors/nu_{int(Nr)}.npy', np.array(sol.y.T))
    gc.collect()

if __name__ == '__main__':
    start = time.perf_counter()
    
    # Integration steps.
    S_STEPS = np.array([fct.s_of_z(z) for z in ZEDS])

    # Draw initial velocities.
    ui = fct.draw_ui(
        phi_points   = PHIs,
        theta_points = THETAs,
        method = METHOD
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
        f'neutrinos={NUS} ; method={METHOD} ; solver={SOLVER} ; halos={halos} ; CPUs={CPUs}'
    )

    # Test 1 neutrino only.
    # backtrack_1_neutrino(y0_Nr[1])

    # '''
    # Run simulation on multiple cores.
    with ProcessPoolExecutor(CPUs) as ex:
        ex.map(backtrack_1_neutrino, y0_Nr)  


    # Compactify all neutrino vectors into 1 file.
    Ns = np.arange(NUS, dtype=int)  # Nr. of neutrinos
    nus = np.array([np.load(f'neutrino_vectors/nu_{Nr+1}.npy') for Nr in Ns])
    
    np.save(
        f'neutrino_vectors/nus_{NUS}_halos_{halos}.npy',
        nus
        )
    
    fct.delete_temp_data('neutrino_vectors/nu_*.npy')    
    # '''

    seconds = time.perf_counter()-start
    minutes = seconds/60.
    hours = minutes/60.
    print(f'Time sec/min/h: {seconds} sec, {minutes} min, {hours} h.')