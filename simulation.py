from shared.preface import *
import shared.functions as fct


def draw_ui(phi_points, theta_points):
    """Get initial velocities for the neutrinos."""

    # Convert momenta to initial velocity magnitudes ("in units of [1]")
    v_kpc = MOMENTA / NU_MASS / (kpc/s)

    # Split up this magnitude into velocity components.
    #NOTE: Done by using spher. coords. trafos, which act as "weights".

    eps = 0.01  # shift in theta, so poles are not included
    ps = np.linspace(0., 2.*Pi, phi_points)
    ts = np.linspace(0.+eps, Pi-eps, theta_points)

    # Minus signs due to choice of coord. system setup (see notes/drawings).
    uxs = [-v*np.cos(p)*np.sin(t) for p in ps for t in ts for v in v_kpc]
    uys = [-v*np.sin(p)*np.sin(t) for p in ps for t in ts for v in v_kpc]
    uzs = [-v*np.cos(t) for _ in ps for t in ts for v in v_kpc]

    ui_array = np.array([[ux, uy, uz] for ux,uy,uz in zip(uxs,uys,uzs)])        

    return ui_array 


def EOMs(s_val, y):
    """Equations of motion for all x_i's and u_i's in terms of s."""

    # Initialize vector and attach astropy units.
    x_i, u_i = np.reshape(y, (2,3))


    #! Switch to numerical reality here.
    x_i *= kpc
    u_i *= (kpc/s)


    # Find z corresponding to s.
    if s_val in s_steps:
        z = ZEDS[s_steps==s_val][0]
    else:
        z = s_to_z(s_val)  # interpolation function defined below

    # Sum gradients of each halo.
    grad_tot = np.zeros(len(x_i))
    if MW_HALO:
        grad_tot += fct.dPsi_dxi_NFW(
            x_i, z, rho0_MW, Mvir_MW, Rvir_MW, Rs_MW, 'MW'
            )
    elif AG_HALO:
        grad_tot += fct.dPsi_dxi_NFW(
            x_i, z, rho0_AG, Mvir_AG, Rvir_AG, Rs_AG, 'AG'
            )
    elif VC_HALO:
        grad_tot += fct.dPsi_dxi_NFW(
            x_i, z, rho0_VC, Mvir_VC, Rvir_VC, Rs_VC, 'VC'
            )


    #! Switch to physical reality here.
    grad_tot /= (kpc/s**2)
    x_i /= kpc
    u_i /= (kpc/s)

    if np.all(grad_tot==0.):
        dyds = TIME_FLOW * np.array([
            u_i, np.zeros(3)
        ])

    else:
        #NOTE: Velocity has to change according to the pointing direction,
        #NOTE: treat all 4 cases seperately.
        signs = np.zeros(3)
        for i, (pos, vel) in enumerate(zip(x_i, u_i)):
            if pos > 0. and vel > 0.:
                signs[i] = -1.
            elif pos > 0. and vel < 0.:
                signs[i] = -1.
            elif pos < 0. and vel > 0.:
                signs[i] = +1.
            else:  # pos < 0. and vel < 0.
                signs[i] = +1.
    

        dyds = TIME_FLOW * np.array([
            u_i, signs * 1./(1.+z)**2 * grad_tot
        ])

    return dyds


def backtrack_1_neutrino(y0_Nr):
    """Simulate trajectory of 1 neutrino."""

    # Split input into initial vector and neutrino number.
    y0, Nr = y0_Nr[0:-1], y0_Nr[-1]

    # Solve all 6 EOMs.
    sol = solve_ivp(
        fun=EOMs, t_span=[s_steps[0], s_steps[-1]], t_eval=s_steps,
        y0=y0, method=SOLVER, vectorized=True
        )
    
    np.save(f'neutrino_vectors/nu_{int(Nr)}.npy', np.array(sol.y.T))



if __name__ == '__main__':
    start = time.time()

    # Integration steps.
    s_steps = np.array([fct.s_of_z(z) for z in ZEDS])
    s_to_z = interp1d(s_steps, ZEDS, kind='linear', fill_value='extrapolate')

    # Position of earth w.r.t Milky Way NFW halo center.
    #NOTE: Earth is placed on x axis of coord. system.
    x1, x2, x3 = 8.5, 0., 0.
    x0 = np.array([x1, x2, x3])

    # Draw initial velocities.
    ui = draw_ui(
        phi_points   = PHIs,
        theta_points = THETAs,
        )
    
    # Combine vectors and append neutrino particle number.
    y0_Nr = np.array([np.concatenate((x0,ui[i],[i+1])) for i in range(NUS)])

    # Test 1 neutrino only.
    # backtrack_1_neutrino(y0_Nr[1])

    # Run simulation on multiple cores.
    Processes = 8
    with ProcessPoolExecutor(Processes) as ex:
        ex.map(backtrack_1_neutrino, y0_Nr)  


    # Compactify all neutrino vectors into 1 file.
    Ns = np.arange(NUS, dtype=int)  # Nr. of neutrinos
    nus = np.array([np.load(f'neutrino_vectors/nu_{Nr+1}.npy') for Nr in Ns])
    
    halos = 'MW'*MW_HALO + '+VC'*VC_HALO + '+AG'*AG_HALO
    np.save(
        f'neutrino_vectors/nus_{NUS}_halos_{halos}.npy',
        nus
        )
    
    # hdf5 file format has same size as npy...
    # with h5py.File(f'neutrino_vectors/nus_{NUS}_halos_{HALOS}.hdf5', 'w') as f:
        # dset = f.create_dataset('nus', data=nus)
    
    fct.delete_temp_data('neutrino_vectors/nu_*.npy')    

    seconds = time.time()-start
    minutes = seconds/60.
    hours = minutes/60.
    print(f'Time sec/min/h: {seconds} sec, {minutes} min, {hours} h.')