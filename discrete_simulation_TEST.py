from shared.preface import *
import shared.functions as fct


def EOMs(s_val, y):

    # Initialize vector.
    x_i, u_i = np.reshape(y, (2,3))

    #! Switch to numerical reality here.
    x_i *= kpc
    u_i *= (kpc/s)

    # Find z corresponding to s via interpolation.
    z = np.interp(s_val, S_STEPS, ZEDS)

    '''
    # Find which (pre-calculated) derivative grid to use at current z.
    #NOTE: Derivative grid of each snapshot will be a stored file.
    Psi_grid = fct.load_derivative_grid(z)
    pos_grid = fct.load_grid_positions(z)

    # Find gradient at neutrino position, i.e. for corresponding cell.
    cell_idx = fct.nu_in_which_cell(x_i, pos_grid)

    # Get derivative of cell.
    grad_tot = Psi_grid[cell_idx,:]
    
    #! Switch to physical reality here.
    grad_tot /= (kpc/s**2)
    '''
    x_i /= kpc
    u_i /= (kpc/s)


    dyds = TIME_FLOW * np.array([
        u_i, 1./(1.+z)**2 * grad_tot
    ])

    return dyds