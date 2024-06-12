from Shared.shared import *
from Shared.specific_CNB_decay import *
from scipy.optimize import fsolve
# note: Using numpy routines for fsolve compatibility

sim_name = f"no_gravity"
#! marias files are inside the Data/decay folder
decay_dir = f"Data/decay"
sim_folder = f"{decay_dir}/sim_output/{sim_name}"
fig_folder = f"{decay_dir}/figures_local/{sim_name}"
nu_m_range = jnp.load(f"{sim_folder}/neutrino_massrange_eV.npy")
nu_m_picks = jnp.array([0.01, 0.05, 0.1, 0.2, 0.3])*Params.eV
simdata = SimData(sim_folder)

save_dir = f"sim_output/no_gravity_decay_test"

# note: We must have realistic neutrino masses
m_light, m_mid, m_heavy = Physics.neutrino_masses(
    m_lightest=0.05*Params.eV, ordering="NO", args=Params())
print(m_light, m_mid, m_heavy)


def find_nearest(array, value):
    idx = jnp.argmin(jnp.abs(array - value))
    return idx, array[idx]


def rest_frame_quantities(m_h, m_l, m_phi):

    # Energies of daughter particles
    E_l = (m_h**2 + m_l**2 - m_phi**2) / (2*m_h)
    E_phi = (m_h**2 - m_l**2 + m_phi**2) / (2*m_h)

    # Momenta of daughter particles
    p_l = np.sqrt((m_h**2+m_l**2-m_phi**2)**2 - 4*m_h**2*m_l**2) / (2*m_h)
    p_phi = np.sqrt((m_h**2-m_l**2+m_phi**2)**2 - 4*m_h**2*m_phi**2) / (2*m_h)

    return E_l, p_l, E_phi, p_phi


# Rest frame (suscript 0) kinematics are completely fixed by masses
E_l_0, p_l_0, E_phi_0, p_phi_0 = rest_frame_quantities(m_h=m_heavy, m_l=m_light, m_phi=0.0)
# print(f"CHECK 1: ", np.isclose(p_l_0 - p_phi_0, [0.0]))
# print(f"CHECK 2: ", np.isclose(E_l_0 + E_phi_0, [m_heavy]))


def lab_frame_parent_momenta(p_h, p_l_z_target, angle, m_h, E_l_0, p_l_0):

    # Decay angle between axis of boost and p_l_0, i.e. momentum of daughter in rest frame
    # note: angle is defined via unit circle, so 0 deg is antiparallel,
    # note: we subtract pi, s.t. 0 deg corresponds to "straight line decay" in same direction
    cos_theta_0 = np.cos(np.deg2rad(angle) - np.pi)

    # Energy, velocity and Lorentz factor
    E_h = np.sqrt(p_h**2 + m_h**2)
    v_h = p_h/E_h
    gamma = 1/np.sqrt(1 - v_h**2)

    # Daughter momentum in lab frame
    p_l_z = gamma*(p_l_0*cos_theta_0 + v_h*E_l_0)

    # Return output as needed for fsolve to solve for it to become zero
    return p_l_z - p_l_z_target



common_args = (m_heavy, E_l_0, p_l_0)
p_num = 1000  # target: p_num of sim = 1000 (or higher if needed, but base is 1k)
p_l_z_target_range = np.geomspace(0.01, 400, p_num)*Params.T_CNB
a_min = 0
a_max = 180
a_num = 1800  # target: 1800
angles = np.linspace(a_min, a_max, a_num)

p_h_sol = np.empty((len(p_l_z_target_range), len(angles), 2))
for i, p_l_z_target in enumerate(p_l_z_target_range):
    
    for j, angle in enumerate(angles):
        p_h_fsolve = fsolve(
            func=lab_frame_parent_momenta, x0=p_l_z_target,
            args=(p_l_z_target, angle, *common_args))[0]
        
        #! If negative, decay angle is impossible with target daughter momentum
        # if p_h_fsolve < 0.0:
        #     p_h_fsolve = np.nan

        p_h_sol[i,j,0] = angle
        p_h_sol[i,j,1] = p_h_fsolve

print(p_h_sol.shape)
np.save(f"{sim_folder}/allowed_decay_angles_and_momenta.npy", p_h_sol)

p_l_select = 3.15*Params.T_CNB
idx, val = find_nearest(p_l_z_target_range, p_l_select)
# print(idx, val/Params.T_CNB)

""" 
fig = plt.figure(figsize=(4,4))
fig.tight_layout()
ax = fig.add_subplot(111)

colors = plt.cm.jet(np.linspace(0, 1, p_num))
for c, p_h_set in enumerate(p_h_sol):

    ax.plot(p_h_set[..., 0], p_h_set[..., 1]/Params.T_CNB, color=colors[c])


ax.axhline(0.0, color="red", ls='dashed')
ax.axvline(90.0, color="orange", ls="dotted")
ax.set_xlabel(f"decay angles rest frame [deg]")
ax.set_ylabel(f"parent momentum [T_CNB]")
ax.set_xlim(a_min, a_max)  # normal
# ax.set_xlim(85, 95) ; ax.set_ylim(-0.5, 0.5)  # zoom-in
plt.show(); plt.close()
"""