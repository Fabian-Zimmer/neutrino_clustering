"""Rotations and healpix skymap helpers."""


import healpy as hp
import jax.numpy as jnp


class Skymaps():
    """Rotations of particles and healpix skymaps."""


    @staticmethod
    def rotate_particles(particles_pos, obs_pos):

        # Calculate the Euler angles, to match frame of observer. This will be
        # the new frame in which DM particles get projected to an all sky map.
        zAngle = jnp.arctan2(obs_pos[1], -obs_pos[0])
        yAngle = jnp.arctan2(-obs_pos[2], jnp.linalg.norm(obs_pos[:2]))
        # print('*** Angles: ***')
        # print(np.rad2deg(zAngle), np.rad2deg(yAngle))

        cz = jnp.cos(zAngle)
        sz = jnp.sin(zAngle)
        cy = jnp.cos(yAngle)
        sy = jnp.sin(yAngle)

        # Rotation matrix around z-axis
        R_z = jnp.array([
            [cz, -sz, 0],
            [sz, cz,  0],
            [0,  0,   1]
        ])

        # Rotation matrix around y-axis
        R_y = jnp.array([
            [cy,  0, sy],
            [0,   1,  0],
            [-sy, 0, cy]
        ])

        # Combined rotation matrix
        rot_mat = jnp.matmul(R_y, R_z)

        # Observer position in new frame (aligned with x-axis).
        obs_orig_in_rot_frame = jnp.matmul(rot_mat, obs_pos.T).T

        # Dark matter positions in observer frame.
        DM_orig_in_rot_frame = jnp.matmul(rot_mat, particles_pos.T).T

        return DM_orig_in_rot_frame, obs_orig_in_rot_frame


    @staticmethod
    def get_rotation_matrix_euler(cell_pos):
        """
        Calculate rotation matrix using Euler angles to transform between reference frames.
        """
        # Calculate Euler angles for rotation
        # First angle: rotation around z-axis
        zAngle = jnp.arctan2(cell_pos[1], -cell_pos[0])
        
        # Second angle: rotation around y-axis
        yAngle = jnp.arctan2(-cell_pos[2], jnp.linalg.norm(cell_pos[:2]))
        
        # Compute trigonometric values
        cz = jnp.cos(zAngle)
        sz = jnp.sin(zAngle)
        cy = jnp.cos(yAngle)
        sy = jnp.sin(yAngle)
        
        # Rotation matrix around z-axis
        R_z = jnp.array([
            [cz, -sz, 0],
            [sz, cz,  0],
            [0,  0,   1]
        ])
        
        # Rotation matrix around y-axis
        R_y = jnp.array([
            [cy,  0, sy],
            [0,   1,  0],
            [-sy, 0, cy]
        ])
        
        # Combined rotation matrix (y-rotation after z-rotation)
        rot_mat = jnp.matmul(R_y, R_z)
        
        return rot_mat


    @staticmethod
    def get_direct_rotation(earth_position, cell_position):
        """
        Directly construct a rotation matrix that rotates earth_position to cell_position.
        """
        # Normalize both vectors
        earth_norm = earth_position / jnp.linalg.norm(earth_position, axis=-1)
        cell_norm = cell_position / jnp.linalg.norm(cell_position, axis=-1)
        
        # Calculate the rotation axis
        axis = jnp.cross(earth_norm, cell_norm)
        axis_norm = jnp.linalg.norm(axis, axis=-1)
        
        # Handle parallel or antiparallel vectors
        if axis_norm < 1e-10:
            # If vectors are parallel, no rotation needed
            if jnp.dot(earth_norm, cell_norm) > 0:
                return jnp.eye(3)
            # If vectors are antiparallel, rotate 180° around z-axis
            else:
                return jnp.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        
        # Normalize the axis
        axis = axis / axis_norm
        
        # Calculate the rotation angle
        cos_angle = jnp.dot(earth_norm, cell_norm)
        sin_angle = axis_norm
        
        # Build rotation matrix using Rodrigues' formula
        K = jnp.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        R = jnp.eye(3) + sin_angle * K + (1 - cos_angle) * jnp.matmul(K, K)
        return R


    @staticmethod
    def rotate_skymap(healpix_map, cell0, lonlat=True):

        # Input map related quantities
        Nside = hp.pixelfunc.get_nside(healpix_map)
        pix_idx = jnp.arange(len(healpix_map))

        # Angles to compensate for location of starting cell (cell0)
        zAngle = jnp.rad2deg(jnp.arctan2(cell0[1], -cell0[0]))
        yAngle = jnp.rad2deg(jnp.arctan2(-cell0[2], jnp.linalg.norm(cell0[:2])))

        # Create rotator object, to place c.o.p. of halo in the map center
        rot = hp.rotator.Rotator(
            rot=[-zAngle, yAngle, 0], deg=True, eulertype='ZYX', inv=False)

        # Create a grid of (theta, phi) coordinates for the map
        phi, theta = hp.pix2ang(
            Nside, pix_idx, lonlat=lonlat)
        
        # Apply rotation to the grid
        phi_rot, theta_rot = rot(phi, theta, lonlat=lonlat)
        
        # Find the pixel indices of the rotated grid
        pix_rot = hp.ang2pix(Nside, phi_rot, theta_rot, lonlat=lonlat)

        # Create a new map with rotated pixel values
        zeros_map = jnp.zeros_like(healpix_map)
        # rotated_map[pix_rot] = healpix_map
        rotated_map = zeros_map.at[pix_rot].set(healpix_map)
        #? pix_rot contains duplicates, and duplicated pixel will get the last healpix_map value...change to mean of all the healpix_map values, which fall onto the new pixel in rotated_map?
        
        return rotated_map


    def interpolate_skymap(healpix_map):

        Nside = hp.pixelfunc.get_nside(healpix_map)
        zero_pixels = jnp.where(healpix_map == 0)[0]
        
        for pix in zero_pixels:

            # Get neighbouring pixels or pixel without value
            neighbours_pix = hp.get_all_neighbours(Nside, pix)

            # Use median of neighbours to fill in missing value
            interp_val = jnp.median(healpix_map.at[neighbours_pix].get())
            
            # Update the pixel value using .at[].set() for immutability
            healpix_map = healpix_map.at[pix].set(interp_val)
        
        return healpix_map


    @staticmethod
    def smooth_skymap(healpix_map):

        Nside = hp.pixelfunc.get_nside(healpix_map)

        # Smooth using a Gaussian kernel.
        reso_1pix_deg = hp.nside2resol(Nside, arcmin=True) / 60
        sigma_rad = jnp.deg2rad(reso_1pix_deg)/4
        smooth_rot_map = hp.smoothing(healpix_map, sigma=sigma_rad)

        return smooth_rot_map


    @staticmethod
    def DM_pos_to_skymap(Nside, DM_pos_in, obs_pos_in):

        # Center on observer position.
        DM_pos = DM_pos_in - obs_pos_in
        xDM, yDM, zDM = DM_pos[:,0], DM_pos[:,1], DM_pos[:,2]

        # Convert x,y,z to angles.
        proj_xy_plane_dis = jnp.sqrt(xDM**2 + yDM**2)

        thetas = jnp.arctan2(zDM, proj_xy_plane_dis)
        phis = jnp.arctan2(yDM, xDM)

        # To galactic latitude and longitude (in degrees) for healpy.
        hp_glon, hp_glat = jnp.rad2deg(phis), jnp.rad2deg(thetas)

        # Convert angles to pixel indices using ang2pix.
        pixel_indices = hp.ang2pix(
            Nside, hp_glon, hp_glat, lonlat=True
        )

        # Create a Healpix map and increment the corresponding pixels
        DM_zeros = jnp.zeros(hp.nside2npix(Nside))
        # jnp.add.at(DM_healpix_map, pixel_indices, 1)
        DM_healpix_map = DM_zeros.at[pixel_indices].add(1)

        return DM_healpix_map
