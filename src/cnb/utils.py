"""Numerical helpers: interpolation, smoothing, random directions, sorting
of simulated velocities, axis formatters."""


import numpy as np
from scipy.signal import savgol_filter
import jax
import jax.numpy as jnp
import chex

from cnb.physics import Physics


@chex.dataclass
class Utils:

    @staticmethod
    def savgol_smooth(data, window_length=51, polyorder=3):
        """
        Apply Savitzky-Golay filter for smoothing.
        
        Parameters:
        -----------
        data : array-like
            The data to smooth
        window_length : int
            Length of the smoothing window. Must be odd and greater than polyorder.
        polyorder : int
            Order of the polynomial used for fitting. Must be less than window_length.
            
        Returns:
        --------
        smoothed_data : array-like
            The smoothed data array
        """
        # Convert from JAX array to NumPy if needed
        try:
            data_np = np.array(data)
        except:
            data_np = data
            
        # Ensure window_length is odd
        if window_length % 2 == 0:
            window_length += 1
            
        # Apply Savitzky-Golay filter
        smoothed_data = savgol_filter(data_np, window_length, polyorder)
        
        # Convert back to JAX array if needed
        try:
            return jnp.array(smoothed_data)
        except:
            return smoothed_data


    @staticmethod
    def make_logz_arr(zA, zB, num_points):
        return jnp.logspace(jnp.log10(zA+1), jnp.log10(zB+1), num_points) - 1


    @staticmethod
    @jax.jit
    def jax_interpolate(x_interp, x_points, y_points):

        # Find indices where x_interp is between x_points
        idx = jnp.argmax(x_points > x_interp) - 1

        # Handle edge case where x_interp is less than all x_points
        idx = jnp.where(idx < 0, 0, idx)  

        # Handle edge case for upper bound
        idx = jnp.where(idx >= len(x_points) - 1, len(x_points) - 2, idx)  

        # Compute the slope (dy/dx) for the interval
        dy_dx = (y_points[idx + 1] - y_points[idx]) / (x_points[idx + 1] - x_points[idx])

        # Calculate the interpolated value
        y_interp = y_points[idx] + dy_dx * (x_interp - x_points[idx])

        return y_interp
    

    @staticmethod
    @jax.jit
    def jax_interpolate_1D(x_interp, x_points, y_points):
        # Ensure x_interp is a 1D array
        x_interp = jnp.atleast_1d(x_interp)

        # Find indices where x_interp is between x_points
        idx = jnp.searchsorted(x_points, x_interp, side='right') - 1

        # Handle edge cases
        idx = jnp.clip(idx, 0, len(x_points) - 2)

        # Compute the slopes (dy/dx) for the intervals
        slopes = (y_points[1:] - y_points[:-1]) / (x_points[1:] - x_points[:-1])

        # Calculate the interpolated values
        y_interp = y_points[idx] + slopes[idx] * (x_interp - x_points[idx])

        return y_interp


    @staticmethod
    @jax.jit
    def vectorized_interpolate_1D(x_interp, x_points, y_points):
        # Ensure inputs are JAX arrays
        x_interp = jnp.asarray(x_interp)
        x_points = jnp.asarray(x_points)
        y_points = jnp.asarray(y_points)

        # Get the shape of the input arrays
        shape = x_points.shape[:-1]
        n_points = x_points.shape[-1]

        # Reshape arrays to 2D: (num_arrays, n_points)
        x_points_2d = x_points.reshape(-1, n_points)
        y_points_2d = y_points.reshape(-1, n_points)
        x_interp_2d = x_interp.reshape(-1, x_interp.shape[-1])

        # Find indices where x_interp is between x_points
        idx = jax.vmap(lambda a, v: jnp.clip(jnp.searchsorted(a, v, side='right') - 1, 0, n_points - 2))(x_points_2d, x_interp_2d)

        # Compute the slopes (dy/dx) for the intervals
        slopes = (y_points_2d[:, 1:] - y_points_2d[:, :-1]) / (x_points_2d[:, 1:] - x_points_2d[:, :-1])

        # Gather the required values
        idx_expanded = idx[:, None, :]
        y_left = jnp.take_along_axis(y_points_2d[:, None, :], idx_expanded, axis=-1)
        x_left = jnp.take_along_axis(x_points_2d[:, None, :], idx_expanded, axis=-1)
        slopes_interp = jnp.take_along_axis(slopes[:, None, :], idx_expanded, axis=-1)

        # Calculate the interpolated values
        y_interp = y_left + slopes_interp * (x_interp_2d[:, None, :] - x_left)

        # Reshape the result back to the original shape
        return y_interp.reshape(shape + x_interp.shape[-1:])


    @staticmethod
    def fill_1d_arrays_with_zeros_jax(array_list):
        # Find the maximum N among all arrays
        max_N = max(arr.shape[0] for arr in array_list)
        
        # Use list comprehension to pad arrays for the (N,) case
        filled_arrays = [jnp.pad(arr, (0, max_N - arr.shape[0]), 
                                mode='constant', constant_values=0)
                        for arr in array_list]
        
        # Stack the padded arrays into a single JAX array
        return jnp.stack(filled_arrays)


    @staticmethod
    def fill_2d_arrays_with_zeros_jax(array_list):

        # Find the maximum N among all arrays
        max_N = max(arr.shape[0] for arr in array_list)
        
        # Use list comprehension to pad arrays
        filled_arrays = [jnp.pad(arr, ((0, max_N - arr.shape[0]), (0, 0)), 
                        mode='constant', constant_values=0)
                        for arr in array_list]
        
        return jnp.array(filled_arrays)


    @staticmethod
    def y_fmt(value, tick_number):
        if value == 1e-6:
            return r'1+$10^{-6}$'
        elif value == 1e-5:
            return r'1+$10^{-5}$'
        elif value == 1e-4:
            return r'1+$10^{-4}$'
        elif value == 1e-3:
            return r'1+$10^{-3}$'
        elif value == 1e-2:
            return r'1+$10^{-2}$'
        elif value == 1e-1:
            return r'1+$10^{-1}$'
        elif value == 1e0:
            return r'1+$10^0$'
        elif value == 1e1:
            return r'1+$10^1$'
        

    @staticmethod
    def int_labels(x, pos):
        if x in [1, 10]:  # Only label these x values
            return f"{int(x)}"
        else:
            return ""
        

    @staticmethod
    @jax.jit
    def v_mag_to_xyz(v_mag, key):
        """
        Generate isotropic velocity components from magnitudes.
        """

        N = v_mag.shape[0]
        
        # Split the key for generating two sets of random numbers
        key1, key2 = jax.random.split(key)
        
        # Generate uniform random numbers
        phis = jax.random.uniform(key1, (N,), minval=0, maxval=2*jnp.pi)
        cos_thetas = 2.0 * jax.random.uniform(key2, (N,)) - 1.0
        
        # Calculate sin_thetas
        sin_thetas = jnp.sqrt(1.0 - cos_thetas**2)
        
        # Calculate x, y, z components
        x = v_mag * jnp.cos(phis) * sin_thetas
        y = v_mag * jnp.sin(phis) * sin_thetas
        z = v_mag * cos_thetas
        
        return jnp.column_stack([x, y, z])


    @staticmethod
    @jax.jit
    def generate_isotropic_vectors(v_mag, key):
        """
        Generate N isotropic 3D velocity vectors with the same magnitude.
        
        Parameters:
        -----------
        v_mag : float
            The magnitude for all generated vectors.
        key : jax.random.PRNGKey
            JAX random key for reproducibility.
        
        Returns:
        --------
        jnp.ndarray
            Array of shape (N, 3) containing randomly oriented vectors all with magnitude v_mag.
        """
        
        # Split the key for generating two sets of random numbers
        key1, key2 = jax.random.split(key)
        
        N = 2_000_001
        # N = 1_001

        # Generate uniform random numbers
        phis = jax.random.uniform(key1, (N,), minval=0, maxval=2*jnp.pi)
        cos_thetas = 2.0 * jax.random.uniform(key2, (N,)) - 1.0
        
        # Calculate sin_thetas
        sin_thetas = jnp.sqrt(1.0 - cos_thetas**2)
        
        # Calculate x, y, z components
        x = v_mag * jnp.cos(phis) * sin_thetas
        y = v_mag * jnp.sin(phis) * sin_thetas
        z = v_mag * cos_thetas
        
        return jnp.column_stack([x, y, z])


    @jax.jit
    def v_mag_to_xyz_biased_1D(v_mag, key, v_direction, cone_angle_rad=jnp.pi/6):
        """
        Generate velocity components biased toward a specific direction.
        
        Parameters:
        - v_mag: Array of velocity magnitudes
        - key: JAX random key
        - v_direction: Direction vector to bias toward (any magnitude)
        - cone_angle_rad: Half-angle of the cone in radians (default: 30 degrees)
        
        Returns:
        - Array of velocity vectors biased in the specified direction
        """
        N = v_mag.shape[0]
        
        # Normalize the direction vector
        v_direction_norm = v_direction / jnp.linalg.norm(v_direction)
        
        # Create a rotation matrix to align [0, 0, 1] with our direction vector
        # First, find the axis of rotation by cross product
        v_z = jnp.array([0.0, 0.0, 1.0])
        
        # Handle the special case where v_direction is parallel to z-axis
        parallel_z = jnp.allclose(jnp.abs(jnp.dot(v_direction_norm, v_z)), 1.0)
        
        # If direction is -z, use [1,0,0] as rotation axis
        rotation_axis = jnp.where(
            parallel_z,
            jnp.array([1.0, 0.0, 0.0]),
            jnp.cross(v_z, v_direction_norm)
        )
        
        # Normalize the rotation axis (if not parallel case)
        rotation_axis = jnp.where(
            parallel_z,
            rotation_axis,
            rotation_axis / jnp.linalg.norm(rotation_axis)
        )
        
        # Find the angle between [0,0,1] and v_direction
        cos_angle = jnp.dot(v_z, v_direction_norm)
        angle = jnp.arccos(jnp.clip(cos_angle, -1.0, 1.0))
        
        # Create the rotation matrix using Rodrigues' rotation formula
        K = jnp.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])
        
        # Rodrigues' rotation formula
        R = jnp.eye(3) + jnp.sin(angle) * K + (1 - jnp.cos(angle)) * jnp.matmul(K, K)
        
        # Special case handling for parallel to z-axis
        is_negative_z = parallel_z & (cos_angle < 0)
        R = jnp.where(
            is_negative_z,
            jnp.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0]
            ]),
            R
        )
        
        # Split the key for generating random angles
        key1, key2 = jax.random.split(key)
        
        # Generate random phi angles (around the cone axis)
        phis = jax.random.uniform(key1, (N,), minval=0, maxval=2*jnp.pi)
        
        # Generate random theta angles (cone spread)
        # Using a distribution that concentrates points within the cone_angle
        thetas = jax.random.uniform(key2, (N,), minval=0, maxval=1.0) * cone_angle_rad
        
        # Calculate points on a cone around the z-axis, then rotate to the target direction
        sin_thetas = jnp.sin(thetas)
        cos_thetas = jnp.cos(thetas)
        
        # Generate points on a unit sphere, biased toward z-axis within cone_angle
        x_local = jnp.cos(phis) * sin_thetas
        y_local = jnp.sin(phis) * sin_thetas
        z_local = cos_thetas
        
        # Stack these coordinates
        local_coords = jnp.column_stack([x_local, y_local, z_local])
        
        # Rotate these coordinates to be centered around our direction vector
        # Apply the rotation matrix to each point
        rotated_coords = jnp.einsum('ij,kj->ki', R, local_coords)
        
        # Scale by velocity magnitude
        velocities = rotated_coords * v_mag[:, jnp.newaxis]
        
        return velocities


    @jax.jit
    def v_mag_to_xyz_biased(
        v_mag, key, v_direction, cone_angle_rad=jnp.pi/6):
        """
        Generate velocity components biased toward a specific direction.
        
        Parameters:
        - v_mag: Either a single velocity magnitude or array of velocity magnitudes
        - key: JAX random key
        - v_direction: Direction vector to bias toward (any magnitude)
        - cone_angle_rad: Half-angle of the cone in radians (default: 30 degrees)
        
        Returns:
        - Array of velocity vectors biased in the specified direction
        """
        # Check if v_mag is a scalar or array
        is_scalar = jnp.isscalar(v_mag) or (hasattr(v_mag, 'shape') and len(v_mag.shape) == 0)
        
        # If v_mag is a scalar and N is provided, create an array of N identical values
        N = 100_001
        v_mag_array = jnp.ones(N) * v_mag if is_scalar else v_mag
        
        # Get the number of velocities to generate
        N_actual = N if is_scalar else v_mag_array.shape[0]
        
        # Normalize the direction vector
        v_direction_norm = v_direction / jnp.linalg.norm(v_direction)
        
        # Create a rotation matrix to align [0, 0, 1] with our direction vector
        # First, find the axis of rotation by cross product
        v_z = jnp.array([0.0, 0.0, 1.0])
        
        # Handle the special case where v_direction is parallel to z-axis
        parallel_z = jnp.allclose(jnp.abs(jnp.dot(v_direction_norm, v_z)), 1.0)
        
        # If direction is -z, use [1,0,0] as rotation axis
        rotation_axis = jnp.where(
            parallel_z,
            jnp.array([1.0, 0.0, 0.0]),
            jnp.cross(v_z, v_direction_norm)
        )
        
        # Normalize the rotation axis (if not parallel case)
        rotation_axis = jnp.where(
            parallel_z,
            rotation_axis,
            rotation_axis / jnp.linalg.norm(rotation_axis)
        )
        
        # Find the angle between [0,0,1] and v_direction
        cos_angle = jnp.dot(v_z, v_direction_norm)
        angle = jnp.arccos(jnp.clip(cos_angle, -1.0, 1.0))
        
        # Create the rotation matrix using Rodrigues' rotation formula
        K = jnp.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])
        
        # Rodrigues' rotation formula
        R = jnp.eye(3) + jnp.sin(angle) * K + (1 - jnp.cos(angle)) * jnp.matmul(K, K)
        
        # Special case handling for parallel to z-axis
        is_negative_z = parallel_z & (cos_angle < 0)
        R = jnp.where(
            is_negative_z,
            jnp.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0]
            ]),
            R
        )
        
        # Split the key for generating random angles
        key1, key2 = jax.random.split(key)
        
        # Generate random phi angles (around the cone axis)
        phis = jax.random.uniform(key1, (N_actual,), minval=0, maxval=2*jnp.pi)
        
        # Generate random theta angles (cone spread)
        # Using a distribution that concentrates points within the cone_angle
        thetas = jax.random.uniform(key2, (N_actual,), minval=0, maxval=1.0) * cone_angle_rad
        
        # Calculate points on a cone around the z-axis, then rotate to the target direction
        sin_thetas = jnp.sin(thetas)
        cos_thetas = jnp.cos(thetas)
        
        # Generate points on a unit sphere, biased toward z-axis within cone_angle
        x_local = jnp.cos(phis) * sin_thetas
        y_local = jnp.sin(phis) * sin_thetas
        z_local = cos_thetas
        
        # Stack these coordinates
        local_coords = jnp.column_stack([x_local, y_local, z_local])
        
        # Rotate these coordinates to be centered around our direction vector
        # Apply the rotation matrix to each point
        rotated_coords = jnp.einsum('ij,kj->ki', R, local_coords)
        
        # Scale by velocity magnitude
        if is_scalar:
            # If v_mag was a scalar, we can broadcast it directly
            velocities = rotated_coords * v_mag
        else:
            # Otherwise use the array version with broadcasting
            velocities = rotated_coords * v_mag_array[:, jnp.newaxis]
        
        return velocities


    @staticmethod
    def sim_vels_to_sorted_z0z4(sim_vels, m_nu_arr, merge_last_axes, args):

        # Convert velocities to momenta
        p_arr, y_arr = Physics.velocities_to_momenta_ND_halo_0th_axis(
        v_arr=sim_vels, 
        m_arr=m_nu_arr,
        args=args)

        # Split into arrays for z0 and z4
        p_z0 = p_arr[...,0]
        p_z4 = p_arr[...,-1]
        y_z0 = y_arr[...,0]
        y_z4 = y_arr[...,-1]

        if merge_last_axes:
            p_z0 = p_z0.reshape(*p_z0.shape[:-2], -1)
            p_z4 = p_z4.reshape(*p_z4.shape[:-2], -1)
            y_z0 = y_z0.reshape(*y_z0.shape[:-2], -1)
            y_z4 = y_z4.reshape(*y_z4.shape[:-2], -1)

        # Sort in ascending order of momentum array today
        ind = p_z0.argsort(axis=-1)
        p_z0_sort = jnp.take_along_axis(p_z0, ind, axis=-1)
        p_z4_sort = jnp.take_along_axis(p_z4, ind, axis=-1)
        y_z0_sort = jnp.take_along_axis(y_z0, ind, axis=-1)
        y_z4_sort = jnp.take_along_axis(y_z4, ind, axis=-1)
        
        return ind, p_z0_sort, p_z4_sort, y_z0_sort, y_z4_sort
    

    @staticmethod
    def sim_vels_to_sorted_z0z4_vec(sim_vels, m_nu_arr, merge_last_axes, args):
        # Convert velocities to momenta
        p_vec, p_mag, y = Physics.velocities_to_momentum_vec_ND_halo_0th_axis(
            v_arr=sim_vels,
            m_arr=m_nu_arr,
            args=args
        )

        # Split into arrays for z0 and z4
        p_vec_z0 = p_vec[..., 0, :]
        p_vec_z4 = p_vec[..., -1, :]
        p_mag_z0 = p_mag[..., 0]
        p_mag_z4 = p_mag[..., -1]
        y_z0 = y[..., 0]
        y_z4 = y[..., -1]

        if merge_last_axes:
            p_vec_z0 = p_vec_z0.reshape(*p_vec_z0.shape[:-3], -1, 3)
            p_vec_z4 = p_vec_z4.reshape(*p_vec_z4.shape[:-3], -1, 3)
            p_mag_z0 = p_mag_z0.reshape(*p_mag_z0.shape[:-2], -1)
            p_mag_z4 = p_mag_z4.reshape(*p_mag_z4.shape[:-2], -1)
            y_z0 = y_z0.reshape(*y_z0.shape[:-2], -1)
            y_z4 = y_z4.reshape(*y_z4.shape[:-2], -1)

        # Sort based on magnitude of momentum today
        ind = p_mag_z0.argsort(axis=-1)
        
        # Apply sorting to all arrays
        p_vec_z0_sort = jnp.take_along_axis(p_vec_z0, ind[..., None], axis=-2)
        p_vec_z4_sort = jnp.take_along_axis(p_vec_z4, ind[..., None], axis=-2)
        p_mag_z0_sort = jnp.take_along_axis(p_mag_z0, ind, axis=-1)
        p_mag_z4_sort = jnp.take_along_axis(p_mag_z4, ind, axis=-1)
        y_z0_sort = jnp.take_along_axis(y_z0, ind, axis=-1)
        y_z4_sort = jnp.take_along_axis(y_z4, ind, axis=-1)
        
        return ind, p_vec_z0_sort, p_vec_z4_sort, p_mag_z0_sort, p_mag_z4_sort, y_z0_sort, y_z4_sort
