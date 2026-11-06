"""Positions and velocities of the Sun and Earth (annual modulation,
Zimmer & Ando 2025, arXiv:2507.01505) and Galactic frame conversions."""

from datetime import datetime, timedelta

import numpy as np
import astropy.units as apu
from astropy.coordinates import SkyCoord, GCRS, Galactic
from astropy.time import Time
from astropy.coordinates import get_body_barycentric, get_body_barycentric_posvel
import jax
import jax.numpy as jnp

from cnb.units import Params


class Astro():
    """Sun/Earth positions and velocities, frame conversions."""


    @staticmethod
    def Galactic_velocity_to_Cartesian_vector(velocity, l, b):
        """
        Convert velocity from galactic coordinates (l, b) to a Cartesian vector.

        Args:
        velocity (float): Magnitude of the velocity in km/s.
        l (float): Galactic longitude in degrees.
        b (float): Galactic latitude in degrees.

        Returns:
        jnp.ndarray: 3D velocity vector in Cartesian coordinates.
        """
        # Create a galactic coordinate object with a unit distance
        galactic_coord = Galactic(
            l=l*apu.degree, b=b*apu.degree, distance=1*apu.kpc)

        # Convert to Cartesian and normalize the direction vector
        x, y, z = galactic_coord.cartesian.xyz
        unit_vector = jnp.array([x.value, y.value, z.value]) / jnp.linalg.norm(jnp.array([x.value, y.value, z.value]))

        # Scale the unit vector by the velocity magnitude
        velocity_vector = unit_vector * velocity

        return velocity_vector


    @staticmethod
    def S_to_Sprime_frame_trafo(u, v, args=None):
        """
        Transform velocity from CNB frame (S) to Earth frame (S') using relativistic velocity addition formula.
        See https://en.wikipedia.org/wiki/Velocity-addition_formula, under "General configuration", 2nd formula (v positive, S -> S').
        
        Parameters:
        u: velocity of object (neutrinos) in frame S
        v: velocity of frame S' as measured in frame S
        
        Returns:
        u_prime: velocity of object in frame S'
        """

        if args is None:
            args = Params()

        u_dot_v = jnp.sum(u*v, axis=-1, keepdims=True)
        v_sq = jnp.sum(v**2, axis=-1, keepdims=True)
        
        # Lorentz factor
        gamma = 1/jnp.sqrt(1 - v_sq**2)
        
        numer = u/gamma - v + (gamma/(1+gamma))*u_dot_v*v
        denom = 1 - u_dot_v
        u_prime = numer/denom
        
        return u_prime


    @staticmethod
    def calculate_fractional_day_numbers(year):
        """
        Calculate the fractional day number n for each day of the given year.
        
        :param year: The year for which to calculate the fractional day numbers
        :return: A list of tuples, each containing (date, fractional day number)
        """
        results = []
        start_date = datetime(year, 1, 1)
        
        for day in range(
            366 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 365):
            current_date = start_date + timedelta(days=day)
            Y, M, D = current_date.year, current_date.month, current_date.day
            
            Y_tilde = Y - 1 if M <= 2 else Y
            M_tilde = M + 12 if M <= 2 else M
            
            n = np.floor(365.25 * Y_tilde) + np.floor(30.61 * (M_tilde + 1)) + D - 730563.5
            
            results.append((current_date.strftime('%Y-%m-%d'), n))
        
        return results


    @staticmethod
    def calculate_average_ecliptic_vectors(year):
        """
        Calculate the average ecliptic unit vectors ϵ_x and ϵ_y for the given year.
        
        :param year: The year for which to calculate the average ecliptic vectors
        :return: A tuple containing (year, average ϵ_x, average ϵ_y)
        """
        # Constants for ϵ_x and ϵ_y calculations
        eps_x_0 = np.array([0.054876, -0.494109, 0.867666])
        eps_x_T = np.array([-0.024232, -0.002689, 1.546e-6])
        eps_y_0 = np.array([0.993821, 0.110992, 0.000352])
        eps_y_T = np.array([0.001316, -0.011851, 0.021267])
        
        eps_x_sum = np.zeros(3)
        eps_y_sum = np.zeros(3)
        
        # Get fractional day numbers for the year
        fractional_days = Astro.calculate_fractional_day_numbers(year)
        
        for _, n in fractional_days:
            # Calculate T
            T = n / 36525
            
            # Calculate ϵ_x and ϵ_y for this day
            eps_x = eps_x_0 + eps_x_T * T
            eps_y = eps_y_0 + eps_y_T * T
            
            # Add to the sum
            eps_x_sum += eps_x
            eps_y_sum += eps_y
        
        # Calculate averages
        days_in_year = len(fractional_days)
        eps_x_avg = eps_x_sum / days_in_year
        eps_y_avg = eps_y_sum / days_in_year
        
        # Normalize the average vectors
        eps_x_avg /= np.linalg.norm(eps_x_avg)
        eps_y_avg /= np.linalg.norm(eps_y_avg)
        
        return eps_x_avg, eps_y_avg


    @staticmethod
    def calculate_earth_position(year, mode="ecliptic"):
        """
        Calculate Earth's position relative to the Sun in ecliptic or galactic coordinates for each day of the year.
        
        :param year: The year for which to calculate Earth's position
        :return: A list of tuples, each containing (date, position vector)
        """
        # Constants
        e = 0.9574  # eccentricity in degrees
        e_rad = np.radians(e)

        # Get fractional day numbers and ecliptic vectors for the year
        fractional_days = Astro.calculate_fractional_day_numbers(year)
        eps_x, eps_y = Astro.calculate_average_ecliptic_vectors(year)

        # For mode == "galactic"
        b_gal_0 = np.array([5.536, -59.574, -29.811])
        b_gal_T = np.array([0.013, 0.002, 0.001])
        lambda_gal_0 = np.array([266.840, 347.340, 180.023])
        lambda_gal_T = np.array([1.397, 1.375, 1.404])

        results = []

        for date, n in fractional_days:

            # Calculate L and g
            L = np.radians(280.460 + 0.9856474 * n)
            g = np.radians(357.528 + 0.9856003 * n)

            # Calculate ecliptic longitude
            l = L + 2 * e_rad * np.sin(g) + 5/4 * e_rad**2 * np.sin(2*g)

            # Calculate Earth-Sun distance (in AU)
            r = 1.00014 - 0.01671 * np.cos(g) - 0.00014 * np.cos(2*g)

            # Calculate position vector
            
            if mode == "ecliptic":
                position = r * (np.cos(l) * eps_x + np.sin(l) * eps_y)

            if mode == "galactic":

                # Calculate T
                T = n / 36525
                
                b_gal = b_gal_0 + b_gal_T * T
                l_gal = (lambda_gal_0 + lambda_gal_T * T) - 180

                position = r * np.cos(b_gal) * np.cos(l - l_gal)
            
            results.append((date, position))

        return results


    @staticmethod
    def calculate_earth_velocity(year, mode="ecliptic"):
        """
        Calculate Earth's velocity for each day of the year.
        
        :param year: The year for which to calculate Earth's velocity
        :return: A list of tuples, each containing (date, velocity vector)
        """
        # Constants
        e = 0.9574  # eccentricity in degrees
        e_rad = np.radians(e)
        u_E_avg = 29.79  # Average Earth velocity in km/s

        # Get fractional day numbers and ecliptic vectors for the year
        fractional_days = Astro.calculate_fractional_day_numbers(year)
        eps_x, eps_y = Astro.calculate_average_ecliptic_vectors(year)

        # For mode == "galactic"
        b_gal_0 = np.array([5.536, -59.574, -29.811])
        b_gal_T = np.array([0.013, 0.002, 0.001])

        results = []

        for date, n in fractional_days:
            # Calculate L and varpi
            L = np.radians(280.460 + 0.9856474 * n)
            varpi = np.radians(282.932 + 0.0000471 * n)
            
            if mode == "ecliptic":
                # Calculate velocity components
                v_x = -u_E_avg * (np.sin(L) + e_rad * np.sin(2*L - varpi))
                v_y = u_E_avg * (np.cos(L) + e_rad * np.cos(2*L - varpi))

                # Calculate velocity vector
                velocity = v_x * eps_x + v_y * eps_y
                
            if mode == "galactic":
                
                # Calculate T
                T = n / 36525
                
                b_gal = b_gal_0 + b_gal_T * T

                velocity = u_E_avg * np.cos(b_gal)

            results.append((date, velocity))

        return results


    @staticmethod
    def SunEarthGC_frame_coords_posvel(
            year, rel_vel="MW", Earth_rel_Sun=False):
        """
        Calculate coordinates and velocities in a frame aligned with the Galactic Center,
        using get_body_barycentric_posvel for more accurate Earth velocities.
        """
        # Define the Galactic Center coordinates
        gc = SkyCoord(l=0*apu.degree, b=0*apu.degree, frame='galactic')
        
        if rel_vel == "CNB":
            # Sun's velocity w.r.t. CNB in galactic coordinates/frame (km/s)
            sun_vel_Gal = np.array([-0.06935679,-0.66222051,0.74609224])*369.82

        if rel_vel == "MW":
            # Sun's velocity w.r.t. MWGC in galactic coordinates/frame (km/s)
            sun_vel_Gal = np.array([0.04420196,0.99859217,0.0293235])*251.19

        # Lists for Sun's positions and velocities, and Earth's velocities
        sun_positions = []
        sun_vels = []
        earth_vels = []
        earth_positions = []
        
        # Loop backwards through each day of the year
        for day in range(365, 0, -1):  
            # Time object of current day at noon UTC
            time = Time(f'{year}-12-31 12:00:00') - (365-day)*apu.day
            
            # Get Earth's position and velocity (ICRS)
            earth_pos, earth_vel = get_body_barycentric_posvel('earth', time)
            
            # Extract Earth's position values, leave in AU
            earth_xyz = np.array([
                earth_pos.x.value,
                earth_pos.y.value,
                earth_pos.z.value
            ])

            # Convert Earth's velocity from AU/day to km/s
            earth_vel_icrs = np.array([
                earth_vel.x.value,
                earth_vel.y.value,
                earth_vel.z.value
            ]) * 1731.45683633  # conversion factor from AU/day to km/s
            
            # Get Sun's position in GCRS coordinates (relative to Earth) in km
            sun_gcrs = SkyCoord(
                get_body_barycentric('sun', time), 
                frame='icrs').transform_to(GCRS(obstime=time))
            sun_vector = sun_gcrs.cartesian.xyz.to(apu.km).value
            
            # Transform the Galactic Center to GCRS
            gc_gcrs = gc.transform_to(GCRS(obstime=time))
            
            # Calculate the unit vector pointing from Earth to Galactic Center
            gc_vector = gc_gcrs.cartesian.xyz.value
            gc_unit_vector = gc_vector / np.linalg.norm(gc_vector)
            
            # Define the new coordinate system
            x_axis = gc_unit_vector
            z_axis = np.cross(x_axis, [0, 0, 1])
            z_axis = z_axis / np.linalg.norm(z_axis)
            y_axis = np.cross(z_axis, x_axis)
            
            # Create rotation matrix
            rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
            
            # Rotate Sun's position and velocity into new coordinate system
            sun_rotated = np.dot(rotation_matrix.T, sun_vector)
            sun_vel_rotated = np.dot(rotation_matrix.T, sun_vel_Gal)
            
            # Rotate Earth's velocity into the new coordinate system (km/s)
            earth_vel_rotated = np.dot(rotation_matrix.T, earth_vel_icrs)
            
            # Calculate Earth's velocity relative to CMB in the rotated frame (km/s)
            if not Earth_rel_Sun:
                earth_vel_rotated = sun_vel_rotated + earth_vel_rotated
            
            # Append the rotated position and velocities to the lists
            sun_positions.append(sun_rotated)
            sun_vels.append(sun_vel_rotated)
            earth_vels.append(earth_vel_rotated)
            earth_positions.append(earth_xyz)
        
        #/ sun_positions: km units ; velocities: km/s units
        return np.array(sun_positions), np.array(sun_vels), np.array(earth_vels), np.array(earth_positions)


    @staticmethod
    def get_sun_pos_vel_earth_gc_frame(
            year, rel_vel="MW", Earth_rel_Sun=False):
        """
        Calculate the daily position and velocity of the Sun relative to Earth
        in a coordinate frame where:
        - Earth is at the origin (0,0,0)
        - The x-axis points AWAY from the Galactic Center (GC is in negative x direction)
        - The Galactic Center is 8 kpc away along the negative x-axis
        
        Parameters:
        -----------
        year : int
            The year for which to calculate positions
        rel_vel : str, optional
            Reference frame for Sun velocity, either "MW" (Milky Way Galactic Center) 
            or "CNB" (Cosmic Neutrino Background), default is "MW"
        Earth_rel_Sun : bool, optional
            If True, don't add Sun's velocity to Earth's velocity, default is False
            
        Returns:
        --------
        tuple: (times, positions, velocities)
            times: list of Time objects for each day (in reverse order, Dec 31 first)
            positions: array of shape (n_days, 3) with x,y,z positions in AU
                    x: away from Galactic Center
                    y: perpendicular to x in the Galactic plane
                    z: perpendicular to the Galactic plane
            velocities: array of shape (n_days, 3) with x,y,z velocities in km/s
        """
        # Determine if it's a leap year
        is_leap_year = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        n_days = 366 if is_leap_year else 365
        
        # Initialize arrays to store results
        times = []
        positions = np.zeros((n_days, 3))
        velocities = np.zeros((n_days, 3))
        
        # Define Sun's velocity vectors in the Galaxy
        if rel_vel == "CNB":
            # Sun's velocity w.r.t. CNB in galactic coordinates/frame (km/s)
            sun_vel_Gal = np.array([-0.06935679, -0.66222051, 0.74609224]) * 369.82
        elif rel_vel == "MW":
            # Sun's velocity w.r.t. MWGC in galactic coordinates/frame (km/s)
            sun_vel_Gal = np.array([0.04420196, 0.99859217, 0.0293235]) * 251.19
        else:
            raise ValueError("rel_vel must be either 'CNB' or 'MW'")
        
        # Define the Galactic Center direction in Galactic coordinates
        galactic_center = SkyCoord(l=0*apu.degree, b=0*apu.degree, frame='galactic')
        
        # Define the north Galactic pole
        galactic_north_pole = SkyCoord(l=0*apu.degree, b=90*apu.degree, frame='galactic')
        
        # Loop backwards through each day of the year
        for i, day in enumerate(range(n_days, 0, -1)):
            # Time object of current day at noon UTC
            t = Time(f'{year}-12-31 12:00:00') - (n_days-day)*apu.day
            times.append(t)
            
            # Get Sun and Earth barycentric positions and velocities
            sun_pos_vel = get_body_barycentric_posvel('sun', t)
            earth_pos_vel = get_body_barycentric_posvel('earth', t)
            
            # Calculate Sun's position and velocity relative to Earth
            sun_pos_rel = sun_pos_vel[0] - earth_pos_vel[0]  # Position
            sun_vel_rel = sun_pos_vel[1] - earth_pos_vel[1]  # Velocity
            
            # Convert to Earth-centered Cartesian coordinates
            sun_pos_earth = sun_pos_rel.xyz.to(apu.au).value
            sun_vel_earth = sun_vel_rel.xyz.to(apu.km/apu.s).value
            
            # Get Galactic Center direction in GCRS for this date
            gc_gcrs = galactic_center.transform_to(GCRS(obstime=t))
            gnp_gcrs = galactic_north_pole.transform_to(GCRS(obstime=t))
            
            # Define the new coordinate axes
            # X-axis points AWAY from the Galactic Center
            x_axis = -gc_gcrs.cartesian.xyz.value
            x_axis = x_axis / np.linalg.norm(x_axis)
            
            # Z-axis is perpendicular to the Galactic plane
            z_axis = gnp_gcrs.cartesian.xyz.value
            z_axis = z_axis / np.linalg.norm(z_axis)
            
            # Y-axis is perpendicular to both X and Z (using cross product)
            y_axis = np.cross(z_axis, x_axis)
            y_axis = y_axis / np.linalg.norm(y_axis)
            
            # Ensure orthogonality by recalculating z_axis
            z_axis = np.cross(x_axis, y_axis)
            
            # Create rotation matrix to transform from GCRS to our custom frame
            rotation_matrix = np.array([x_axis, y_axis, z_axis])
            
            # Transform position and velocity to our custom Earth-centered, 
            # Galactic-Center-oriented frame
            sun_pos_custom = np.dot(rotation_matrix, sun_pos_earth)
            
            # Rotate Sun's reference velocity (CNB or MW) into this frame
            # sun_vel_rotated = np.dot(rotation_matrix, sun_vel_Gal)
            sun_vel_rotated = sun_vel_Gal
            
            # Rotate Earth's velocity into the new coordinate system (km/s)
            # earth_vel_rotated = np.dot(rotation_matrix, sun_vel_earth)
            earth_vel_rotated = sun_vel_earth
            
            # Calculate Earth's velocity relative to CNB or MW in the rotated frame (km/s)
            if not Earth_rel_Sun:
                earth_vel_rotated = sun_vel_rotated + earth_vel_rotated
            
            velocities[i] = earth_vel_rotated
            positions[i] = sun_pos_custom
        
        return times, positions, velocities


    @staticmethod
    def get_sun_earth_galactic_data(year, rel_vel="MW", Earth_rel_Sun=False):
        """
        Calculate Sun and Earth positions and velocities in the Galactic coordinate system.
        
        Parameters:
        -----------
        year : int
            The year for which to calculate data
        rel_vel : str, optional
            Reference frame for velocities, either "MW" (Milky Way Galactic Center) 
            or "CNB" (Cosmic Neutrino Background), default is "MW"
        Earth_rel_Sun : bool, optional
            If False, add Sun's velocity to Earth's relative velocity, default is False
            
        Returns:
        --------
        tuple: (times, sun_positions, sun_velocities, earth_velocities)
            times: list of Time objects for each day (in reverse order, Dec 31 first)
            sun_positions: array of shape (n_days, 3) with x,y,z positions in AU
                        Positions are relative to Earth in Galactic coordinates
                        x: toward Galactic Center (l=0°)
                        y: toward l=90° (left when facing GC)
                        z: toward North Galactic Pole (b=90°)
            sun_velocities: array of shape (n_days, 3) with x,y,z velocities in km/s
                        Sun's velocity relative to MW/CNB (same each day)
            earth_velocities: array of shape (n_days, 3) with x,y,z velocities in km/s
                        By default: Earth's velocity relative to the Sun in Galactic coordinates
                        If not Earth_rel_Sun: Earth's velocity with Sun's MW/CNB motion added
        """
        # Determine if it's a leap year
        is_leap_year = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        n_days = 366 if is_leap_year else 365
        
        # Initialize arrays to store results
        times = []
        sun_positions = np.zeros((n_days, 3))
        earth_velocities = np.zeros((n_days, 3))
        
        # Define Sun's velocity vectors in the Galaxy
        if rel_vel == "CNB":
            # Sun's velocity w.r.t. CNB in galactic coordinates/frame (km/s)
            sun_vel_Gal = np.array([-0.06935679, -0.66222051, 0.74609224]) * 369.82
        elif rel_vel == "MW":
            # Sun's velocity w.r.t. MWGC in galactic coordinates/frame (km/s)
            sun_vel_Gal = np.array([0.04420196, 0.99859217, 0.0293235]) * 251.19
        else:
            raise ValueError("rel_vel must be either 'CNB' or 'MW'")
        
        # Create sun_velocities array with the same velocity for each day
        sun_velocities = np.tile(sun_vel_Gal, (n_days, 1))
        
        # Loop backwards through each day of the year
        for i, day in enumerate(range(n_days, 0, -1)):
            # Time object of current day at noon UTC
            t = Time(f'{year}-12-31 12:00:00') - (n_days-day)*apu.day
            times.append(t)
            
            # Get Sun and Earth barycentric positions and velocities
            sun_pos_vel = get_body_barycentric_posvel('sun', t)
            earth_pos_vel = get_body_barycentric_posvel('earth', t)
            
            # Calculate Sun's position and velocity relative to Earth in ICRS
            sun_pos_rel = sun_pos_vel[0] - earth_pos_vel[0]
            sun_vel_rel = sun_pos_vel[1] - earth_pos_vel[1]
            
            # Create SkyCoord object for Sun's position in ICRS
            sun_icrs = SkyCoord(
                x=sun_pos_rel.x,
                y=sun_pos_rel.y,
                z=sun_pos_rel.z,
                frame='icrs',
                representation_type='cartesian'
            )
            
            # Transform to Galactic coordinates
            sun_gal = sun_icrs.transform_to('galactic')
            
            # Get Cartesian representation in Galactic frame
            sun_pos_gal = sun_gal.cartesian.xyz.to(apu.au).value
            
            # Store Sun's position relative to Earth in Galactic coordinates
            sun_positions[i] = sun_pos_gal
            
            # Calculate Earth's velocity relative to the Sun
            # Earth_vel = -Sun_vel_rel (in the Earth-Sun system)
            earth_sun_vel_rel = -(sun_vel_rel)
            
            # We need to create a SkyCoord with both position and velocity
            # First, temporarily use the Earth's position (at origin in ICRS)
            earth_icrs = SkyCoord(
                x=0*apu.au, y=0*apu.au, z=0*apu.au,
                v_x=earth_sun_vel_rel.x,
                v_y=earth_sun_vel_rel.y,
                v_z=earth_sun_vel_rel.z,
                frame='icrs',
                representation_type='cartesian',
                differential_type='cartesian'
            )
            
            # Transform to Galactic frame
            earth_gal = earth_icrs.transform_to('galactic')
            
            # Extract just the velocity components in Galactic coordinates
            earth_vel_gal_xyz = earth_gal.velocity.d_xyz.to(apu.km/apu.s).value
            
            # Add Sun's galactic motion if requested
            if not Earth_rel_Sun:
                earth_vel_gal_xyz = earth_vel_gal_xyz + sun_vel_Gal
            
            earth_velocities[i] = earth_vel_gal_xyz
        
        return times, sun_positions, sun_velocities, earth_velocities


    @staticmethod
    def SunEarthGC_frame_coords(year):

        # Define the Galactic Center coordinates
        gc = SkyCoord(l=0*apu.degree, b=0*apu.degree, frame='galactic')
        
        # Sun's velocity w.r.t. CNB in galactic coordinates/frame (km/s)
        sun_vel_CNB = np.array([-0.06935679, -0.66222051, 0.74609224]) * 369.82

        # Sun's velocity w.r.t. MWGC in galactic coordinates/frame (km/s)
        sun_vel_MW = np.array([11.1029257, 250.8323298, 7.3656565])
        
        # Choose which relative velocity:
        #? I think vel_CNB for unbound, vel_MW for bound
        # sun_vel_Gal = sun_vel_CNB
        sun_vel_Gal = sun_vel_MW

        # Lists for Sun's positions and velocities, and Earth's velocities
        sun_positions = []
        sun_vels = []
        earth_vels = []
        
        # Loop backwards through each day of the year
        for day in range(365, 0, -1):  
            
            # Time objects of current and "next" (in the past) day at noon UTC
            time = Time(f'{year}-12-31 12:00:00') - (365-day)*apu.day
            time_next = time - 1*apu.day
            
            # Earth's position in ICRS coords. at current and "next" day (in km)
            earth_icrs = SkyCoord(
                get_body_barycentric('earth', time), 
                frame='icrs').cartesian.xyz
            earth_icrs_next = SkyCoord(
                get_body_barycentric('earth', time_next), 
                frame='icrs').cartesian.xyz
            
            # Calculate Earth's velocity in ICRS frame (km/s)
            earth_vel_icrs = (earth_icrs_next - earth_icrs).to(apu.km).value / apu.day.to(apu.s)
            
            # Get Sun's position in GCRS coordinates (relative to Earth) in km
            sun_gcrs = SkyCoord(
                get_body_barycentric('sun', time), 
                frame='icrs').transform_to(GCRS(obstime=time))
            sun_vector = sun_gcrs.cartesian.xyz.to(apu.km).value
            
            # Transform the Galactic Center to GCRS
            gc_gcrs = gc.transform_to(GCRS(obstime=time))
            
            # Calculate the unit vector pointing from Earth to Galactic Center
            gc_vector = gc_gcrs.cartesian.xyz.value
            gc_unit_vector = gc_vector / np.linalg.norm(gc_vector)
            
            # Define the new coordinate system
            x_axis = gc_unit_vector
            z_axis = np.cross(x_axis, [0, 0, 1])
            z_axis = z_axis / np.linalg.norm(z_axis)
            y_axis = np.cross(z_axis, x_axis)
            
            # Create rotation matrix
            rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
            
            # Rotate Sun's position and velocity into new coordinate system
            sun_rotated = np.dot(rotation_matrix.T, sun_vector)
            sun_vel_rotated = np.dot(rotation_matrix.T, sun_vel_Gal)
            
            # Rotate Earth's velocity into the new coordinate system (km/s)
            earth_vel_rotated = np.dot(rotation_matrix.T, earth_vel_icrs)
            
            # Calculate Earth's velocity relative to CMB in the rotated frame (km/s)
            earth_vel_rotated = sun_vel_rotated + earth_vel_rotated
            
            # Append the rotated position and velocities to the lists
            sun_positions.append(sun_rotated)
            sun_vels.append(sun_vel_rotated)
            earth_vels.append(earth_vel_rotated)
        
        # note: sun_positions: km units ; velocities: km/s units
        return np.array(sun_positions), np.array(sun_vels), np.array(earth_vels)


    @staticmethod
    def SunEarthGC_frame_coords_circular(year):
        """
        Calculate coordinates and velocities in a frame aligned with the Galactic Center,
        approximating Earth's orbit as perfectly circular.
        """
        # Define the Galactic Center coordinates
        gc = SkyCoord(l=0*apu.degree, b=0*apu.degree, frame='galactic')
        
        # Sun's velocity w.r.t. CNB in galactic coordinates/frame (km/s)
        sun_vel_CNB = np.array([-0.06935679, -0.66222051, 0.74609224]) * 369.82

        # Sun's velocity w.r.t. MWGC in galactic coordinates/frame (km/s)
        sun_vel_MW = np.array([11.1029257, 250.8323298, 7.3656565])

        # Choose which relative velocity:
        #? I think vel_CNB for unbound, vel_MW for bound
        # sun_vel_Gal = sun_vel_CNB
        sun_vel_Gal = sun_vel_MW

        # Constants
        earth_orbital_period = 365.256363004  # days
        avg_earth_sun_distance = 149597870.7  # km (1 AU)
        orbital_speed = 29.78  # km/s (Earth's average orbital speed)

        sun_positions = []
        sun_vels = []
        earth_vels = []
        
        # Loop backwards through each day of the year
        for day in range(365, 0, -1):
            # Time object of current day at noon UTC
            time = Time(f'{year}-12-31 12:00:00') - (365-day)*apu.day
            
            # Calculate angle in the orbital plane (in radians)
            angle = 2 * np.pi * day / earth_orbital_period
            
            # Calculate simplified circular position
            # Using the ecliptic plane approximation
            x = avg_earth_sun_distance * np.cos(angle)
            y = avg_earth_sun_distance * np.sin(angle)
            z = 0  # Assuming orbit is in x-y plane
            #? I think we can do this because the paper which computes the 
            #? Earth-Sun orbital dynamics said that the orbit lies to a very 
            #? good approximation in the ecliptic plane
            sun_vector = np.array([x, y, z])
            
            # Calculate simplified circular velocity
            # Velocity is perpendicular to position in circular motion
            vel_x = -orbital_speed * np.sin(angle)
            vel_y = orbital_speed * np.cos(angle)
            vel_z = 0
            earth_vel_circular = np.array([vel_x, vel_y, vel_z])
            
            # Transform the Galactic Center to GCRS
            gc_gcrs = gc.transform_to(GCRS(obstime=time))
            
            # Calculate the unit vector pointing from Earth to Galactic Center
            gc_vector = gc_gcrs.cartesian.xyz.value
            gc_unit_vector = gc_vector / np.linalg.norm(gc_vector)
            
            # Define the new coordinate system
            x_axis = gc_unit_vector
            z_axis = np.cross(x_axis, [0, 0, 1])
            z_axis = z_axis / np.linalg.norm(z_axis)
            y_axis = np.cross(z_axis, x_axis)
            
            # Create rotation matrix
            rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
            
            # Rotate Sun's position and velocity into new coordinate system
            sun_rotated = np.dot(rotation_matrix.T, sun_vector)
            sun_vel_rotated = np.dot(rotation_matrix.T, sun_vel_Gal)
            
            # Rotate Earth's velocity into the new coordinate system
            earth_vel_rotated = np.dot(rotation_matrix.T, earth_vel_circular)
            
            # Calculate Earth's total velocity relative to CMB in the rotated frame
            earth_vel_rotated = sun_vel_rotated + earth_vel_rotated
            
            # Append the rotated position and velocities to the lists
            sun_positions.append(sun_rotated)
            sun_vels.append(sun_vel_rotated)
            earth_vels.append(earth_vel_rotated)
        
        return np.array(sun_positions), np.array(sun_vels), np.array(earth_vels)


    @staticmethod
    @jax.jit
    def v_infinity(v_s, r_s):
        """
        Calculate the initial Solar-frame velocity for particles, s.t. they have velocity v_s at Earth's location.
        """
        
        v_GM = 2*Params.G*Params.Msun/jnp.linalg.norm(r_s, axis=-1)
        v_inf2 = jnp.linalg.norm(v_s, axis=-1)**2 - v_GM
        v_inf = jnp.sqrt(jnp.maximum(0, v_inf2))
        # v_inf = jnp.sqrt(v_inf2)
        r_s_unit = r_s / jnp.linalg.norm(r_s, axis=-1)
        vr_s_dot = jnp.dot(v_s, r_s_unit)

        numer = v_inf2[..., None]*v_s + v_inf[..., None]*v_GM/2*r_s_unit - v_inf[..., None]*v_s*vr_s_dot[..., None]
        denom = v_inf2 + v_GM/2 - v_inf*vr_s_dot
        
        return numer / denom[..., None]
