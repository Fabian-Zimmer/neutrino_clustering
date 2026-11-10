"""Select a reduced set of grid cells as starting positions (neutrino halo project)."""


import numpy as np
from scipy.spatial import cKDTree


class CellSelection():
    """Select a reduced set of starting cells."""


    @staticmethod
    def fibonacci_sphere(samples, radius=1):
        """Generate points on a sphere using the Fibonacci spiral method."""
        points = []
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius_at_y = np.sqrt(1 - y * y)  # radius at y
            theta = phi * i  # golden angle increment
            x = np.cos(theta) * radius_at_y
            z = np.sin(theta) * radius_at_y
            points.append((x * radius, y * radius, z * radius))
        return np.array(points)


    @staticmethod
    def select_spherical_grid(cell_centers, target_n, n_shells=10):
        """
        Select cells from the original grid to form a spherically symmetric grid.
        
        Parameters:
        cell_centers (np.array): Array of shape (N, 3) containing cell center coordinates.
        target_n (int): Target number of cells in the final grid.
        n_shells (int): Number of spherical shells to use.
        
        Returns:
        np.array: Selected cell centers forming a spherically symmetric grid.
        """
        # Find the center and maximum radius of the original grid
        center = np.mean(cell_centers, axis=0)
        max_radius = np.max(np.linalg.norm(cell_centers - center, axis=1))
        
        # Create KD-Tree for efficient nearest neighbor search
        tree = cKDTree(cell_centers)
        
        selected_cells = []
        cells_per_shell = target_n // n_shells
        
        for i in range(n_shells):
            radius = max_radius * (i + 1) / n_shells
            # Generate points on the sphere for this shell
            sphere_points = CellSelection.fibonacci_sphere(cells_per_shell, radius) + center
            
            # Find the nearest cells to these points
            _, indices = tree.query(sphere_points)
            selected_cells.extend(cell_centers[indices])
        
        # If we haven't reached target_n, add more cells from the last shell
        while len(selected_cells) < target_n:
            extra_points = CellSelection.fibonacci_sphere(target_n - len(selected_cells), max_radius) + center
            _, indices = tree.query(extra_points)
            selected_cells.extend(cell_centers[indices])
        
        return np.array(selected_cells[:target_n])


    @staticmethod
    def reduce_cells(cell_centers, target_n, k_neighbors=10):
        """
        Reduce the number of cells while preserving regions of higher cell density.
        
        Parameters:
        cell_centers (np.array): Array of shape (N, 3) containing cell center coordinates.
        target_n (int): Target number of cells after reduction.
        k_neighbors (int): Number of neighbors to consider for density estimation.
        
        Returns:
        np.array: Reduced array of cell centers with shape (target_n, 3).
        """
        N = len(cell_centers)
        if target_n >= N:
            return cell_centers[:target_n]  # Return all cells if target_n is greater or equal
        
        # Create a KD-tree for efficient nearest neighbor search
        tree = cKDTree(cell_centers)
        
        # Estimate local density for each cell
        distances, _ = tree.query(cell_centers, k=k_neighbors)
        local_density = 1 / np.mean(distances, axis=1)
        
        # Normalize densities to use as weights
        weights = local_density / np.sum(local_density)
        
        # Use weights to select cells
        np.random.seed(0)
        selected_indices = np.random.choice(N, size=target_n, replace=False, p=weights)
        
        return cell_centers[selected_indices]
