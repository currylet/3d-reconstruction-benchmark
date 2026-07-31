import os
import numpy as np
from skimage.measure import marching_cubes
import gpytoolbox as gpy

from .base_wrapper import BaseReconstructor

class StochasticPSRReconstructor(BaseReconstructor):
    def __init__(self, grid_size=64, solve_subspace_dim=5000):
        """
        Stochastic PSR Reconstructor
        
        Args:
            grid_size (int): 3D grid edge size for solving the Gaussian Process (default 64x64x64)
            solve_subspace_dim (int): Subspace solver dimension (default 5000)
        """
        self.grid_size = grid_size
        self.solve_subspace_dim = solve_subspace_dim

    def reconstruct(self, points, normals, **kwargs):
        # 1. Determine the specific function name of gpytoolbox (compatibility check for old and new versions)
        if hasattr(gpy, 'stochastic_poisson_surface_reconstruction'):
            recon_func = gpy.stochastic_poisson_surface_reconstruction
            kwargs_stoc = {"output_variance": True}
        else:
            recon_func = gpy.poisson_surface_reconstruction
            kwargs_stoc = {"stochastic": True}

        # 2. Dynamically compute the axis-aligned bounding box of the point cloud, and expand by 0.1 as a buffer zone
        bbox_min = points.min(axis=0) - 0.1
        bbox_max = points.max(axis=0) + 0.1
        
        # 3. Configure grid parameters
        gs = np.array([self.grid_size, self.grid_size, self.grid_size], dtype=np.int32)
        h = (bbox_max - bbox_min) / (self.grid_size - 1)
        corner = bbox_min

        print(f"      [Stochastic PSR] Running Gaussian Process subspace solver (Subspace Dim: {self.solve_subspace_dim})...")
        
        # 4. Call gpytoolbox to solve the stochastic Poisson Gaussian Process
        # Returns mean scalar field, variance (uncertainty) field, and grid vertices
        scalar_mean, scalar_variance, grid_vertices = recon_func(
            points.astype(np.float64), 
            normals.astype(np.float64), 
            gs=gs, 
            h=h, 
            corner=corner,
            solve_subspace_dim=self.solve_subspace_dim,
            **kwargs_stoc
        )

        # 5. Extract the isosurface from the mean signed distance field using Marching Cubes
        print(f"      [Stochastic PSR] Poisson Gaussian Process solving completed. Extracting isosurface...")
        
        # Reshape the 1D flattened array returned by gpytoolbox into a 3D NumPy array
        # Reshaping must use the Fortran order ('F'), otherwise the coordinate axes of the geometry will be disordered
        try:
            scalar_mean_3d = np.reshape(
                scalar_mean, 
                (self.grid_size, self.grid_size, self.grid_size), 
                order='F'
            )
        except Exception as e:
            print(f"      [Stochastic PSR] Reshaping array to 3D failed: {e}")
            scalar_mean_3d = scalar_mean

        # Adaptively select the isosurface level (Iso-level)
        # The official stochastic Poisson algorithm internally performs distance shift, so its zero-isosurface is physically indeed 0.0
        # To absolutely prevent errors, we add a boundary check: ensure 0.0 is between the minimum and maximum scalar values
        level = 0.0
        if not (scalar_mean_3d.min() < level < scalar_mean_3d.max()):
            level = (scalar_mean_3d.min() + scalar_mean_3d.max()) / 2.0

        print(f"      [Stochastic PSR] Selected isosurface extraction height: {level:.2f}")

        try:
            # Run standard skimage.measure.marching_cubes
            vertices, faces, _, _ = marching_cubes(scalar_mean_3d, level=level)
            
            # Map the grid index coordinates from [0, grid_size-1] back to the physical 3D space
            min_bound = bbox_min
            max_bound = bbox_max
            vertices = min_bound + (vertices / (self.grid_size - 1)) * (max_bound - min_bound)
        except ValueError as e:
            print(f"      [Stochastic PSR] [Warning] Marching Cubes extraction failed ({e}), returning placeholder triangle.")
            vertices = np.array([[0,0,0], [0.1,0,0], [0,0.1,0]])
            faces = np.array([[0,1,2]])

        # =========================================================================
        # Fixed: Cropped the reconstructed mesh using the point cloud's tight bounding box to 
        # eliminate peripheral faces
        # =========================================================================
        import open3d as o3d
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
        
        pcd_temp = o3d.geometry.PointCloud()
        pcd_temp.points = o3d.utility.Vector3dVector(points)
        bbox = pcd_temp.get_axis_aligned_bounding_box()
        cropped_mesh = mesh.crop(bbox)
        
        vertices = np.asarray(cropped_mesh.vertices)
        faces = np.asarray(cropped_mesh.triangles)
        # =========================================================================

        return vertices, faces