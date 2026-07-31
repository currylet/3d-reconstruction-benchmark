import os
import subprocess
import tempfile
import numpy as np
import open3d as o3d

from .base_wrapper import BaseReconstructor

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
DEFAULT_POISSON_EXE = os.path.join(PROJECT_ROOT, "bin/PoissonRecon")

class PoissonReconstructor(BaseReconstructor):
    def __init__(self, poisson_exe=DEFAULT_POISSON_EXE, point_weight=0, depth=8):
        self.poisson_exe = poisson_exe
        self.point_weight = point_weight
        self.depth = depth

        if not os.path.isfile(self.poisson_exe):
            raise FileNotFoundError(f"PoissonRecon executable not found: {self.poisson_exe}")

    def reconstruct(self, points, normals, **kwargs):
        """
        Reconstruct a surface mesh from an oriented point cloud using the PoissonRecon executable.

        The input points and normals are temporarily saved as an oriented point cloud in PLY format.
        PoissonRecon is then executed with the specified reconstruction parameters, and the resulting
        mesh is loaded and returned as vertex coordinates and triangle face indices.

        Args:
            points (np.ndarray): Point coordinates of the input point cloud, shape (N, 3)
            normals (np.ndarray): Surface normals corresponding to the input points, shape (N, 3)
            **kwargs: Additional optional arguments for compatibility with the common reconstruction interface

        Returns:
            vertices (np.ndarray): Vertex coordinates of the reconstructed mesh, shape (M, 3)
            faces (np.ndarray): Triangle face indices of the reconstructed mesh, shape (F, 3)
            points = np.asarray(points)
            normals = np.asarray(normals)
        """
        if len(points) == 0:
            raise ValueError("Input point cloud is empty.")

        if points.shape != normals.shape:
            raise ValueError(f"Points and normals must have the same shape: {points.shape} vs {normals.shape}")
        
        # create an oriented Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)

        # create temporary files for PoissonRecon input and output
        with tempfile.TemporaryDirectory() as tmpdir:
            input_ply = os.path.join(tmpdir, "input.ply")
            output_mesh = os.path.join(tmpdir, "output.ply")

            if not o3d.io.write_point_cloud(input_ply, pcd):
                raise RuntimeError(f"Failed to write temporary point cloud: {input_ply}")
            
            # build and run the PoissonRecon command
            command = [str(self.poisson_exe), "--in", input_ply, "--out", output_mesh, "--depth", str(self.depth), "--pointWeight", str(self.point_weight)]

            subprocess.run(command, check=True)

            if not os.path.exists(output_mesh):
                raise RuntimeError(f"Output mesh was not created: {output_mesh}")

            mesh = o3d.io.read_triangle_mesh(output_mesh)

            ## 裁切严重飞面
            bbox = pcd.get_axis_aligned_bounding_box()
            mesh = mesh.crop(bbox)

            if mesh.is_empty():
                raise RuntimeError("PoissonRecon returned an empty mesh.")

            # convert the mesh to NumPy arrays
            vertices = np.asarray(mesh.vertices).copy()
            faces = np.asarray(mesh.triangles).copy()

        return vertices, faces


class PSRReconstructor(PoissonReconstructor):
    def __init__(self, poisson_exe=DEFAULT_POISSON_EXE, depth=8):
        super().__init__(poisson_exe=poisson_exe, point_weight=0, depth=depth)


class SPSRReconstructor(PoissonReconstructor):
    def __init__(self, poisson_exe=DEFAULT_POISSON_EXE, depth=8, point_weight=4):
        super().__init__(poisson_exe=poisson_exe, point_weight=point_weight, depth=depth)