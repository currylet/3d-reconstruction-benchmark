import numpy as np
import open3d as o3d
import point_cloud_utils as pcu

def evaluate_mesh(gt_points_path, recon_vertices, recon_faces, num_samples=10000):
    """
    Compute the geometric reconstruction error between the reconstructed mesh and the ground truth clean point cloud (.npy / .ply).
    The evaluation is performed in a unified normalized space to ensure a scale-independent, rigorous metric.
    
    Args:
        gt_points_path (str): Path to the ground truth clean point cloud (supports .npy, .ply, .obj formats, etc.)
        recon_vertices (np.ndarray): Normalized vertex coordinates of the reconstructed mesh, shape (M, 3)
        recon_faces (np.ndarray): Face indices of the reconstructed mesh, shape (F, 3)
        num_samples (int): Number of points to resample on the surface of the reconstructed mesh (default 10000)
        
    Returns:
        metrics (dict): A dictionary containing Chamfer distance and Hausdorff distance
    """
    # 1. Load the original, clean ground truth point cloud (adaptively supporting .npy, .ply, and .obj)
    if gt_points_path.endswith(".npy"):
        gt_points = np.load(gt_points_path)
        gt_points = np.asarray(gt_points[:, :3], dtype=np.float32)
    elif gt_points_path.endswith(".ply") or gt_points_path.endswith(".obj"):
        pcd_temp = o3d.io.read_point_cloud(gt_points_path)
        gt_points = np.asarray(pcd_temp.points, dtype=np.float32)
    else:
        raise ValueError(f"The evaluation function does not support the ground truth point cloud format: {gt_points_path}")
    
    # 2. Perform the exact same normalization on the ground truth point cloud as in dataset.py
    center = np.mean(gt_points, axis=0)
    gt_points_centered = gt_points - center
    scale = np.max(np.linalg.norm(gt_points_centered, axis=1))
    if scale == 0:
        scale = 1.0
    gt_points_norm = gt_points_centered / scale

    # 3. Fault tolerance: if reconstruction fails (producing placeholder empty triangles), return a penalty value of 999.0
    if len(recon_vertices) <= 3 or len(recon_faces) == 0:
        return {
            "chamfer_distance": 999.0,
            "hausdorff_distance": 999.0
        }

    # 4. Uniformly sample 10000 points with high density on the surface of the reconstructed triangular mesh
    try:
        # pcu.sample_mesh_random official specification returns face indices (fid) and barycentric coordinates (bc)
        fid, bc = pcu.sample_mesh_random(
            recon_vertices.astype(np.float64), 
            recon_faces.astype(np.int32), 
            num_samples
        )
        
        # Must use pcu's built-in barycentric coordinate interpolation function to compute and reconstruct the actual 3D sampled point coordinates!
        recon_samples = pcu.interpolate_barycentric_coords(
            recon_faces.astype(np.int32), 
            fid, 
            bc, 
            recon_vertices.astype(np.float64)
        )
        recon_samples = np.asarray(recon_samples, dtype=np.float32)
    except Exception as e:
        print(f"      [Evaluator] [Warning] Mesh surface resampling failed: {e}. Returning penalty metrics.")
        return {
            "chamfer_distance": 999.0,
            "hausdorff_distance": 999.0
        }

    # 5. Compute Chamfer distance and Hausdorff distance
    chamfer_dist = pcu.chamfer_distance(gt_points_norm, recon_samples)
    hausdorff_dist = pcu.hausdorff_distance(gt_points_norm, recon_samples)

    return {
        "chamfer_distance": float(chamfer_dist),
        "hausdorff_distance": float(hausdorff_dist)
    }