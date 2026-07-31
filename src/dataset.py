import os
import numpy as np
import open3d as o3d


def load_and_preprocess_point_cloud(input_path, output_path=None, noise_std=0.0, sampling_ratio=1.0, 
                                    normal_noise_std=0.0, num_holes=0, hole_radius=0.15, outlier_ratio=0.0,
                                    normal_k=30, seed=42):
    """
    Load and preprocess a point cloud for surface reconstruction.

    Args:
        input_path (str): path to the input point cloud file (.npy .obj .ply)
        output_path (str): path to save the processed point cloud file (.ply)
        noise_std (float): standard deviation of Gaussian noise added to point coordinates
        sampling_ratio (float): proportion of surface points retained after random downsampling; 1.0 keeps all points; (0, 1]
        normal_noise_std (float): standard deviation of Gaussian noise added to surface normals
        num_holes (int): number of synthetic holes created in the point cloud
        hole_radius (float): radius of each synthetic hole in normalized coordinates
        outlier_ratio (float): ratio of random outlier points relative to the number of surface points
        normal_k (int): number of nearest neighbors used for normal estimation and orientation
        seed (int): random seed for reproducibility

    Returns:
        info (dict): preprocessing information including 
                     original and processed point counts,
                     normalization scale, 
                     and original point cloud center
    """
    # set random seed for reproducibility
    rng = np.random.default_rng(seed)

    # load original point clouds
    ## 改动：兼容各种格式的点云输入
    if input_path.endswith(".npy"):
        points = np.load(input_path)
        points = np.asarray(points[:, :3], dtype=np.float32)
    elif input_path.endswith(".ply") or input_path.endswith(".obj"):
        pcd_temp = o3d.io.read_point_cloud(input_path)
        points = np.asarray(pcd_temp.points, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported input file format: {input_path}")

    # record the original number of points
    original_points = len(points)

    # centering
    center = np.mean(points, axis=0)
    points = points - center

    # normalization
    scale = np.max(np.linalg.norm(points, axis=1))
    points = points / scale

    # add synthetic holes
    if num_holes > 0:
        for _ in range(num_holes):
            hole_center = points[rng.choice(len(points))]
            distances = np.linalg.norm(points - hole_center, axis=1)
            points = points[distances > hole_radius]

    # add Gaussian noise to the points
    if noise_std > 0:
        points = points + rng.normal(0.0, noise_std, size=points.shape)

    # downsampling
    if sampling_ratio < 1.0:
        points = points[rng.choice(len(points), size=max(1, int(len(points) * sampling_ratio)), replace=False)]

    # add random outliers
    if outlier_ratio > 0:
        outliers = rng.uniform(low=-1.0, high=1.0, size=(int(len(points) * outlier_ratio), 3))
        points = np.vstack([points, outliers])

    # convert the NumPy point array to an Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # estimate normals
    # using k-nearest neighbors
    k = min(normal_k, len(points) - 1)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    # consistently orient neighboring normals
    pcd.orient_normals_consistent_tangent_plane(k)
    # normalize all normals to unit length
    pcd.normalize_normals()

    # add Gaussian noise to the normal vectors
    normals = np.asarray(pcd.normals).copy()
    if normal_noise_std > 0:
        normals += rng.normal(0.0, normal_noise_std, size=normals.shape)
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        lengths[lengths == 0] = 1.0 # avoid division by zero
        normals /= lengths # renormalize the normals to unit length
        pcd.normals = o3d.utility.Vector3dVector(normals)

    # optionally save processed point cloud
    if output_path is not None:
        output_dir = os.path.dirname(output_path)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        o3d.io.write_point_cloud(output_path, pcd)

    return points, normals, scale, center
