import os
import time
import csv
import torch
import numpy as np
import point_cloud_utils as pcu
import open3d as o3d

# ==========================================
# Grid Saving and Denormalization
# ==========================================
def save_reconstructed_mesh(file_path, vertices, faces, scale=1.0, center=None):
    """
    Save the reconstructed mesh to a file (supports .obj and .ply).
    Before writing, the grid coordinates are automatically restored 
    from the normalized [-1, 1] space to their original dimensions.
    
    Args:
        file_path (str): File path for saving (e.g., 'results/meshes/dgp_bunny.obj')
        vertices (np.ndarray): Vertex coordinates of the reconstructed mesh, shape (M, 3)
        faces (np.ndarray): Mesh face indices, shape (F, 3)
        scale (float): Scaling parameters for the normalization of the raw point cloud
        center (np.ndarray): Translation center used during the normalization of the raw point cloud, shape (3,)
    """
    # Deep-copy the vertex coordinates to prevent modifying the original predicted mesh in memory
    denorm_vertices = vertices.copy()
    
    # Denormalize：multiply by the scaling factor, add back the center of translation
    if scale != 1.0:
        denorm_vertices = denorm_vertices * scale

    if center is not None:
        denorm_vertices = denorm_vertices + center

    output_dir = os.path.dirname(file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(denorm_vertices.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))

    success = o3d.io.write_triangle_mesh(file_path, mesh)

    if not success:
        raise RuntimeError(f"Failed to save mesh: {file_path}")

    print(f"[IO] Successfully saved mesh to: {file_path}")


# ==========================================
# Context Manager for Timing Code Execution
# ==========================================
class Timer:
    """
    Context manager for high-precision measurement of code block execution time
    
    Usage:
        with Timer() as t:
            # Run the reconstruction algorithm...
            time.sleep(1) 
        print(f"time consumption: {t.interval:.3f} seconds")
    """
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start


# ==========================================
# PyTorch Context Manager for Tracking Peak VRAM Usage
# ==========================================
class GPUMemoryMonitor:
    """
    Context manager for tracking the peak additional VRAM usage of 
    a code block on a CUDA GPU
    
    Usage:
        with GPUMemoryMonitor() as m:
            # Run PyTorch neural network code....
        print(f"Maximum VRAM usage: {m.peak_memory_mb:.2f} MB")
    """
    def __enter__(self):
        if torch.cuda.is_available():
            # Reset the maximum video memory statistics for the current device
            torch.cuda.reset_peak_memory_stats()
            self.start_mem = torch.cuda.memory_allocated()
        else:
            self.start_mem = 0
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            # Record the peak video memory usage reached during this period
            self.peak_mem = torch.cuda.max_memory_allocated()
            # Convert to MB
            self.peak_memory_mb = (self.peak_mem - self.start_mem) / (1024 ** 2)
        else:
            self.peak_memory_mb = 0.0


# ==========================================
# Log metrics to CSV file
# ==========================================
def log_metrics_to_csv(csv_path, row_data):
    """
    Automatically records the various quantitative metrics from a test 
    into a unified CSV data table 
    If the CSV file does not exist, it is automatically created with 
    the header row written
    
    Args:
        csv_path (str): Path to the metrics file, typically 'results/metrics.csv'
        row_data (dict): The data dictionary to be recorded, 
        where keys represent fields and values represent numerical values
    
    Sample row_data:
        {
            "Model": "bunny",
            "Method": "DGP",
            "Noise_Std": 0.01,
            "Num_Holes": 1,
            "Chamfer_Dist": 0.0023,
            "Hausdorff_Dist": 0.0142,
            "Time_Sec": 32.5,
            "VRAM_MB": 128.5
        }
    """
    file_exists = os.path.exists(csv_path)
    
    # Create parent directories if they do not exist
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Write data in append mode ('a')
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        # If the file is newly created, write the header first
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)
        
    print(f"[Log] Evaluation data has been successfully written to: {csv_path}")