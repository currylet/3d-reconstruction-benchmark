import os
import torch
import torch.nn as nn
import numpy as np
import open3d as o3d

from .base_wrapper import BaseReconstructor

# ==========================================
# 1. Chamfer Distance Loss
# ==========================================
def pytorch_chamfer_distance(x, y):
    x_sq = torch.sum(x ** 2, dim=-1, keepdim=True)  # (N, 1)
    y_sq = torch.sum(y ** 2, dim=-1, keepdim=True).transpose(-2, -1)  # (1, M)
    xy = torch.matmul(x, y.transpose(-2, -1))  # (N, M)
    dist = torch.clamp(x_sq - 2 * xy + y_sq, min=0.0)

    min_dist_x = torch.min(dist, dim=-1)[0]  # (N,)
    min_dist_y = torch.min(dist, dim=-2)[0]  # (M,)
    return torch.mean(min_dist_x) + torch.mean(min_dist_y)


# ==========================================
# 2. Farthest Point Sampling
# ==========================================
def farthest_point_sampling(pts, num_samples):
    farthest_pts = np.zeros(num_samples, dtype=np.int32)
    distances = np.ones(len(pts)) * 1e10
    farthest_pts[0] = np.random.randint(len(pts))
    
    for i in range(1, num_samples):
        curr_pt = pts[farthest_pts[i-1]]
        dist = np.sum((pts - curr_pt) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        farthest_pts[i] = np.argmax(distances)
        
    return farthest_pts


# ==========================================
# 3. Local Chart Fitting Network (2D -> 3D Local Displacement)
# ==========================================
class LocalChartMLP(nn.Module):
    def __init__(self, hidden_features=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, 3)
        )

    def forward(self, uv):
        return self.net(uv)


# ==========================================
# 4. DGP Upgraded Reconstructor
# ==========================================
class DGPReconstructor(BaseReconstructor):
    def __init__(self, num_charts=50, patch_size=300, steps=800, lr=1e-3, spsr_depth=8):
        """
        DGP Upgraded Version: Introducing local coordinate system and Tanh radius lock
        """
        self.num_charts = num_charts
        self.patch_size = patch_size
        self.steps = steps
        self.lr = lr
        self.spsr_depth = spsr_depth

    def reconstruct(self, points, normals, **kwargs):
        # Force conversion to float32 to prevent numpy from automatically promoting to float64, which causes PyTorch errors
        points = np.asarray(points, dtype=np.float32)
        normals = np.asarray(normals, dtype=np.float32)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.init()

        N = len(points)
        print(f"      [DGP] Running Farthest Point Sampling (FPS) to partition {self.num_charts} small local charts...")
        
        # 1. Determine chart centers using FPS
        anchor_indices = farthest_point_sampling(points, self.num_charts)
        anchors = points[anchor_indices]

        # 2. k-NN neighborhood search to establish highly flat local patches
        patches_pts = []
        patches_normals = []
        for anchor in anchors:
            dists = np.sum((points - anchor) ** 2, axis=1)
            knn_indices = np.argsort(dists)[:min(self.patch_size, N)]
            patches_pts.append(points[knn_indices])
            patches_normals.append(normals[knn_indices])

        # 3. Establish local manifold networks and optimize local coordinate systems
        print(f"      [DGP] Starting parallel optimization of {self.num_charts} local manifold charts on the GPU...")
        models = [LocalChartMLP().to(device) for _ in range(self.num_charts)]
        
        for chart_idx in range(self.num_charts):
            model = models[chart_idx]
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            
            # 3.1 Core: Transform ground truth point cloud to the local coordinate system with Anchor as the origin
            anchor_np = anchors[chart_idx]
            anchor_tensor = torch.tensor(anchor_np, dtype=torch.float32, device=device)
            
            target_pts_global = torch.tensor(patches_pts[chart_idx], dtype=torch.float32, device=device)
            target_pts_local = target_pts_global - anchor_tensor  # Subtract center, convert to relative coordinates
            
            # Compute the maximum radius of the current patch, used for physical locking
            patch_radius = torch.max(torch.norm(target_pts_local, dim=-1))
            
            model.train()
            for step in range(self.steps):
                optimizer.zero_grad()
                
                # Sample 2D parameters
                uv_samples = torch.rand((len(target_pts_global), 2), device=device)
                
                # 3.2 Use Tanh to limit the local output range, then multiply by the actual physical radius to mathematically forbid points from flying out
                pred_pts_local = torch.tanh(model(uv_samples)) * patch_radius
                
                # Compute Chamfer loss in the local coordinate system
                loss = pytorch_chamfer_distance(pred_pts_local, target_pts_local)
                loss.backward()
                optimizer.step()

            if (chart_idx + 1) % 15 == 0 or chart_idx == 0:
                print(f"      - Chart [{chart_idx+1}/{self.num_charts}] local manifold fitting completed.")

        # 4. High-density resampling to generate dense, smooth point clouds (eliminate tedious and slow normal matching, generating only coordinates)
        print(f"      [DGP] Resampling from optimized local manifolds to generate ultra-high-density point clouds...")
        dense_points_list = []
        num_dense_samples_per_patch = 8000
        
        for chart_idx in range(self.num_charts):
            model = models[chart_idx]
            model.eval()
            
            anchor_np = anchors[chart_idx]
            patch_original_pts = patches_pts[chart_idx]
            patch_radius_np = np.max(np.linalg.norm(patch_original_pts - anchor_np, axis=1))
            
            with torch.no_grad():
                uv_dense = torch.rand((num_dense_samples_per_patch, 2), device=device)
                pred_dense_local = torch.tanh(model(uv_dense)).cpu().numpy() * patch_radius_np
                pred_dense_global = pred_dense_local + anchor_np  # Add back the center, convert back to global coordinates
                dense_points_list.append(pred_dense_global)

        # Combine dense coordinates
        dense_points = np.vstack(dense_points_list)

        # 5. SPSR post-processing to generate the final clean watertight mesh
        print(f"      [DGP] Dense point cloud synthesis completed (total {len(dense_points)} points), re-estimating and aligning normal vectors consistently using Open3D...")
        
        dense_pcd = o3d.geometry.PointCloud()
        dense_pcd.points = o3d.utility.Vector3dVector(dense_points)
        
        # 5.1 Re-estimate and consistently align normals on the C++ side, completely solving the normal flipping issue at thin walls with extremely fast speed
        dense_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        dense_pcd.orient_normals_consistent_tangent_plane(30)
        dense_pcd.normalize_normals()

        print(f"      [DGP] Normal alignment completed, performing SPSR post-processing and automatically cropping boundaries...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            dense_pcd, depth=self.spsr_depth
        )
        
        # 5.2 Crop the reconstructed mesh using the tight Bounding Box of the point cloud, eliminating the huge peripheral "curtain/background artifacts"
        bbox = dense_pcd.get_axis_aligned_bounding_box()
        cropped_mesh = mesh.crop(bbox)
        
        vertices = np.asarray(cropped_mesh.vertices)
        faces = np.asarray(cropped_mesh.triangles)
        
        return vertices, faces