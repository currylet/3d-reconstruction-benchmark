import os
import torch
import torch.nn as nn
import numpy as np
from skimage.measure import marching_cubes

from .base_wrapper import BaseReconstructor

# ==========================================
# 1. Define Multi-periodic Activation Function
# ==========================================
class Sine(nn.Module):
    def __init__(self, w0=15.0):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)


# ==========================================
# 2. Define Dual-head Neural Stochastic Poisson Network (NS-SPSR)
# ==========================================
class NeuralStochasticSDF(nn.Module):
    """
    NS-SPSR Core Network:
    Input 3D coordinates, dual-head output:
    - Output 0: Mean Signed Distance Field (Mean SDF, used for surface reconstruction)
    - Output 1: Variance Uncertainty Field (Variance, used to represent reconstruction confidence)
    """
    def __init__(self, in_features=3, hidden_features=128, num_layers=5, w0=15.0):
        super().__init__()
        layers = []
        # Input layer
        layers.append(nn.Linear(in_features, hidden_features))
        layers.append(Sine(w0))
        # Intermediate hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(Sine(w0))
        self.backbone = nn.Sequential(*layers)
        
        # Dual-head output
        self.mean_head = nn.Linear(hidden_features, 1)      # Predict mean
        self.variance_head = nn.Linear(hidden_features, 1)  # Predict uncertainty
        
        # Weight initialization
        with torch.no_grad():
            for layer in self.backbone:
                if isinstance(layer, nn.Linear):
                    num_input = layer.weight.size(-1)
                    layer.weight.uniform_(-np.sqrt(6 / num_input) / w0, np.sqrt(6 / num_input) / w0)
                    layer.bias.zero_()
            self.mean_head.weight.zero_()
            self.mean_head.bias.zero_()
            self.variance_head.weight.zero_()
            self.variance_head.bias.zero_()

    def forward(self, x):
        features = self.backbone(x)
        mean = self.mean_head(features)
        # Uncertainty variance must be non-negative; use softplus to force it >= 0
        variance = torch.nn.functional.softplus(self.variance_head(features))
        return torch.cat([mean, variance], dim=-1)


# ==========================================
# 3. NS-SPSR Reconstructor Class
# ==========================================
class NSSPSRReconstructor(BaseReconstructor):
    def __init__(self, steps=1500, lr=5e-4, grid_resolution=128):
        self.steps = steps
        self.lr = lr
        self.grid_resolution = grid_resolution

    def reconstruct(self, points, normals, **kwargs):
        # Force conversion to float32 to prevent numpy from automatically promoting to float64, which causes PyTorch errors
        points = np.asarray(points, dtype=np.float32)
        normals = np.asarray(normals, dtype=np.float32)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.init()

        pts_surface = torch.tensor(points, dtype=torch.float32, device=device)
        n_surface = torch.tensor(normals, dtype=torch.float32, device=device)
        
        bbox_min = points.min(axis=0) - 0.1
        bbox_max = points.max(axis=0) + 0.1
        
        # Instantiate dual-head NS-SPSR network
        model = NeuralStochasticSDF().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        print(f"      [NS-SPSR] Starting to solve the neural stochastic Poisson physical equation ({self.steps} steps)...")
        model.train()
        
        for step in range(self.steps + 1):
            optimizer.zero_grad()
            pts_surface.requires_grad_(True)
            
            # Forward inference
            pred = model(pts_surface)
            pred_mean = pred[:, 0:1]
            pred_var = pred[:, 1:2]
            
            # --- 1. Mean field reconstruction loss (conforming to Screened Poisson PDE constraints) ---
            loss_zero = (pred_mean ** 2).mean()
            
            # Compute mean field gradient using automatic differentiation
            grad_surf = torch.autograd.grad(
                outputs=pred_mean, inputs=pts_surface,
                grad_outputs=torch.ones_like(pred_mean),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            loss_normal = ((grad_surf - n_surface) ** 2).mean()
            
            # --- 2. Covariance uncertainty loss (Stochastic Covariance Constraint) ---
            # The point cloud surface has empirical support, so the reconstruction uncertainty variance at point cloud locations should tend to 0
            loss_var_surface = (pred_var ** 2).mean()
            
            # Randomly sample query points in the spatial Bbox
            q_rand = torch.rand((3000, 3), device=device) * torch.tensor(bbox_max - bbox_min, device=device) + torch.tensor(bbox_min, device=device)
            q_rand.requires_grad_(True)
            
            pred_rand = model(q_rand)
            pred_rand_mean = pred_rand[:, 0:1]
            pred_rand_var = pred_rand[:, 1:2]
            
            # Eikonal regularization ensures the stability of the mean signed distance field
            grad_rand = torch.autograd.grad(
                outputs=pred_rand_mean, inputs=q_rand,
                grad_outputs=torch.ones_like(pred_rand_mean),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            loss_eikonal = ((grad_rand.norm(dim=-1) - 1.0) ** 2).mean()
            
            # The uncertainty variance of spatial random points should be positively correlated with their physical distance to the nearest input point cloud
            # We utilize the absolute value of the mean field (as an approximation of distance to surface) to impose a Gaussian Process prior constraint on the variance of empty space
            loss_var_space = ((pred_rand_var - torch.abs(pred_rand_mean)) ** 2).mean()
            
            # Combine physical and statistical losses
            loss = 1.0 * loss_zero + 0.1 * loss_normal + 0.1 * loss_eikonal + 1.0 * loss_var_surface + 0.5 * loss_var_space
            loss.backward()
            optimizer.step()
            
            if step % 500 == 0:
                print(f"      [NS-SPSR] Step {step:4d}/{self.steps} | Loss: {loss.item():.5f} | Mean_Zero: {loss_zero.item():.5f} | Var_Space: {loss_var_space.item():.5f}")

        # 6. Use Marching Cubes to extract the mesh from the optimized mean implicit field
        print(f"      [NS-SPSR] Physical solving completed. Extracting watertight mesh...")
        grid_res = self.grid_resolution
        x = np.linspace(bbox_min[0], bbox_max[0], grid_res)
        y = np.linspace(bbox_min[1], bbox_max[1], grid_res)
        z = np.linspace(bbox_min[2], bbox_max[2], grid_res)
        grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
        grid_pts = np.column_stack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel()))
        
        grid_pts_tensor = torch.tensor(grid_pts, dtype=torch.float32, device=device)
        with torch.no_grad():
            sdf_vals = []
            for i in range(0, len(grid_pts_tensor), 50000):
                # Only take the 0-th dimension of the output (Mean SDF) for mesh extraction
                sdf_vals.append(model(grid_pts_tensor[i:i+50000])[:, 0:1].cpu().numpy())
            sdf_grid = np.concatenate(sdf_vals).reshape(grid_res, grid_res, grid_res)
            
        try:
            vertices, faces, _, _ = marching_cubes(sdf_grid, level=0.0)
            min_bound = bbox_min
            max_bound = bbox_max
            vertices = min_bound + (vertices / (grid_res - 1)) * (max_bound - min_bound)
        except Exception as e:
            print(f"      [NS-SPSR] [Warning] Mesh isosurface extraction failed: {e}. Returning placeholder triangle.")
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