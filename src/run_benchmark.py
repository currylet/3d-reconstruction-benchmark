import os
import sys

# Ensure Python can correctly locate the modules in the `src` directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import load_and_preprocess_point_cloud
from evaluators import evaluate_mesh
from utils import Timer, GPUMemoryMonitor, save_reconstructed_mesh, log_metrics_to_csv

# ==========================================
# Import algorithm encapsulation class
# ==========================================
# PSR, SPSR, DGP, STO-PSR, NS-SPSR
from wrappers.psr_spsr import PSRReconstructor, SPSRReconstructor
from wrappers.dgp_wrapper import DGPReconstructor
from wrappers.stoc_psr import StochasticPSRReconstructor
from wrappers.ns_spsr import NSSPSRReconstructor


# ==========================================
# Define test sets and evaluation scenarios
# ==========================================

# List of test models
TEST_MODELS = {
    "model_01": "../data/pointcloud/01.xyz.npy",
    "model_02": "../data/pointcloud/02.xyz.npy",
    "model_03": "../data/pointcloud/03.xyz.npy",
    "model_teapot": "../data/pointcloud/Utah_teapot_(solid).xyz.npy"
}

# Evaluation scenario setup (controlled variable experiment)
# Format: "Scenario Name": {Data Processing Parameters}
EVAL_SETTINGS = {
    "Clean": {
        "noise_std": 0.0, "sampling_ratio": 1.0, "normal_noise_std": 0.0
    },
    "Light_Point_Noise": {
        "noise_std": 0.005, "sampling_ratio": 1.0, "normal_noise_std": 0.0
    },
    "Heavy_Point_Noise": {
        "noise_std": 0.02, "sampling_ratio": 1.0, "normal_noise_std": 0.0
    },
    "Light_Normal_Noise": {
        "noise_std": 0.0, "sampling_ratio": 1.0, "normal_noise_std": 0.02
    },
    "Heavy_Normal_Noise": {
        "noise_std": 0.0, "sampling_ratio": 1.0, "normal_noise_std": 0.08
    },
    "Sparsity": {
        "noise_std": 0.0, "sampling_ratio": 0.1, "normal_noise_std": 0.0
    },
    "Holes": {
         "noise_std": 0.0, "sampling_ratio": 1.0, "normal_noise_std": 0.0, "num_holes": 1, "hole_radius": 0.15
    },
    "Outliers": {
        "noise_std": 0.0, "sampling_ratio": 1.0, "normal_noise_std": 0.0, "outlier_ratio": 0.01
    }
}


# ==========================================
# Main Evaluation Pipeline
# ==========================================
def main():
    ## 这里改了路径，方便直接创建/覆盖文件夹
    results_root = "../results"
    csv_log_path = os.path.join(results_root, "metrics.csv")

    pointcloud_root = os.path.join(results_root, "pointclouds(debug)")
    mesh_root = os.path.join(results_root, "meshes")
    os.makedirs(pointcloud_root, exist_ok=True)
    os.makedirs(mesh_root, exist_ok=True)
    
    # Initialize all algorithms participating in the comparison
    methods = {
        "PSR": PSRReconstructor(depth=8),
        "SPSR": SPSRReconstructor(depth=8),
        "DGP": DGPReconstructor(num_charts=30, patch_size=1000, steps=800, spsr_depth=8),      
        "Stochastic_PSR": StochasticPSRReconstructor(grid_size=48, solve_subspace_dim=2000),
        "NS-SPSR": NSSPSRReconstructor(steps=2500, lr=1e-3, grid_resolution=128)               
    }

    print("==================================================")
    print("      3D Surface Reconstruction Benchmark         ")
    print("==================================================")
    
    # Nested loop: Model -> Scenario -> Algorithm
    for model_name, model_path in TEST_MODELS.items():
        ## 这里也改了路径
        model_pointcloud_dir = os.path.join(pointcloud_root, model_name)
        os.makedirs(model_pointcloud_dir, exist_ok=True)

        model_mesh_dir = os.path.join(mesh_root, model_name)
        os.makedirs(model_mesh_dir, exist_ok=True)
            
        for setting_name, params in EVAL_SETTINGS.items():  
            print(f"\n>>> Evaluating model: [{model_name}] | Scenario: [{setting_name}]")
            
            # Load and apply geometric degradation
            # Pass configuration parameters to the function in `dataset.py` by unpacking the dictionary
            # save altered point cloud for reference
            ## 这里也改了路径
            debug_ply_path = os.path.join(model_pointcloud_dir, f"{setting_name}.ply")
            
            try:
                points, normals, scale, center = load_and_preprocess_point_cloud(
                    model_path,
                    output_path=debug_ply_path,
                    noise_std=params.get("noise_std", 0.0),
                    sampling_ratio=params.get("sampling_ratio", 1.0),
                    normal_noise_std=params.get("normal_noise_std", 0.0),
                    num_holes=params.get("num_holes", 0),
                    hole_radius=params.get("hole_radius", 0.15),
                    outlier_ratio=params.get("outlier_ratio", 0.0)
                )
            except Exception as e:
                print(f"[Error] Data preprocessing failed: {e}")
                continue
            
            # Run each algorithm in a loop
            for method_name, reconstructor in methods.items():
                print(f"    Running algorithm: {method_name} ...")
                
                # Timing and Hardware Monitoring
                try:
                    with Timer() as t, GPUMemoryMonitor() as m:
                        vertices, faces = reconstructor.reconstruct(points, normals)
                except Exception as e:
                    print(f"    [Fail] {method_name} An error occurred during execution: {e}")
                    continue
                
                # Save the generated mesh model
                ## 这里也改了路径，这样文件夹结构为 model_01/Clean/PSR_model_01_clean.obj
                setting_mesh_dir = os.path.join(model_mesh_dir, setting_name)
                os.makedirs(setting_mesh_dir, exist_ok=True)
                output_mesh_path = os.path.join(setting_mesh_dir, f"{method_name}_{model_name}_{setting_name}.obj")

                try:
                    save_reconstructed_mesh(
                        output_mesh_path, vertices, faces, scale, center
                    )
                except Exception as e:
                    print(f"    [Error] Failed to save the mesh: {e}")
                    continue
                
                # Evaluate reconstructed geometric accuracy (calculate Chamfer/Hausdorff distance)
                print(f"    Calculating geometric metrics...")
                try:
                    metrics = evaluate_mesh(model_path, vertices, faces)
                except Exception as e:
                    print(f"    [Error] Metric calculation failed (usually because the output grid is empty): {e}")
                    metrics = {"chamfer_distance": -1.0, "hausdorff_distance": -1.0}
                
                # Record data to a unified CSV table
                row_data = {
                    "Model": model_name,
                    "Setting": setting_name,
                    "Method": method_name,
                    "Chamfer_Dist": f"{metrics['chamfer_distance']:.6f}",
                    "Hausdorff_Dist": f"{metrics['hausdorff_distance']:.6f}",
                    "Time_Sec": f"{t.interval:.3f}",
                    "Peak_VRAM_MB": f"{m.peak_memory_mb:.2f}"
                }
                log_metrics_to_csv(csv_log_path, row_data)

                print(f"    [Success] {method_name} completed. Chamfer: {metrics['chamfer_distance']:.6f} | Time Consumption: {t.interval:.2f} seconds")
    
    print("\n==================================================")
    print("             All evaluation tasks have been completed!                ")
    print("==================================================")

if __name__ == "__main__":
    main()