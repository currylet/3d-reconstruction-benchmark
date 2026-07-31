# 3D Surface Reconstruction Benchmark

This repository implements a unified, robust benchmarking framework to compare and evaluate five distinct surface reconstruction paradigms from oriented point clouds:
1. **Poisson Surface Reconstruction (PSR)** (Kazhdan et al., 2006)
2. **Screened Poisson Surface Reconstruction (SPSR)** (Kazhdan & Hoppe, 2013)
3. **Explicit Deep Geometric Prior (DGP)** (Williams et al., 2019)
4. **Stochastic Screened Poisson Reconstruction** (Sellán & Jacobson, 2022)
5. **Neural Stochastic Screened Poisson (NS-SPSR)** (Sellán & Jacobson, 2023)

Our pipeline evaluates geometric reconstruction accuracy (Chamfer and Hausdorff distances), runtime efficiency, and peak VRAM footprints under various degradation settings (noise, sparsity, holes, and outliers).

---

## Directory Structure

```text
3d-reconstruction-benchmark/
├── bin/                    # Contains compiled third-party C++ executables
│   └── PoissonRecon        # Core C++ solver for classical PSR and SPSR
├── data/
│   └── mesh/               # Directory to place meshes
│   └── pointcloud/         # Directory to place point clouds
├── src/
│   ├── wrappers/           # Algorithm encapsulation layers inheriting from BaseReconstructor
│   │   ├── base_wrapper.py
│   │   ├── psr_spsr.py     # Classical SPSR & PSR wrapper
│   │   ├── dgp_wrapper.py  # Explicit local-charts DGP
│   │   ├── stoc_psr.py     # Stochastic SPSR wrapper
│   │   └── ns_spsr.py      # Neural Stochastic SPSR (Dual-head SIREN) wrapper
│   ├── dataset.py          # Unified data loading and degradation pipeline
│   ├── evaluators.py       # Adaptive Chamfer/Hausdorff metric computation
│   ├── utils.py            # Timers, GPU monitors, and CSV logging utilities
│   └── run_benchmark.py    # Master control room for batch evaluation
├── results/                # Output directory
│   ├── meshes/             # Reconstructed .obj meshes
│   ├── pointclouds/        # Degraded point clouds generated at runtime
│   └── metrics.csv         # Automatically generated evaluation results table
└── requirements.txt
```

---

## Environment Setup

We recommend using Anaconda/Miniconda to set up a clean Python 3.10 environment.

### 1. Create Environment and Install PyTorch with CUDA 13.0
Install the matching PyTorch CUDA binary:
```bash
conda create -n 3dvision python=3.10 -y
conda activate 3dvision

# Install PyTorch compiled with CUDA 13.0
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

### 2. Install Remaining Dependencies
```bash
pip install -r requirements.txt
```

---

## Prerequisites: Compiling `PoissonRecon`

The classical SPSR and PSR wrappers, as well as the explicit DGP post-processing, execute the official C++ reference binary developed by Misha Kazhdan. 

Run the following commands to compile the executable and place it into your project folder.

```bash
# 1. Create the bin directory inside your project
mkdir -p bin/

# 2. Clone the official PoissonRecon source code into a temporary directory
mkdir -p ~/tmp_build && cd ~/tmp_build
git clone https://github.com/mkazhdan/PoissonRecon.git
cd PoissonRecon

# 3. Install image dependencies locally via Conda
conda install -c conda-forge libjpeg-turbo libpng zlib -y
export CPATH=$CONDA_PREFIX/include:$CPATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 4. Compile the source code
make clean
make

# 5. Copy the executable to the project directory and clean up
cp Bin/Linux/PoissonRecon ~/3d-reconstruction-benchmark/bin/
rm -rf ~/tmp_build
```

---

## Running the Benchmark

### 1. Prepare Test Data
Place your clean oriented point cloud files (supporting both `.ply` and `.npy` formats) into the `data/pointcloud` directory.

### 2. Register Your Models
Open `src/run_benchmark.py` and add your files to the `TEST_MODELS` dictionary:
```python
TEST_MODELS = {
    "model_01": "../data/pointcloud/01.xyz.npy",
    "model_02": "../data/pointcloud/02.xyz.npy",
    "model_03": "../data/pointcloud/03.xyz.npy",
    "model_teapot": "../data/pointcloud/Utah_teapot_(solid).xyz.npy"
}
```

### 3. Run the master benchmark script
```bash
python src/run_benchmark.py
```

### 4. Collect Results
* **Mesh Visualizations:** Reconstructed water-tight `.obj` models are saved to `results/meshes/`.
* **Degraded Point Clouds:** Test point clouds generated under different robust settings are saved to `results/pointclouds/`.
* **Quantitative Data:** All evaluation statistics (CD, HD, Runtime, Peak VRAM) are automatically logged to `results/metrics.csv`.
```
