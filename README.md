# 3d-reconstruction-benchmark


(暂时，待讨论待修改)
#### **第一周：环境搭建、数据准备与模型**

| 时间 | Li(PSR / SPSR) | Zhu (DGP / Stochastic PSR / NS-SPSR) |
| :--- | :--- | :--- |
| **Day 1-2** | **数据集准备与噪声生成：**<br>1. 下载 ABC dataset和其他dataset（待定） 的子集（或从 Point2Surf 提供的链接下载处理好的点云）。<br>2. 编写点云数据预处理脚本，实现对输入点云随机添加高斯噪声（位置噪声）和法向扰动（噪声测试）。 | **评测模块与通用接口：**<br>1. 配置PyTorch 运行环境。<br>2. 创建共享的 GitHub 仓库，确定统一的输入（`.ply` 格式，带 `x,y,z,nx,ny,nz`）与输出（`.obj` 或 `.ply` 网格）格式。<br>3. 基于 `point-cloud-utils` (pcu) 编写通用的几何精度评测模块（计算 Chamfer 距离、Hausdorff 距离等）。 |
| **Day 3-5** | **经典方法实现：**<br>1. 利用 Open3D 快速实现 PSR 和 SPSR，确保能够正确读取准备的数据并输出重建网格。<br>2. 封装成统一的 Python 接口。 | **DGP 方法实现：**<br>编写DGP代码，编写适配脚本，使 DGP 能读取统一格式的输入点云，并在单张样本上成功跑通优化流程，输出网格。 |
| **Day 6-7** | **辅助支持与可视化准备：**<br>在本地配置可视化工具（如 MeshLab、CloudCompare 或 Polyscope），准备用于渲染对比图。 | **Stochastic PSR & NS-SPSR 实现：**<br>编写 Stochastic PSR 和 NS-SPSR 的代码。输出网格。

#### **第二周：批量评测、定性/定量分析与报告撰写**

| 时间 | Li | Zhu |
| :--- | :--- | :--- |
| **Day 8-10** | **批量生成与运行：**<br>1. 在不同噪声水平、不同点云密度（测试 Scalability 鲁棒性）的设置下，批量运行五种方法，并将结果保存在指定目录。 | **批量评测与数据收集：**<br>1. 运行写好的评测脚本，计算并记录每种方法在不同测试集上的 Chamfer 距离、Hausdorff 距离。<br>2. 统计各方法的运行时间（Time）和显存/内存消耗（Memory）。 |
| **Day 11-12** | **定性对比与渲染：**<br>1. 选择有代表性（如含尖锐特征、孔洞、重度噪声）的物体，进行多视角网格渲染对比。<br>2. 将不确定性方法（Stochastic PSR, NS-SPSR）输出的方差/不确定性场可视化。 | **数据整合与图标绘制：**<br>1. 整理定量结果，制作对比表格（Table）。<br>2. 绘制折线图或柱状图（例如：Chamfer 距离随噪声增加的变化曲线，重建时间随点数增加的规模曲线）。 |
| **Day 13-14** | **共同撰写：**<br>撰写 Final Report。重点讨论优化型方法（DGP）和经典/随机泊松方法（PSR/SPSR）的折中（Trade-offs）——包括重建时间、噪声容忍度、是否需要训练等

---

### 二、 代码目录结构

```text
3d-reconstruction-benchmark/
│
├── data/
│   ├── raw/                 # 原始 ABC/ShapeNet 干净点云
│   └── processed/           # 添加了不同标准差噪声、或下采样后的点云
│
├── src/
│   ├── dataset.py           # Li：数据加载与加噪脚本
│   ├── evaluators.py        # Zhu：几何指标计算（Chamfer, Hausdorff）
│   ├── utils.py             # 通用IO，运行耗时、显存监视器
│   │
│   ├── wrappers/            # 算法封装层，继承自相同的基类
│   │   ├── __init__.py
│   │   ├── base_wrapper.py  # 基础抽象类
│   │   ├── psr_spsr.py      # Li：Open3D PSR/SPSR
│   │   ├── dgp_wrapper.py   # Zhu：DGP wrapper
│   │   ├── stoc_psr.py      # Zhu：Stochastic PSR wrapper
│   │   └── ns_spsr.py       # Zhu：NS-SPSR wrapper
│   │
│   └── run_benchmark.py     # 统一评测入口脚本
│
├── results/                 # 存放输出
│   ├── meshes/              # 每一个算法输出的 .ply / .obj 网格
│   └── metrics.csv          # 汇总的评测数据
│
├── requirements.txt
└── README.md
```

---

### 三、 代码流程与核心实现(可以再看)

#### 1. 统一的算法基类 (`src/wrappers/base_wrapper.py`)
定义一个统一的接口，使评测脚本能够以相同的方式调用不同算法。

```python
from abc import ABC, abstractmethod
import numpy as np

class BaseReconstructor(ABC):
    @abstractmethod
    def reconstruct(self, points: np.ndarray, normals: np.ndarray, **kwargs):
        """
        Args:
            points: (N, 3) 形状的 NumPy 数组，表示点坐标
            normals: (N, 3) 形状的 NumPy 数组，表示法向量
        Returns:
            vertices: (M, 3) 重建网格的顶点
            faces: (F, 3) 重建网格的面片索引
        """
        pass
```

#### 2. Open3D SPSR 实现示例 (`src/wrappers/psr_spsr.py`)

```python
import open3d as o3d
import numpy as np
from .base_wrapper import BaseReconstructor

class SPSRReconstructor(BaseReconstructor):
    def __init__(self, depth=8, scale=1.1, linear_fit=False):
        self.depth = depth
        self.scale = scale
        self.linear_fit = linear_fit

    def reconstruct(self, points, normals, **kwargs):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        
        # 运行 Screened Poisson 表面重建
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=self.depth, scale=self.scale, linear_fit=self.linear_fit
        )
        
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        return vertices, faces
```

---
