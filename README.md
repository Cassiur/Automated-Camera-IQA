# Automated-Camera-IQA

> **车载相机 ISP 链仿真与画质自动评估工具**  
> Automotive Camera ISP Simulation & Image Quality Assessment Toolkit  
> `v0.1.0` | Python ≥ 3.10 | 2026.01

---

## 项目概览

本项目基于 Python / NumPy 构建车载视觉 **ISP（图像信号处理器）流水线全链路仿真**，并集成符合 **ISO 12233** 标准的 SFR/MTF 画质评估引擎与多相机 **Boresight 对齐校核**模块，支持 CLI + YAML 配置驱动的批量评测与版本回归对比。

### 架构总览

```
RAW 图像
  │
  ▼ meta = {bayer_pattern, bit_depth}
┌────────────┐
│    BLC     │  逐通道黑电平减法 + clip
└─────┬──────┘
      │
      ▼
┌────────────┐
│  Demosaic  │  Malvar-He-Cutler 5×5 卷积 (Bayer [H,W] → RGB [H,W,3])
└─────┬──────┘
      │
      ▼
┌────────────┐
│    AWB     │  Gray-World / Perfect Reflector / Manual
└─────┬──────┘
      │
      ▼
┌────────────┐
│    CCM     │  3×3 色彩校正矩阵 + 色域裁剪
└─────┬──────┘
      │
      ▼
  RGB 图像 (.png / .tiff)
      │
      ├──► SFRAnalyzer  ──► MTF50 (ISO 12233) → JSON / CSV 报告
      │     ESF → LSF → FFT → MTF → MTF50
      │
      └──► BoresightChecker ──► 重投影误差 + Baseline 偏差 → 对齐报告
            外参 + 世界点 → PASS / FAIL
```

---

## 快速上手

### 安装

```bash
git clone https://github.com/yourname/Automated-Camera-IQA.git
cd Automated-Camera-IQA
pip install -e ".[dev]"
```

### CLI 使用

```bash
# 1. ISP 流水线（RAW → RGB）
iqa isp \
  --input  data/raw/*.raw \
  --output results/isp/ \
  --config configs/default_isp.yaml \
  --bayer  RGGB \
  --bit-depth 12 \
  --debug                  # 保存每个 stage 中间结果

# 2. SFR / MTF 批量评测（ISO 12233）
iqa sfr \
  --input  data/sfr/*.png \
  --roi    "100,50,400,350" \
  --output results/sfr/ \
  --format both \
  --plot                   # 生成 MTF 曲线图

# 3. Boresight 对齐校核
iqa boresight \
  --config configs/boresight.yaml \
  --output results/boresight/
```

### Python API

```python
from iqa import ISPPipeline, SFRAnalyzer, BoresightChecker
from iqa.utils.config_loader import load_isp_config, SFRConfig
import numpy as np

# --- ISP ---
cfg = load_isp_config("configs/default_isp.yaml")
pipeline = ISPPipeline(cfg)
result = pipeline.run(raw_array, meta={"bayer_pattern": "RGGB", "bit_depth": 12})
print(f"ISP done in {result.elapsed_ms:.1f} ms, output shape: {result.image.shape}")

# --- SFR ---
analyzer = SFRAnalyzer(SFRConfig(oversample_factor=4, pixel_size_um=3.0))
sfr_result = analyzer.analyze(roi_image)
print(f"MTF50 = {sfr_result.mtf50_cy_px:.4f} cy/px  "
      f"({sfr_result.mtf50_lp_mm:.1f} lp/mm)")

# --- Boresight ---
from iqa.calibration.extrinsic import load_extrinsic_npz
cameras = [load_extrinsic_npz("front", "calib/front.npz", "calib/front_K.npz")]
checker = BoresightChecker(cameras, rms_threshold=0.5)
report  = checker.run(world_points, observations)
print(report.to_text())
```

---

## 核心模块

### 1. ISP 流水线

| Stage | 算法 | 参考 |
|-------|------|------|
| **BLC** | 逐通道黑电平减法，支持 RGGB/BGGR/GRBG/GBRG | — |
| **Demosaic** | Malvar-He-Cutler 2004，PSNR ≈ +5 dB vs 双线性 | [ICASSP 2004] |
| **AWB** | Gray-World + Perfect Reflector + Manual | — |
| **CCM** | 3×3 线性变换，色域裁剪 | IEC 61966-2-1 |

### 2. SFR / MTF 评测引擎

按 ISO 12233 标准实现：

```
倾斜刃边 ROI
  → 边缘定位（线性回归，角度验证 4°–15°）
  → ESF 超采样（默认 4×，亚像素 bin 中位数）
  → LSF（Savitzky-Golay 平滑 + np.gradient）
  → Hamming 窗 FFT → 归一化 MTF 曲线
  → MTF50（线性插值）
```

输出：`mtf50_cy_px`、`mtf50_lp_mm`（需 pixel_size_um 配置）、MTF 曲线数据。

### 3. Boresight 对齐校核

- **重投影误差**：`P_cam = R @ P_w + t` → 针孔投影 → 逐点 RMS
- **Baseline 一致性**：`T_ij = T_j @ inv(T_i)`，提取平移距离与旋转角
- 与设计标称值比较，输出 PASS / FAIL 及偏差量

---

## 运行测试

```bash
pytest tests/ -v --tb=short
# 含覆盖率
pytest tests/ -v --cov=iqa --cov-report=term-missing
```

---

## 项目结构

```
Automated-Camera-IQA/
├── pyproject.toml          # PEP 517 构建，CLI entry point
├── configs/                # YAML 配置模板
│   ├── default_isp.yaml
│   ├── sfr_batch.yaml
│   └── boresight.yaml
├── src/iqa/
│   ├── pipeline/           # ISP stages + 编排器
│   ├── metrics/            # SFR / MTF + 报告
│   ├── calibration/        # Boresight + 外参工具
│   ├── cli/                # Click CLI
│   └── utils/              # IO / Config / Logger
└── tests/                  # pytest 测试套件
```

---

## 依赖

| 包 | 用途 |
|----|------|
| `numpy` | 核心数值计算 |
| `scipy` | Savitzky-Golay、FFT 工具 |
| `opencv-python` | 图像 IO、卷积、Demosaic 备用 |
| `pyyaml` | 配置文件解析 |
| `click` | CLI 框架 |
| `matplotlib` *(dev)* | MTF 曲线可视化 |
| `pytest` *(dev)* | 单元测试 |

---

## License

MIT © 2026
