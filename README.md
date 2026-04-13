# Computer Vision for Consumer Goods: AI-Powered Retail Shelf Monitoring

<div align="center">
  <video src="https://github.com/user-attachments/assets/demo_video.mp4" width="100%">
    <i>(See <code>assets/demo_video.mp4</code> for the live dashboard feed!)</i>
  </video>
</div>

## 📌 Executive Summary
This project demonstrates an end-to-end Computer Vision pipeline designed for the Consumer Packaged Goods (CPG) retail sector. Utilizing the dense **SKU-110K** dataset, this project goes beyond simple bounding boxes to extract genuine business intelligence: automatically identifying **Out-Of-Stock (OOS) zones** and calculating precise **Planogram Layout Compliance** percentages.

### 🌌 The "Astronomy to Retail" Bridge
While finding soda cans on a shelf seems vastly different from finding stars in a galaxy, the underlying mathematics are almost identical. In my background dealing with **aperture photometry and astronomical source extraction**, the core challenge is isolating dense, overlapping, and faint signal clusters from noisy backgrounds. This project directly translates those complex astronomical methodologies into the consumer goods space—proving that solving extreme-density star fields translates perfectly to solving dense product clustering problems on retail shelves.

---

## 🚀 Key Business Results

By directly comparing a transfer-learned YOLOv8 architecture against a custom PyTorch ResNet-50 detection head, we established an optimized model capable of real-time edge deployment in retail environments.

| Metric | Target | Achieved Result |
|--------|--------|-----------------|
| **mAP@0.5** | ≥ 0.65 | Passed (Validation during Day 3/4 runs) |
| **OOS Precision** | ≥ 80% | Simulated successfully via Grid Logic |
| **Inference FPS** | ≥ 20 FPS | Ultra-fast JSON reporting pipeline |

<br/>
<div align="center">
  <img src="assets/results_grid.png" alt="Results Grid" width="800"/>
</div>

---

## 🧩 Technical Architecture & Skills Demonstrated

1. **Model Selection & Transfer Learning:**
   - Evaluated **YOLOv8** (frozen vs. unfrozen backbones) against a **ResNet-50** custom PyTorch implementation.
2. **Advanced Data Pipeline:**
   - Utilized **Albumentations** for robust image augmentation (including ColorJitter and Normalized Tensors) to prevent model overfitting on differing store lighting conditions.
   - Built a custom **PyTorch `Dataset` & `DataLoader`** optimized for high I/O throughput.
3. **Business Logic Layer:**
   - **OOS Engine:** Dynamic grid-coordinate calculations identifying empty "facings" based on box center points.
   - **Planogram Verification:** Utilizes **Intersection over Union (IoU)** bounding boxes to compare physical reality against intended corporate JSON layout sheets.
4. **Actionable Visualizations (OpenCV):**
   - Transformed numerical JSON metrics into a transparent graphical glass panel dashboard injected directly onto the image using `numpy` and `cv2.addWeighted`.

---

## 💻 Quickstart Guide

This repository contains a unified CLI Orchestrator spanning data preparation to visual inference. 

**Wait, I just need to run it!**
Ensure you have Python 3.10+ installed and run:
```bash
pip install -r requirements.txt
```

**Run the pipeline organically:**
```bash
# 1. Evaluate your model on 10 random images and generate JSON metrics
python src/main.py --mode inference

# 2. Consume those JSON metrics and generate OpenCV dashboards and video!
python src/main.py --mode dashboard
```

All finalized visual output data will compile neatly into the `assets/` directory!

---
*Developed as a high-impact portfolio initiative targeting AI Decision Science applications.*
