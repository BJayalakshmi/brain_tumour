# 🧠 MRI-Based Brain Tumor Detection Using YOLOv11n (C3K2)

This project focuses on **automated brain tumor detection from MRI images** using an enhanced **YOLOv11n** object detection architecture that integrates the **C3K2 module** for efficient feature learning and faster convergence.  
The goal is to accurately **detect and localize tumor regions** within brain MRI scans from the **Br35H 2020 dataset**.

---

## 🚀 Project Overview

Brain tumor diagnosis through MRI scans is crucial but often time-consuming and dependent on radiologist expertise.  
This project leverages **deep learning and computer vision** to automate this process with real-time detection capabilities.

The lightweight **YOLOv11n (nano)** model ensures faster inference while maintaining high accuracy, making it suitable for **deployment on edge devices** or medical imaging systems.

---

## 🧩 Key Features

- ⚙️ **YOLOv11n Architecture** — Optimized for speed and efficiency.  
- 🔁 **C3K2 Module** — Replaces the C2f block to improve feature reuse and gradient flow.  
- 🧠 **Tumor Localization** — Identifies and marks tumor regions in MRI scans.  
- 📈 **Custom Dataset Training** — Fine-tuned on the *Br35H: Brain Tumor Detection 2020* dataset.  
- 💻 **Edge-Deployable** — Nano model variant designed for low-latency inference.  
- 🩺 **Explainable AI Ready** — Can be extended with Grad-CAM or saliency maps for interpretability.

---

## 📊 Dataset: Br35H 2020

**Source:** [Br35H - Brain Tumor Detection 2020 Dataset (Kaggle)](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection)  

| Property | Description |
|-----------|-------------|
| **Type** | MRI Images (yes/no tumor) |
| **Classes** | `tumor`, `no_tumor` |
| **Total Images** | ~3,000 |
| **Format** | JPEG/PNG |
| **Annotation** | Bounding boxes created using LabelImg in YOLO format |

---

## 🧰 Tech Stack

| Component | Technology |
|------------|-------------|
| **Language** | Python 3.10+ |
| **Framework** | [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics) |
| **Deep Learning** | PyTorch |
| **IDE** | Visual Studio Code |
| **Visualization** | OpenCV, Matplotlib, TensorBoard |
| **Annotation Tool** | LabelImg |

---


## 🧩 Setup Instructions (VS Code)

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/Brain-Tumor-Detection-YOLOv11n.git
cd Brain-Tumor-Detection-YOLOv11n

python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
venv\Scripts\activate

pip install -r requirements.txt
```

### 2️⃣ Donwload the dataset
```bash
https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection?resource=download
```

### 3️⃣ Train the model
```bash
yolo detect train model=models/yolov11n_brain.yaml data=brain_tumor.yaml epochs=100 imgsz=640 batch=8 device=0
```

### 4️⃣ Run Interface
```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=sample.jpg show=True
```
