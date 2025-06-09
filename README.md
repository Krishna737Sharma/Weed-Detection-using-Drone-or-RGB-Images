# Weed Detection using Drone or RGB Images

This project implements a semantic segmentation model to detect weeds and crops in RGB images, primarily using drone-captured data from the CWFID dataset. The model classifies pixels into three classes: background, crops, and weeds, using PyTorch and Hugging Face Transformers. It includes data preprocessing, model training, evaluation, and visualization, optimized for performance with precomputed masks to address slow image preparation.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Preprocessing](#preprocessing)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The project aims to develop an efficient weed detection system for precision agriculture. Key components:
- **Input**: RGB images and grayscale PNG masks (white background=255, black objects=0) to identify crop/weed regions.
- **Annotations**: YAML files with polygon coordinates and class labels (crop=1, weed=2, background=0).
- **Model**: Segformer (or optional CNN) for semantic segmentation with three classes.
- **Optimization**: Precomputed class-labeled masks to speed up data loading.

Features:
- Robust handling of invalid YAML annotations (e.g., skipping malformed entries in `009_annotation.yaml`, `018_annotation.yaml`).
- Support for GPU/CPU training.
- Metrics: Accuracy, IoU, mAP, SSIM, PSNR.
- Visualization: Predictions (background: black, crop: green, weed: red).

## Dataset
The **CWFID dataset (v3.0.1)** is used, containing:
- **Images**: 60 RGB images (`data/dataset-1.0/images/001_image.png` to `060_image.png`), assumed 1280x960, resized to 512x512.
- **Masks**: Grayscale PNGs (`data/dataset-1.0/masks/001_mask.png` to `060_mask.png`):
  - White (255): Background.
  - Black (0): Crops and weeds.
- **Annotations**: YAML files (`data/dataset-1.0/annotations/001_annotation.yaml` to `060_annotation.yaml`) with polygon coordinates and labels (crop/weed).
- **Split**: `data/dataset-1.0/train_test_split.yaml` (~40 train, ~20 test samples).

**Download**:
- Get the dataset from [CWFID GitHub](https://github.com/cwfid/dataset).
- Place in `data/` folder:
```
data/dataset-1.0/
├── images/
├── masks/
├── annotations/
└── train_test_split.yaml
```

## Requirements
Dependencies are listed in `requirements.txt`:

```plaintext
timm==0.9.12
transformers==4.38.2
torch==2.2.1
torchvision==0.17.1
openmim==0.3.9
mmsegmentation==1.2.2
opencv-python-headless==4.9.0.80
numpy==1.26.3
tqdm==4.66.1
pandas==2.1.4
scikit-learn==1.4.0
albumentations==1.3.1
torchsummary==1.5.1
torchmetrics>=0.23.0
pycocotools>=2.0.8
pynvml>=11.5.0
matplotlib>=3.2
psutil>=5.9.0
scikit-image>=0.24.0
fvcore>=0.1.5
```

## Installation

### Clone the Repository:
```bash
git clone https://github.com/Krishna737Sharma/Weed-Detection-using-Drone-or-RGB-Images.git
cd Weed-Detection-using-Drone-or-RGB-Images
```

### Set Up a Virtual Environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
```

### Install Dependencies:
```bash
pip install -r requirements.txt
```

### Verify PyTorch with CUDA (if GPU available):
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'No GPU')"
```

If CUDA is not available, install the CUDA-enabled PyTorch:
```bash
pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu118
```

### Download the CWFID Dataset:
Follow the Dataset section to download and place in `data/dataset-1.0/`.

## Project Structure
```
weed-detection-project/
├── data/
│   └── dataset-1.0/
│       ├── images/
│       ├── masks/
│       ├── annotations/
│       ├── preprocessed/
│       │   └── masks/
│       └── train_test_split.yaml
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # Data loading and preprocessing
│   ├── model.py           # Model definitions (Segformer, CNN)
│   ├── train.py           # Training loop
│   ├── evaluate.py        # Model evaluation and visualization
│   └── utils.py           # Utility functions (e.g., model summary)
├── main.py                # Main script for training and evaluation
├── preprocess_cwfid.py    # Script to precompute class-labeled masks
├── requirements.txt       # Dependencies
├── model_summary.txt      # Model architecture summary
├── prediction_results.png # Visualization of predictions
├── weed_detector_model.pth # Trained model weights
└── README.md              # Project documentation
```

## Usage

### Preprocessing
To optimize data loading, precompute class-labeled masks (0: background, 1: crop, 2: weed) from YAML annotations:

```bash
python preprocess_cwfid.py
```

**Output**: Masks saved in `data/dataset-1.0/preprocessed/masks/` (e.g., `001_class_mask.png`).
**Time**: ~1-2 minutes for 60 images.

**Verification**:
```python
import cv2
import matplotlib.pyplot as plt
mask = cv2.imread('data/dataset-1.0/preprocessed/masks/001_class_mask.png', cv2.IMREAD_GRAYSCALE)
plt.imshow(mask, cmap='viridis', vmin=0, vmax=2)
plt.colorbar()
plt.show()
print("Unique values:", np.unique(mask))  # Should be [0, 1, 2]
```

### Training
Run the main script to train the model:

```bash
python main.py
```

**Steps**:
- Loads CWFID dataset (uses precomputed masks if `use_precomputed_masks=True` in `main.py`).
- Prompts for model choice (Transformer: Segformer, or CNN).
- Trains for 5 epochs (adjust in `main.py`).
- Evaluates on test set.
- Saves model (`weed_detector_model.pth`), visualizations (`prediction_results.png`), and training history.

**Options**:
Edit `main.py` to change:
- `epochs`: Increase for better accuracy (e.g., 20-50).
- `batch_size`: Reduce to 2 if memory is limited.
- `use_precomputed_masks`: Set to `True` after running `preprocess_cwfid.py`.
- Choose Transformer (y) for better performance or CNN (n) for lighter computation.

**Output**:
- Training progress with loss and IoU per epoch.
- Evaluation report with metrics (accuracy, precision, recall, F1, mAP, SSIM, PSNR).
- Visualization showing original images, ground truth, and predictions (background: black, crop: green, weed: red).

### Evaluation
Evaluation (run automatically in `main.py`) generates:

- **Metrics**: Accuracy, IoU, mAP, SSIM, PSNR, confusion matrix.
- **Visualization**: Saved as `prediction_results.png`.
- **Report**: Printed to console and saved via `utils.save_training_report`.

To evaluate a trained model separately:

```python
from src.evaluate import ModelEvaluator
from src.model import EfficientWeedDetector
from src.data_loader import CWFIDDataLoader
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EfficientWeedDetector(num_classes=3).to(device)
model.load_state_dict(torch.load('weed_detector_model.pth'))
loader = CWFIDDataLoader('data/dataset-1.0', use_precomputed_masks=True)
_, _, X_test, y_test = loader.load_split_data()
evaluator = ModelEvaluator(model, X_test, y_test, device=device)
report = evaluator.generate_metrics_report()
evaluator.visualize_results(num_samples=5)
print(report)
```

## Troubleshooting

### YAML Parsing Errors
**Issue**: Warnings like `Invalid coordinates in data/dataset-1.0/annotations/009_annotation.yaml for weed: x=457.0, y=650.0`.
**Cause**: YAML files (`009_annotation.yaml`, `018_annotation.yaml`) contain single floats instead of coordinate lists.
**Fix**:
- Inspect problematic files:
```python
import yaml
for idx in [9, 18]:
    with open(f'data/dataset-1.0/annotations/{idx:03d}_annotation.yaml', 'r') as f:
        ann = yaml.safe_load(f)
        print(f"Index {idx}:", ann)
```
- Correct invalid entries (e.g., change `x: 457.0` to `x: [457.0, 458.0, 457.0]` for a small polygon) or remove them.
- Current `data_loader.py` skips invalid entries, allowing training to proceed.

### Slow Data Loading
**Issue**: Image preparation is slow without precomputed masks.
**Fix**:
- Run `preprocess_cwfid.py` and set `use_precomputed_masks=True` in `main.py`.
- Test loading speed:
```python
import time
from src.data_loader import CWFIDDataLoader
loader = CWFIDDataLoader('data/dataset-1.0', use_precomputed_masks=True)
start = time.time()
img, mask = loader.load_image_mask_pair(1)
print(f"Time: {time.time() - start:.2f} seconds")
```

### Training Freezes or Crashes
**Issue**: Training doesn't start or freezes, possibly due to CPU usage or memory constraints.
**Fix**:
- Check GPU availability:
```python
import torch
print(torch.cuda.is_available())
```
- Reduce `batch_size` to 2 in `main.py` and `train.py`.
- Set `num_workers=0` in `main.py`:
```python
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
```
- Monitor memory:
```bash
free -h
```

### DataLoader Worker Warning
**Issue**: Warning about excessive worker processes (`num_workers=4`).
**Fix**: Set `num_workers=2` in `main.py` and `train.py` to match system recommendations.

### Incorrect Mask Values
**Issue**: Training fails with errors like `Target 2 is out of bounds`.
**Fix**:
- Verify mask values:
```python
from src.data_loader import CWFIDDataLoader
loader = CWFIDDataLoader('data/dataset-1.0')
_, mask = loader.load_image_mask_pair(1)
print("Unique mask values:", np.unique(mask))
```
- Ensure values are [0, 1, 2]. If not, check `preprocess_cwfid.py` or YAML files.

## Contributing

1. **Fork the Repository**:
```bash
git clone https://github.com/<your-username>/Weed-Detection-using-Drone-or-RGB-Images.git
```

2. **Create a Feature Branch**:
```bash
git checkout -b feature/<your-feature>
```

3. **Commit Changes**:
```bash
git add .
git commit -m "Added <your-feature>"
```

4. **Push and Create a Pull Request**:
```bash
git push origin feature/<your-feature>
```
Open a pull request on GitHub.

5. **Exclude Data Files**:
Ensure `data/dataset-1.0/` is in `.gitignore`:
```plaintext
data/dataset-1.0/
```

## Results

### Training Performance
The model was successfully trained for 10 epochs with the following performance progression:

| Epoch | Train Loss | Train IoU | Val Loss | Val IoU |
|-------|------------|-----------|----------|---------|
| 1     | 0.8558     | 0.4391    | 0.9865   | 0.4399  |
| 2     | 0.6268     | 0.5308    | 0.7670   | 0.5274  |
| 3     | 0.5441     | 0.5909    | 0.6008   | 0.5853  |
| 4     | 0.4768     | 0.6539    | 0.4638   | 0.6417  |
| 5     | 0.4272     | 0.6912    | 0.3690   | 0.6873  |
| 6     | 0.3850     | 0.7028    | 0.3725   | 0.6980  |
| 7     | 0.3411     | 0.7262    | 0.3040   | 0.7266  |
| 8     | 0.3233     | 0.7408    | 0.2690   | 0.7296  |
| 9     | 0.2827     | 0.7645    | 0.2623   | 0.7417  |
| 10    | 0.2753     | 0.7441    | 0.2555   | 0.7422  |

### Model Evaluation Metrics

**Model Complexity:**
- Parameters: 3,714,915

**Accuracy Metrics:**
- **Accuracy**: 93.60%
- **Precision**: 93.93%
- **Recall**: 93.60%
- **F1-Score**: 93.73%
- **Mean Average Precision (mAP)**: 89.41%
- **SSIM**: 89.44%
- **PSNR**: 14.05 dB
- **MSE**: 0.1859

**Performance Metrics:**
- **Inference Time**: 1,601.37 ± 19.50 ms
- **Throughput**: 0.62 FPS

**Confusion Matrix:**
```
              Predicted
Actual    Background    Crop      Weed
Background  4,231,780   40,371   147,796
Crop           16,118  177,141    43,950
Weed           75,851   28,112   743,905
```
![Image](https://github.com/user-attachments/assets/1b8330d7-39d8-4392-b2c1-b43a8c113b68)

![Image](https://github.com/user-attachments/assets/8835bc3b-10d6-431b-b16b-49a1d6657161)

![Image](https://github.com/user-attachments/assets/2d3c3cef-66e7-49f9-85fe-3e1afb2916d4)

### Key Achievements
- Successfully achieved **93.60% accuracy** on the test set
- **IoU improved from 0.44 to 0.74** during training
- Model effectively distinguishes between crops and weeds with high precision
- Robust performance despite invalid YAML annotations in the dataset
- Generated comprehensive visualizations and training reports

### Output Files
The training process generates several output files:
- `weed_detector_model.pth` - Trained model weights
- `prediction_results.png` - Visualization of model predictions
- `training_history.png` - Training/validation loss and IoU plots
- `training_report.txt` - Detailed evaluation report
- `model_summary.txt` - Model architecture summary

## License
This project is licensed under the MIT License. See LICENSE for details.
