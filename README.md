# Weed Detection using Drone or RGB Images

This project implements a deep learning-based weed detection system using semantic segmentation on the CWFID dataset. It supports two models: a transformer-based Segformer and a CNN-based UNet, designed to identify soil (Class 0), crop (Class 1), and weed (Class 2) pixels in RGB images.

## Project Structure
- `main.py`: Entry point for training and evaluation.
- `src/train.py`: Training logic with weighted loss and learning rate scheduling.
- `src/evaluate.py`: Evaluation metrics computation, including per-class Average Precision.
- `src/data_loader.py`: Dataset loading with data augmentation.
- `src/model.py`: Model definitions for Segformer and UNet.
- `src/utils.py`: Utility functions.
- `src/check_dataset.py`: Script to verify dataset integrity.
- `data/dataset-1.0/`: CWFID dataset (images, masks, train-test split).
- `requirements.txt`: Project dependencies.
- `README.md`: Project documentation.
- Outputs: `model_summary.txt`, `prediction_results.png`, `training_history.png`, `training_report.txt`, `weed_detector_model.pth`, `best_model.pth`.

## Dataset
- **CWFID Dataset**: Contains 40 training samples and 21 test samples.
- **Classes**: Soil (0), Crop (1), Weed (2).
- **Image Size**: 512x512 pixels (RGB).

## Models
- **Segformer**: A transformer-based model (`nvidia/mit-b0`) leveraging attention mechanisms for robust feature extraction, suitable for high-accuracy segmentation.
- **UNet**: A CNN-based model designed for efficient and fast inference, ideal for real-time applications.
- **Why Chosen**:
  - **Segformer**: Selected for its state-of-the-art performance in semantic segmentation, capturing long-range dependencies in drone imagery.
  - **UNet**: Chosen for its lightweight architecture and faster inference, balancing accuracy and computational efficiency.

## Current Results (5 Epochs, Segformer)
Training was conducted for 5 epochs on a CPU, with a batch size of 4 and a learning rate of 1e-4. Data augmentation (random rotation, flip, brightness/contrast) was applied.

### Training Metrics
- **Epoch 1**: Train Loss: 0.8889, Train IoU: 0.3763, Val Loss: 0.9744, Val IoU: 0.4543
- **Epoch 2**: Train Loss: 0.6883, Train IoU: 0.4675, Val Loss: 0.8428, Val IoU: 0.5078
- **Epoch 3**: Train Loss: 0.5554, Train IoU: 0.5181, Val Loss: 0.6481, Val IoU: 0.5343
- **Epoch 4**: Train Loss: 0.4459, Train IoU: 0.5476, Val Loss: 0.4041, Val IoU: 0.5691
- **Epoch 5**: Train Loss: 0.3814, Train IoU: 0.5580, Val Loss: 0.2698, Val IoU: 0.5772

### Evaluation Metrics
- **Accuracy**: 0.9765
- **Precision**: 0.9783
- **Recall**: 0.9765
- **F1-Score**: 0.9772
- **Mean Average Precision (mAP)**: 0.9633
- **Structural Similarity Index (SSIM)**: 0.9118
- **Peak Signal-to-Noise Ratio (PSNR)**: 16.9981
- **Mean Squared Error (MSE)**: 0.0935
- **Confusion Matrix**:
[[ 394151      10   39488]
[      0       0       0]
[  88944     868 4981563]]
- **Class 0 (Soil)**: 394151 true positives, but 39488 misclassified as Class 2.
- **Class 1 (Crop)**: No pixels predicted or present, indicating either absence in the test set or model failure to detect.
- **Class 2 (Weed)**: 4981563 true positives, but 88944 misclassified as Class 0.

- **Model Complexity**: 3,714,915 parameters
- **Inference Time**: 1382.38 ± 17.02 ms
- **Throughput**: 0.72 FPS (on CPU)

### Observations
- **Improved Performance**: Compared to 1 epoch (Train IoU: 0.3763, Val IoU: 0.4543; PSNR: 11.0331; MSE: 0.3453), 5 epochs significantly improved IoU, PSNR, and MSE.
- **Class Imbalance**: High Accuracy, Precision, Recall, and F1-Score may be inflated due to Class 2 dominance and Class 1 absence.
- **Limitations**: 
- No Class 1 pixels in the test set or undetected by the model.
- Slow inference (0.72 FPS) due to Segformer on CPU.
- IoU (0.5772) and false negatives for Class 2 (88944) indicate room for improvement.

## Optimization Plan
To address class imbalance, Class 1 absence, and performance limitations:
- **Verify Class Distribution**: Run `check_class_distribution.py` to confirm Class 1 presence in train/test sets.
- **Weighted Loss**: Implement class-weighted CrossEntropyLoss (in `train.py`) to penalize minority classes.
- **Increase Epochs**: Train for 25 epochs to improve IoU and reduce false negatives.
- **Test UNet**: Evaluate CNN model for faster inference (>2 FPS).
- **Per-Class AP**: Report Average Precision per class (in `evaluate.py`) to assess Class 1 performance.
- **GPU Acceleration**: If available, use GPU to reduce training/inference time.

## Installation
```bash
git clone https://github.com/Krishna737Sharma/Weed-Detection-using-Drone-or-RGB-Images
cd weed-detection-project
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
