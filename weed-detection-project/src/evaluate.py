import torch
import numpy as np
import time
import psutil
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, average_precision_score
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from src.data_loader import WeedDataset
from torch.utils.data import DataLoader
import os
from datetime import datetime
from PIL import Image
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

try:
    from fvcore.nn import FlopCountAnalysis
    FLOPS_AVAILABLE = True
except ImportError:
    FLOPS_AVAILABLE = False
    print("fvcore not available, FLOPs calculation disabled")

GPU_MONITORING = False
try:
    import pynvml
except ImportError:
    print("pynvml not available, GPU monitoring disabled")

class ModelEvaluator:
    def __init__(self, model, X_test, y_test, device='cuda'):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model.eval()
        
        self.handle = None
        if self.device == 'cuda' and 'pynvml' in globals():
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                global GPU_MONITORING
                GPU_MONITORING = True
            except pynvml.NVMLError:
                print("Failed to initialize pynvml, GPU monitoring disabled")
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
    
    def evaluate_accuracy(self):
        """Evaluate model accuracy and additional metrics for 3 classes"""
        test_dataset = WeedDataset(self.X_test, self.y_test, transform=self.transform)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
        
        all_preds = []
        all_labels = []
        ap_scores = []
        ssim_scores = []
        psnr_scores = []
        mse_scores = []
        
        with torch.no_grad():
            for images, masks in test_loader:
                images = images.to(self.device)
                outputs = self.model(pixel_values=images) if hasattr(self.model, 'model') else self.model(images)
                
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = torch.nn.functional.interpolate(
                        logits, size=masks.shape[-2:], mode='bilinear', align_corners=False
                    )
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(masks.cpu().numpy().flatten())
                
                pred_probs = torch.softmax(logits, dim=1).cpu().numpy()
                true_masks = masks.cpu().numpy()
                for b in range(preds.shape[0]):
                    for c in range(3):  # 3 classes: background, crop, weed
                        pred_prob = pred_probs[b, c].flatten()
                        true_mask = (true_masks[b] == c).flatten()
                        if np.sum(true_mask) > 0:
                            ap = average_precision_score(true_mask, pred_prob)
                            if not np.isnan(ap):
                                ap_scores.append(ap)
                
                for i in range(preds.shape[0]):
                    pred_mask = preds[i].cpu().numpy().astype(np.uint8)
                    true_mask = masks[i].cpu().numpy().astype(np.uint8)
                    ssim_score = ssim(pred_mask, true_mask, data_range=2, channel_axis=None)
                    psnr_score = psnr(pred_mask, true_mask, data_range=2)
                    mse_score = np.mean((pred_mask - true_mask) ** 2)
                    ssim_scores.append(ssim_score)
                    psnr_scores.append(psnr_score)
                    mse_scores.append(mse_score)
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])  # 3 classes
        mAP = np.mean(ap_scores) if ap_scores else 0.0
        avg_ssim = np.mean(ssim_scores)
        avg_psnr = np.mean(psnr_scores)
        avg_mse = np.mean(mse_scores)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,
            'mAP': mAP,
            'ssim': avg_ssim,
            'psnr': avg_psnr,
            'mse': avg_mse
        }
    
    def measure_inference_time(self, num_samples=100):
        sample_input = np.random.rand(1, 512, 512, 3).astype(np.float32) * 255
        sample_input = sample_input[0]
        sample_input = self.transform(Image.fromarray(sample_input.astype(np.uint8))).unsqueeze(0).to(self.device)
        
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(pixel_values=sample_input) if hasattr(self.model, 'model') else self.model(sample_input)
        
        times = []
        for _ in range(num_samples):
            start_time = time.time()
            with torch.no_grad():
                _ = self.model(pixel_values=sample_input) if hasattr(self.model, 'model') else self.model(sample_input)
            if self.device == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            times.append((end_time - start_time) * 1000)
        
        return np.mean(times), np.std(times)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters())
    
    def visualize_results(self, num_samples=5, file_path='prediction_results.png'):
        if len(self.X_test) == 0:
            print("No test data available for visualization")
            return
            
        # Color map for visualization
        vis_color_map = {
            0: [0, 0, 0],      # Black: Background
            1: [0, 255, 0],    # Green: Crop
            2: [255, 0, 0]     # Red: Weed
        }
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples*5))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(min(num_samples, len(self.X_test))):
            img = self.X_test[i].astype(float) / 255.0
            img_tensor = self.transform(Image.fromarray(self.X_test[i])).unsqueeze(0).to(self.device)
            
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            display_img = img * std + mean
            display_img = np.clip(display_img, 0, 1)
            
            axes[i, 0].imshow(display_img)
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
            
            # Convert ground truth to RGB
            gt_mask = self.y_test[i]
            gt_rgb = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
            for cls, color in vis_color_map.items():
                gt_rgb[gt_mask == cls] = color
            axes[i, 1].imshow(gt_rgb)
            axes[i, 1].set_title('Ground Truth (Crop: Green, Weed: Red)')
            axes[i, 1].axis('off')
            
            with torch.no_grad():
                output = self.model(pixel_values=img_tensor) if hasattr(self.model, 'model') else self.model(img_tensor)
                logits = output.logits if hasattr(output, 'logits') else output
                if logits.shape[-2:] != (512, 512):
                    logits = torch.nn.functional.interpolate(
                        logits, size=(512, 512), mode='bilinear', align_corners=False
                    )
                pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
            
            # Convert prediction to RGB
            pred_rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
            for cls, color in vis_color_map.items():
                pred_rgb[pred == cls] = color
            axes[i, 2].imshow(pred_rgb)
            axes[i, 2].set_title('Prediction (Crop: Green, Weed: Red)')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to {file_path}")
    
    def generate_metrics_report(self):
        print("Evaluating model performance...")
        
        metrics = self.evaluate_accuracy()
        avg_time, time_std = self.measure_inference_time()
        num_params = self.count_parameters()
        
        gpu_memory = "N/A"
        if GPU_MONITORING and self.device == 'cuda':
            try:
                gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(self.handle).used / (1024**2)
                gpu_memory = f"{gpu_memory:.1f} MB"
            except pynvml.NVMLError:
                gpu_memory = "Error retrieving GPU memory"
        
        conf_matrix_str = "\nConfusion Matrix:\n" + str(metrics['confusion_matrix'])
        
        report = f"""
MODEL EVALUATION REPORT
========================
Model Complexity:
-----------------
Parameters: {num_params:,}

Accuracy Metrics:
-----------------
Accuracy: {metrics['accuracy']:.4f}
Precision: {metrics['precision']:.4f}
Recall: {metrics['recall']:.4f}
F1-Score: {metrics['f1_score']:.4f}
Mean Average Precision (mAP): {metrics['mAP']:.4f}
SSIM: {metrics['ssim']:.4f}
PSNR: {metrics['psnr']:.4f}
MSE: {metrics['mse']:.4f}
{conf_matrix_str}

Performance Metrics:
--------------------
Inference Time: {avg_time:.2f} ± {time_std:.2f} ms
Throughput: {1000/avg_time:.2f} FPS

System Info:
-----------
Device: {self.device}
CPU Usage: {psutil.cpu_percent():.1f}%
RAM Usage: {psutil.Process().memory_info().rss / (1024**2):.1f} MB
GPU Memory Usage: {gpu_memory}
"""
        return report

class EfficiencyEvaluator:
    def __init__(self, model, test_input, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.test_input = test_input.to(device)
        self.model.eval()
        
    def calculate_flops(self):
        if not FLOPS_AVAILABLE:
            return 0
        
        with torch.no_grad():
            flops = FlopCountAnalysis(self.model, self.test_input)
            return flops.total() / 1e9
    
    def evaluate(self):
        results = {}
        results['parameters'] = sum(p.numel() for p in self.model.parameters())
        results['flops'] = self.calculate_flops()
        
        times = []
        for _ in range(100):
            start = time.time()
            with torch.no_grad():
                _ = self.model(pixel_values=self.test_input) if hasattr(self.model, 'model') else self.model(self.test_input)
            if self.device == 'cuda':
                torch.cuda.synchronize()
            times.append(time.time() - start)
        
        results['inference_time_ms'] = np.mean(times) * 1000
        results['fps'] = 1000 / results['inference_time_ms']
        
        return results