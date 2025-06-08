import torch
import numpy as np
import time
import psutil
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from src.data_loader import WeedDataset
from torch.utils.data import DataLoader
import os
from datetime import datetime
from PIL import Image  # Added import for Image

try:
    from fvcore.nn import FlopCountAnalysis
    FLOPS_AVAILABLE = True
except ImportError:
    FLOPS_AVAILABLE = False
    print("fvcore not available, FLOPs calculation disabled")

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_MONITORING = True
except ImportError:
    GPU_MONITORING = False
    print("pynvml not available, GPU monitoring disabled")

class ModelEvaluator:
    def __init__(self, model, X_test, y_test, device='cuda'):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model.eval()
        
        if GPU_MONITORING and self.device == 'cuda':
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        # Define transform for visualization
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def evaluate_accuracy(self):
        """Evaluate model accuracy"""
        test_dataset = WeedDataset(self.X_test, self.y_test, transform=self.transform)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, masks in test_loader:
                images = images.to(self.device)
                outputs = self.model(pixel_values=images) if hasattr(self.model, 'model') else self.model(images)
                
                # Handle different output formats
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
                # Resize if needed
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = torch.nn.functional.interpolate(
                        logits, size=masks.shape[-2:], mode='bilinear', align_corners=False
                    )
                
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds.flatten())
                all_labels.extend(masks.cpu().numpy().flatten())
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def measure_inference_time(self, num_samples=100):
        """Measure inference time"""
        # Create a sample input with proper normalization
        sample_input = torch.randn(1, 3, 512, 512)
        sample_input = self.transform(sample_input).to(self.device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(pixel_values=sample_input) if hasattr(self.model, 'model') else self.model(sample_input)
        
        # Measure
        times = []
        for _ in range(num_samples):
            start_time = time.time()
            with torch.no_grad():
                _ = self.model(pixel_values=sample_input) if hasattr(self.model, 'model') else self.model(sample_input)
            if self.device == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        return np.mean(times), np.std(times)
    
    def count_parameters(self):
        """Count model parameters"""
        return sum(p.numel() for p in self.model.parameters())
    
    def visualize_results(self, num_samples=5, file_path='prediction_results.png'):
        """Visualize some predictions"""
        if len(self.X_test) == 0:
            print("No test data available for visualization")
            return
            
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples*5))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(min(num_samples, len(self.X_test))):
            # Original image (denormalized for visualization)
            img = self.X_test[i].astype(float) / 255.0  # Assume X_test is in [0, 255]
            img_tensor = self.transform(Image.fromarray(self.X_test[i])).unsqueeze(0).to(self.device)
            
            # Denormalize for display
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            display_img = img * std + mean
            display_img = np.clip(display_img, 0, 1)
            
            axes[i, 0].imshow(display_img)
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
            
            # Ground truth
            axes[i, 1].imshow(self.y_test[i], cmap='viridis', vmin=0, vmax=2)
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            # Prediction
            with torch.no_grad():
                output = self.model(pixel_values=img_tensor) if hasattr(self.model, 'model') else self.model(img_tensor)
                logits = output.logits if hasattr(output, 'logits') else output
                if logits.shape[-2:] != (512, 512):
                    logits = torch.nn.functional.interpolate(
                        logits, size=(512, 512), mode='bilinear', align_corners=False
                    )
                pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
            
            axes[i, 2].imshow(pred, cmap='viridis', vmin=0, vmax=2)
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
        plt.savefig(file_path, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        print(f"Visualization saved to {file_path}")
    
    def generate_metrics_report(self):
        """Generate comprehensive evaluation report"""
        print("Evaluating model performance...")
        
        # Accuracy metrics
        accuracy_metrics = self.evaluate_accuracy()
        
        # Performance metrics
        avg_time, time_std = self.measure_inference_time()
        
        # Model complexity
        num_params = self.count_parameters()
        
        # GPU memory usage
        gpu_memory = "N/A"
        if GPU_MONITORING and self.device == 'cuda':
            try:
                gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(self.handle).used / (1024**2)
                gpu_memory = f"{gpu_memory:.1f} MB"
            except:
                gpu_memory = "Error retrieving GPU memory"
        
        report = f"""
MODEL EVALUATION REPORT
========================
Model Complexity:
-----------------
Parameters: {num_params:,}

Accuracy Metrics:
-----------------
Accuracy: {accuracy_metrics['accuracy']:.4f}
Precision: {accuracy_metrics['precision']:.4f}
Recall: {accuracy_metrics['recall']:.4f}
F1-Score: {accuracy_metrics['f1_score']:.4f}

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
        """Calculate FLOPs if fvcore is available"""
        if not FLOPS_AVAILABLE:
            return 0
        
        with torch.no_grad():
            flops = FlopCountAnalysis(self.model, self.test_input)
            return flops.total() / 1e9  # Convert to GFLOPs
    
    def evaluate(self):
        """Basic efficiency evaluation"""
        results = {}
        results['parameters'] = sum(p.numel() for p in self.model.parameters())
        results['flops'] = self.calculate_flops()
        
        # Timing
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