import torch
from torchsummary import summary
from transformers import SegformerForSemanticSegmentation
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np

def print_model_summary(model, file_path='model_summary.txt', input_size=(3, 512, 512)):
    """
    Print model summary for either CNN or Transformer models with improved handling
    
    Args:
        model: The model to summarize
        file_path: Path to save summary (default: 'model_summary.txt')
        input_size: Input tensor size for CNN models (default: (3, 512, 512))
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
    
    try:
        with open(file_path, 'w') as f:
            f.write(f"Model Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*50 + "\n\n")
            
            if isinstance(model, SegformerForSemanticSegmentation) or hasattr(model, 'model'):
                # Handle transformer model summary
                f.write("Segformer Model Architecture:\n")
                config = model.model.config if hasattr(model, 'model') else model.config
                f.write(str(config) + "\n\n")
                
                # Detailed layer information
                f.write("Layer Details:\n")
                target_model = model.model if hasattr(model, 'model') else model
                for name, layer in target_model.named_children():
                    f.write(f"{name}: {layer.__class__.__name__}\n")
                
                # Parameters count
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                f.write("\nParameters:\n")
                f.write(f"Total parameters: {total_params:,}\n")
                f.write(f"Trainable parameters: {trainable_params:,}\n")
                f.write(f"Non-trainable parameters: {total_params - trainable_params:,}\n")
                
                print(f"Transformer model summary saved to {file_path}")
            else:
                # Handle CNN model summary
                try:
                    # Try using torchsummary
                    model_summary = summary(model, input_size=input_size, device='cpu', verbose=0)
                    f.write(str(model_summary))
                    
                    # Additional parameter count
                    total_params = sum(p.numel() for p in model.parameters())
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    f.write("\n\nAdditional Parameter Information:\n")
                    f.write(f"Total parameters: {total_params:,}\n")
                    f.write(f"Trainable parameters: {trainable_params:,}\n")
                    f.write(f"Non-trainable parameters: {total_params - trainable_params:,}\n")
                    
                    print(f"CNN model summary saved to {file_path}")
                except Exception as e:
                    # Fallback to simple print
                    f.write("Simplified Model Architecture:\n")
                    f.write(str(model))
                    f.write("\n\nNote: Detailed summary unavailable. Using simplified representation.\n")
                    print(f"Basic model architecture saved to {file_path}")
                    
    except Exception as e:
        print(f"Error saving model summary: {str(e)}")

def plot_training_history(history, file_path='training_history.png', show=False):
    """
    Plot training and validation metrics with improved visualization
    
    Args:
        history: Dictionary containing training history
        file_path: Path to save the plot (default: 'training_history.png')
        show: Whether to display the plot (default: False)
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
        
        plt.figure(figsize=(15, 6))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], 'b-', label='Train Loss')
        if 'val_loss' in history:
            plt.plot(history['val_loss'], 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Plot IoU
        plt.subplot(1, 2, 2)
        plt.plot(history['train_iou'], 'b-', label='Train IoU')
        if 'val_iou' in history:
            plt.plot(history['val_iou'], 'r-', label='Validation IoU')
        plt.title('Training and Validation IoU')
        plt.xlabel('Epoch')
        plt.ylabel('IoU')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.suptitle(f'Training Progress - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        plt.tight_layout()
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
        print(f"Training history plot saved to {file_path}")
    except Exception as e:
        print(f"Error plotting training history: {str(e)}")

def save_training_report(history, file_path='training_report.txt'):
    """
    Save detailed training report to file
    
    Args:
        history: Dictionary containing training history
        file_path: Path to save the report (default: 'training_report.txt')
    """
    try:
        with open(file_path, 'w') as f:
            f.write(f"Training Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*50 + "\n\n")
            
            # Final metrics
            if 'val_loss' in history:
                f.write(f"Final Validation Loss: {history['val_loss'][-1]:.4f}\n")
                f.write(f"Final Validation IoU: {history['val_iou'][-1]:.4f}\n\n")
            
            f.write(f"Final Training Loss: {history['train_loss'][-1]:.4f}\n")
            f.write(f"Final Training IoU: {history['train_iou'][-1]:.4f}\n\n")
            
            # Best metrics
            if 'val_loss' in history:
                best_loss_epoch = np.argmin(history['val_loss'])
                best_iou_epoch = np.argmax(history['val_iou'])
                
                f.write("Best Validation Metrics:\n")
                f.write(f"- Lowest Loss: {history['val_loss'][best_loss_epoch]:.4f} (epoch {best_loss_epoch+1})\n")
                f.write(f"- Highest IoU: {history['val_iou'][best_iou_epoch]:.4f} (epoch {best_iou_epoch+1})\n\n")
            
            f.write("\nTraining History:\n")
            f.write("Epoch\tTrain Loss\tTrain IoU\tVal Loss\tVal IoU\n")
            for epoch in range(len(history['train_loss'])):
                line = f"{epoch+1}\t{history['train_loss'][epoch]:.4f}\t{history['train_iou'][epoch]:.4f}"
                if 'val_loss' in history:
                    line += f"\t{history['val_loss'][epoch]:.4f}\t{history['val_iou'][epoch]:.4f}"
                f.write(line + "\n")
        
        print(f"Training report saved to {file_path}")
    except Exception as e:
        print(f"Error saving training report: {str(e)}")