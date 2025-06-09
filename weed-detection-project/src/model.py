import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation

class EfficientWeedDetector(nn.Module):
    def __init__(self, num_classes=3, model_name="nvidia/mit-b0"):
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=num_classes,  # 3 classes: background, crop, weed
            ignore_mismatched_sizes=True
        )
        
    def forward(self, pixel_values, labels=None):
        return self.model(pixel_values=pixel_values, labels=labels)
    
    def predict(self, x):
        with torch.no_grad():
            outputs = self(x)
        return torch.argmax(outputs.logits, dim=1)

def build_model(input_shape=(3, 512, 512), num_classes=3):
    """Alternative simple CNN model for comparison"""
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(32, num_classes, 2, stride=2),
            )
            
        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x
    
    return SimpleCNN(num_classes)