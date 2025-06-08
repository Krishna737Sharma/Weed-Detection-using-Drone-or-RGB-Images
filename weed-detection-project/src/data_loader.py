import os
import cv2
import numpy as np
import yaml
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms as transforms

class CWFIDDataLoader:
    def __init__(self, data_path='data/dataset-1.0'):
        self.data_path = data_path
        self.image_path = os.path.join(data_path, 'images')
        self.mask_path = os.path.join(data_path, 'masks')
        self.split_file = os.path.join(data_path, 'train_test_split.yaml')
        self.classes = ['soil', 'crop', 'weed']
        
        # Validate paths
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Image directory not found at {self.image_path}")
        if not os.path.exists(self.mask_path):
            raise FileNotFoundError(f"Mask directory not found at {self.mask_path}")
        if not os.path.exists(self.split_file):
            raise FileNotFoundError(f"Split file not found at {self.split_file}")

    def _get_filenames(self, index):
        """Convert index to properly formatted filenames"""
        return (f"{index:03d}_image.png", f"{index:03d}_mask.png")

    def load_image_mask_pair(self, index):
        """Load single image-mask pair by index"""
        img_name, mask_name = self._get_filenames(index)
        img_path = os.path.join(self.image_path, img_name)
        mask_path = os.path.join(self.mask_path, mask_name)
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image at {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512))
        
        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read mask at {mask_path}")
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        
        # Convert mask to 3-class labels (0=soil, 1=crop, 2=weed)
        mask = np.where(mask == 255, 2, np.where(mask == 128, 1, 0))
        
        return img, mask

    def load_split_data(self):
        """Load data according to the official train-test split"""
        with open(self.split_file, 'r') as f:
            split_data = yaml.safe_load(f)
        
        train_indices = split_data['train']
        test_indices = split_data['test']
        
        # Load training data
        train_images, train_masks = [], []
        for idx in train_indices:
            try:
                img, mask = self.load_image_mask_pair(idx)
                train_images.append(img)
                train_masks.append(mask)
            except Exception as e:
                print(f"Skipping train index {idx}: {str(e)}")
        
        # Load test data
        test_images, test_masks = [], []
        for idx in test_indices:
            try:
                img, mask = self.load_image_mask_pair(idx)
                test_images.append(img)
                test_masks.append(mask)
            except Exception as e:
                print(f"Skipping test index {idx}: {str(e)}")
        
        return (
            np.array(train_images), 
            np.array(train_masks),
            np.array(test_images),
            np.array(test_masks)
        )

    def get_datasets(self, transform=None):
        """Get train and test datasets with transforms"""
        X_train, y_train, X_test, y_test = self.load_split_data()
        
        if len(X_train) == 0 or len(X_test) == 0:
            raise ValueError("No data loaded - check your dataset files")
        
        # Default transforms if none provided
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        
        train_dataset = WeedDataset(X_train, y_train, transform=transform)
        test_dataset = WeedDataset(X_test, y_test, transform=transform)
        
        return train_dataset, test_dataset

class WeedDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        mask = self.masks[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.from_numpy(mask).long()