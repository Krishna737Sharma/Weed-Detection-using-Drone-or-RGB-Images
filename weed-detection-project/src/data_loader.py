import os
import cv2
import numpy as np
import yaml
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms as transforms
from skimage.draw import polygon

class CWFIDDataLoader:
    def __init__(self, data_path='data/dataset-1.0', use_precomputed_masks=False):
        self.data_path = data_path
        self.use_precomputed_masks = use_precomputed_masks
        self.image_path = os.path.join(data_path, 'images')
        self.mask_path = os.path.join(data_path, 'masks')
        self.annotation_path = os.path.join(data_path, 'annotations')
        self.precomputed_mask_path = os.path.join(data_path, 'preprocessed/masks')
        self.split_file = os.path.join(data_path, 'train_test_split.yaml')
        self.classes = ['background', 'crop', 'weed']  # 3 classes
        
        # Validate paths
        for path in [self.image_path, self.mask_path, self.annotation_path, self.split_file]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Directory/file not found at {path}")
        if use_precomputed_masks and not os.path.exists(self.precomputed_mask_path):
            raise FileNotFoundError(f"Precomputed mask directory not found at {self.precomputed_mask_path}")

    def _get_filenames(self, index):
        """Convert index to properly formatted filenames"""
        return (
            f"{index:03d}_image.png",
            f"{index:03d}_mask.png",
            f"{index:03d}_annotation.yaml",
            f"{index:03d}_class_mask.png"
        )

    def load_image_mask_pair(self, index):
        """Load image and create class-labeled mask from YAML or load precomputed mask"""
        img_name, mask_name, ann_name, class_mask_name = self._get_filenames(index)
        img_path = os.path.join(self.image_path, img_name)
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image at {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512))
        
        # Load class-labeled mask
        if self.use_precomputed_masks:
            class_mask_path = os.path.join(self.precomputed_mask_path, class_mask_name)
            class_mask = cv2.imread(class_mask_path, cv2.IMREAD_GRAYSCALE)
            if class_mask is None:
                raise ValueError(f"Could not read class mask at {class_mask_path}")
        else:
            # Load grayscale mask
            mask_path = os.path.join(self.mask_path, mask_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Could not read mask at {mask_path}")
            mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
            
            # Load YAML annotations
            ann_path = os.path.join(self.annotation_path, ann_name)
            try:
                with open(ann_path, 'r') as f:
                    annotations = yaml.safe_load(f)
            except Exception as e:
                raise ValueError(f"Failed to parse YAML at {ann_path}: {e}")
            
            # Create class-labeled mask
            height, width = 512, 512
            class_mask = np.zeros((height, width), dtype=np.uint8)  # 0: background
            
            # Scale coordinates (assuming original size 1280x960)
            original_width, original_height = 1280, 960  # Adjust if different
            x_scale = width / original_width
            y_scale = height / original_height
            
            if not annotations or 'annotation' not in annotations:
                print(f"Warning: No annotations in {ann_path}")
                return img, class_mask
            
            for ann in annotations.get('annotation', []):
                class_type = ann.get('type')
                class_idx = {'crop': 1, 'weed': 2}.get(class_type, 0)
                points = ann.get('points', {})
                x_coords = points.get('x', [])
                y_coords = points.get('y', [])
                
                # Skip invalid coordinates
                if not isinstance(x_coords, list) or not isinstance(y_coords, list):
                    print(f"Warning: Invalid coordinates in {ann_path} for {class_type}: x={x_coords}, y={y_coords}")
                    continue
                if len(x_coords) < 3 or len(y_coords) < 3 or len(x_coords) != len(y_coords):
                    print(f"Warning: Insufficient coordinates in {ann_path} for {class_type}")
                    continue
                
                # Scale coordinates
                try:
                    scaled_x = np.array(x_coords, dtype=np.float32) * x_scale
                    scaled_y = np.array(y_coords, dtype=np.float32) * y_scale
                except Exception as e:
                    print(f"Warning: Failed to scale coordinates in {ann_path}: {e}")
                    continue
                
                # Rasterize polygon
                try:
                    rr, cc = polygon(scaled_y, scaled_x, shape=(height, width))
                    class_mask[rr, cc] = class_idx
                except Exception as e:
                    print(f"Warning: Skipping invalid polygon in {ann_path}: {e}")
        
        return img, class_mask

    def load_split_data(self):
        """Load data according to the official train-test split"""
        with open(self.split_file, 'r') as f:
            split_data = yaml.safe_load(f)
        
        train_indices = split_data['train']
        test_indices = split_data['test']
        
        train_images, train_masks = [], []
        for idx in train_indices:
            try:
                img, mask = self.load_image_mask_pair(idx)
                train_images.append(img)
                train_masks.append(mask)
            except Exception as e:
                print(f"Skipping train index {idx}: {str(e)}")
        
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