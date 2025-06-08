import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import logging
import torch.nn.functional as F

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(model, train_dataset, val_dataset=None, epochs=25, batch_size=4, learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")
    model = model.to(device)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True if device.type == 'cuda' else False
        )
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_iou': [], 'val_iou': []}
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_iou = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        try:
            for batch_idx, (images, masks) in enumerate(progress_bar):
                logger.debug(f"Processing batch {batch_idx+1}/{len(train_loader)}")
                images, masks = images.to(device), masks.to(device)
                
                try:
                    optimizer.zero_grad()
                    outputs = model(pixel_values=images, labels=masks) if hasattr(model, 'model') else model(images)
                    loss = outputs.loss if hasattr(outputs, 'loss') else criterion(outputs, masks)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    
                    # Upsample logits to match ground truth resolution (512x512)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                    if logits.shape[-2:] != (512, 512):
                        logits = F.interpolate(logits, size=(512, 512), mode='bilinear', align_corners=False)
                    
                    # Calculate IoU
                    preds = torch.argmax(logits, dim=1)
                    iou = calculate_iou(preds, masks)
                    epoch_iou += iou
                    
                    progress_bar.set_postfix({'loss': loss.item(), 'iou': iou})
                except Exception as e:
                    logger.error(f"Error in batch {batch_idx+1}: {str(e)}")
                    raise
                
        except Exception as e:
            logger.error(f"Training failed at epoch {epoch+1}: {str(e)}")
            raise
        
        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_iou = epoch_iou / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        history['train_iou'].append(avg_train_iou)
        
        if val_dataset:
            val_loss, val_iou = evaluate(model, val_loader, device)
            history['val_loss'].append(val_loss)
            history['val_iou'].append(val_iou)
            scheduler.step(val_loss)
            
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), 'best_model.pth')
            
            logger.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Train IoU: {avg_train_iou:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")
        else:
            logger.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Train IoU: {avg_train_iou:.4f}")
    
    return model, history

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_iou = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            # Call model with labels to get loss
            outputs = model(pixel_values=images, labels=masks) if hasattr(model, 'model') else model(images)
            
            # Compute loss
            if hasattr(model, 'model'):  # Transformer model (Segformer)
                loss = outputs.loss
            else:  # CNN model
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                if logits.shape[-2:] != (512, 512):
                    logits = F.interpolate(logits, size=(512, 512), mode='bilinear', align_corners=False)
                loss = criterion(logits, masks)
            
            total_loss += loss.item()
            
            # Upsample logits for IoU
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            if logits.shape[-2:] != (512, 512):
                logits = F.interpolate(logits, size=(512, 512), mode='bilinear', align_corners=False)
            
            # Calculate IoU
            preds = torch.argmax(logits, dim=1)
            total_iou += calculate_iou(preds, masks)
    
    return total_loss / len(dataloader), total_iou / len(dataloader)

def calculate_iou(preds, targets, num_classes=3):
    ious = []
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (targets == cls)
        intersection = (pred_inds & target_inds).long().sum().item()
        union = (pred_inds | target_inds).long().sum().item()
        
        if union == 0:
            ious.append(0.0)
        else:
            ious.append(intersection / union)
    
    return np.mean(ious)