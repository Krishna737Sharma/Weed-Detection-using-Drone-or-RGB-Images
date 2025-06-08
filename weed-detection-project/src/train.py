import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm

def train_model(model, train_dataset, val_dataset=None, epochs=25, batch_size=8, learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )
    
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True
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
        
        for images, masks in progress_bar:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(pixel_values=images, labels=masks) if hasattr(model, 'model') else model(images)
            loss = outputs.loss if hasattr(outputs, 'loss') else criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Calculate IoU
            preds = torch.argmax(outputs.logits if hasattr(outputs, 'logits') else outputs, dim=1)
            iou = calculate_iou(preds, masks)
            epoch_iou += iou
            
            progress_bar.set_postfix({'loss': loss.item(), 'iou': iou})
        
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
            
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Train IoU: {avg_train_iou:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")
        else:
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Train IoU: {avg_train_iou:.4f}")
    
    return model, history

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_iou = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(pixel_values=images) if hasattr(model, 'model') else model(images)
            loss = outputs.loss if hasattr(outputs, 'loss') else criterion(outputs, masks)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs.logits if hasattr(outputs, 'logits') else outputs, dim=1)
            total_iou += calculate_iou(preds, masks)
    
    avg_loss = total_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    return avg_loss, avg_iou

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
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    
    return np.nanmean(ious)