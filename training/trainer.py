"""
Model Training Module
Training loop for deepfake detection models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict, Callable
import time
import json


class DeepfakeTrainer:
    """Trainer for deepfake detection models."""
    
    def __init__(self, model: nn.Module, device: str = 'cpu',
                 learning_rate: float = 1e-4, weight_decay: float = 1e-5):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            device: 'cpu' or 'cuda'
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
        """
        self.model = model.to(device)
        self.device = torch.device(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
        
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            # Handle video (B, T, C, H, W) or image (B, C, H, W) input
            if len(inputs.shape) == 5:
                # Video: use middle frame or aggregate
                inputs = inputs[:, inputs.shape[1] // 2]
            
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': correct / total
        }
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                if len(inputs.shape) == 5:
                    inputs = inputs[:, inputs.shape[1] // 2]
                    
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': correct / total
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 50, save_dir: str = './checkpoints',
              callback: Optional[Callable] = None) -> Dict:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            save_dir: Directory to save checkpoints
            callback: Optional callback(epoch, metrics)
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        best_val_acc = 0
        
        for epoch in range(epochs):
            start = time.time()
            
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            self.scheduler.step()
            
            elapsed = time.time() - start
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs} ({elapsed:.1f}s)")
            print(f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']*100:.2f}%")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']*100:.2f}%")
            
            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                self.save_checkpoint(save_path / 'best_model.pth', epoch, val_metrics)
                print(f"  ✓ New best model saved!")
            
            # Callback
            if callback:
                callback(epoch, {'train': train_metrics, 'val': val_metrics})
        
        # Save final model and history
        self.save_checkpoint(save_path / 'final_model.pth', epochs - 1, val_metrics)
        
        with open(save_path / 'history.json', 'w') as f:
            json.dump(self.history, f)
        
        return self.history
    
    def save_checkpoint(self, path: Path, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint.get('epoch', 0)
