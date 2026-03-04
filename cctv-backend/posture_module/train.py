"""
Training script for posture analysis model.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from pathlib import Path
import json

from model import PostureLSTM

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "datasets" / "processed" / "pose_sequences"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

class PoseSequenceDataset(Dataset):
    """Dataset for loading pose sequences."""
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.samples = []
        
        # Load all .npz files
        for npz_file in sorted(self.data_dir.glob("*.npz")):
            data = np.load(npz_file)
            X = data["X"]  # (T, F)
            y = data["y"]  # scalar label
            
            # Only include sequences with minimum length
            if X.shape[0] >= 10:  # At least 10 frames
                self.samples.append((X, int(y)))
        
        print(f"Loaded {len(self.samples)} sequences")
        # TODO: consider sliding-window augmentation for fall sequences so that
        # windows include the standing→falling transition, not only static lying.
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        X, y = self.samples[idx]
        
        # Pad or truncate to fixed length
        target_length = 60  # Fixed sequence length
        if X.shape[0] < target_length:
            # Pad with zeros
            pad_length = target_length - X.shape[0]
            X = np.pad(X, ((0, pad_length), (0, 0)), mode="constant")
        elif X.shape[0] > target_length:
            # Truncate (take middle portion)
            start = (X.shape[0] - target_length) // 2
            X = X[start:start + target_length]
        
        return torch.FloatTensor(X), torch.LongTensor([y])[0]

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy

def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            logits = model(X)
            loss = criterion(logits, y)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy, all_preds, all_labels

def main():
    # Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    DROPOUT = 0.3
    TRAIN_RATIO = 0.8
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = PoseSequenceDataset(DATA_DIR)
    
    if len(dataset) == 0:
        print("ERROR: No data found. Please run extract_pose_sequences.py first.")
        return
    
    # Split dataset
    train_size = int(TRAIN_RATIO * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    model = PostureLSTM(
        input_size=66,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=3,
        dropout=DROPOUT
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    
    # Training loop
    best_val_acc = 0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    print("\nStarting training...")
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = MODEL_DIR / "posture_model_best.pth"
            torch.save(model.state_dict(), model_path)
            print(f"  ✓ Saved best model (val_acc={val_acc:.4f})")
    
    # Final evaluation
    print("\n" + "="*50)
    print("Final Evaluation")
    print("="*50)
    
    model.load_state_dict(torch.load(MODEL_DIR / "posture_model_best.pth"))
    val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
    
    print(f"\nValidation Accuracy: {val_acc:.4f}")
    print("\nClassification Report:")
    # Get unique classes in the data
    unique_labels = sorted(set(val_labels))
    class_names = ["normal", "fall", "lying"]
    # Only include classes that exist in the data
    target_names = [class_names[i] for i in unique_labels]
    print(classification_report(val_labels, val_preds, target_names=target_names, labels=unique_labels))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(val_labels, val_preds, labels=unique_labels))
    
    # Save final model
    final_model_path = MODEL_DIR / "posture_model_final.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"\nSaved final model to {final_model_path}")
    
    # Save training history
    history = {
        "train_loss": train_losses,
        "train_acc": train_accs,
        "val_loss": val_losses,
        "val_acc": val_accs
    }
    with open(MODEL_DIR / "training_history.json", "w") as f:
        json.dump(history, f)
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()

