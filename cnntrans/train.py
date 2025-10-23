import os
import gc
import time 
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from cnn_transf import HybridCNNTransformer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_checkpoint(state, filename="./transf_sic/cnntransf_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, verbose=True, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode

    def __call__(self, val_metric):
        score = val_metric
        if self.best_score is None:
            self.best_score = score
            return
        
        should_stop = False
        if self.mode == 'max':
            if score < self.best_score + self.min_delta:
                should_stop = True
        elif self.mode == 'min':
            if score > self.best_score - self.min_delta:
                should_stop = True

        if should_stop:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            if self.verbose:
                print("New best score. Resetting counter.")

def load_trained_model(image_size, sequence_length, pretrained):
    model = HybridCNNTransformer(sequence_length=sequence_length, image_size=image_size, pretrained=pretrained)
    checkpoint = torch.load("./transf_sic/cnntransf_best_model.pt", weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def train_model(train_loader, val_loader, image_size, sequence_length, pretrained):
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    set_seed(42)
    
    def smooth_labels(labels, smoothing=0.1):
        return labels * (1 - smoothing) + 0.5 * smoothing

    if pretrained:
        model = HybridCNNTransformer(sequence_length=sequence_length, image_size=image_size, pretrained=pretrained).to(device)
        print("Using pretrained ResNet as extractor model for training.")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
    else:
        model = HybridCNNTransformer(sequence_length=sequence_length, image_size=image_size, pretrained=pretrained).to(device)
        print("Using CNN as extractor model from scratch for training.")
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)

    start_epoch  = 0
    best_val_auc = 0.0
    checkpoint_file = "./transf_sic/cnntransf_checkpoint.pth.tar" 
    if os.path.exists(checkpoint_file):
        print("📦 Checkpoint found. Loading saved model...")
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_auc = checkpoint['best_val_auc']
        print(f"🔁 Resuming from epoch {start_epoch} with best_val_auc = {best_val_auc:.2f}%")

    # --- Training loop ---
    print("Starting training...")
    start_time = time.time()
    num_epochs = 200
    save_every = 3
    early_stopping = EarlyStopping(patience=20, min_delta=0.001, verbose=True, mode='max')

    for epoch in range(start_epoch, num_epochs):
        if pretrained: 
            unfreeze_plan = [
                ("layer3", 20, 5e-5), 
                ("layer2", 40, 1e-5), 
                ("layer1", 70, 5e-6), 
                ("conv1", 100, 1e-6), 
            ] 
            for layer_name, target_epoch, new_lr in unfreeze_plan: 
                if epoch == target_epoch: 
                    print(f"Unfreezing {layer_name} at epoch {epoch}") 
                    layer = getattr(model.cnn_extractor, layer_name) 
                    for param in layer.parameters(): 
                        param.requires_grad = True 
                    for g in optimizer.param_groups: 
                        g['lr'] = new_lr
                        
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            #labels = smooth_labels(labels, smoothing=0.1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)

        model.eval()
        all_labels, all_predscores = [], []
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(images)
                predscores = torch.sigmoid(outputs)
                val_loss += criterion(outputs, labels).item()
                all_labels.extend(labels.cpu().numpy())
                all_predscores.extend(predscores.cpu().numpy())
        try:
            val_auc = roc_auc_score(all_labels, all_predscores)
        except:
            val_auc = 0.0
        preds    = (np.array(all_predscores) > 0.5).astype(int)
        val_acc  = balanced_accuracy_score(all_labels, preds)
        val_loss = val_loss / len(val_loader)
        scheduler.step(val_auc)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val AUC: {val_auc:.4f}, "
              f"Balanced Val ACC: {val_acc:.4f}")
        
        if val_auc >= best_val_auc:
            best_val_auc = max(best_val_auc, val_auc)
            torch.save(model.state_dict(), "./transf_sic/cnntransf_best_model.pt")
            print("⭐ New best model saved!")

        if (epoch + 1) % save_every == 0:
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_auc': best_val_auc
            }
            save_checkpoint(checkpoint)

        early_stopping(val_auc)
        if early_stopping.early_stop:
            print("Early stopping in main training loop")
            break
            
    end_time = time.time()
    duration = end_time - start_time
    
    print("Training complete.")
    print(f"Total training time: {duration:.2f} seconds")
    
    torch.cuda.empty_cache()
    gc.collect()
    model.eval()
    return model
