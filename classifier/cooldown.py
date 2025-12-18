import os
import argparse
import sys
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm
from timm.data import resolve_data_config
from tqdm import tqdm

# ================= UTILS =================
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0001, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best = None
        self.early_stop = False

    def __call__(self, metric):
        if self.best is None:
            self.best = metric
            return

        if self.mode == "min":
            improvement = metric < (self.best - self.min_delta)
        else:  # "max"
            improvement = metric > (self.best + self.min_delta)

        if improvement:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            print(f"[EarlyStopping] Counter: {self.counter}/{self.patience} (Best: {self.best:.4f})")
            if self.counter >= self.patience:
                self.early_stop = True

# ================= DATASET =================
class EchoDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, class_map=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.class_map = class_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (384, 384))

        if self.transform:
            image = self.transform(image)

        label = self.class_map[row['label']]
        return image, label

# ================= MODEL =================
def get_model(model_name, num_classes, pretrained_path=None):
    print(f"Loading {model_name}...")
    model = timm.create_model(
        model_name,
        pretrained=False, # We load our own weights below
        num_classes=num_classes,
        drop_path_rate=0.0, # Disable drop path for cooldown
    )

    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading user weights from: {pretrained_path}")
        # Load state dict (handling both full checkpoint dict and pure state dict)
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        msg = model.load_state_dict(state_dict, strict=True)
        print(f"Weights loaded successfully: {msg}")
    else:
        print(f"ERROR: Could not find weights at {pretrained_path}")
        sys.exit(1)

    return model

# ================= TRAINING / EVAL HELPERS =================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="Cooling", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(loader.dataset)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    return epoch_loss, epoch_f1

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="Val", leave=False)
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    return epoch_loss, epoch_f1, all_labels, all_preds

def get_all_preds_tta(model, loader, device):
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="TTA Eval"):
            images, labels = images.to(device), labels.to(device)
            # TTA: Average of original and flip
            out1 = model(images).softmax(dim=1)
            out2 = model(torch.flip(images, dims=[3])).softmax(dim=1)
            avg_out = (out1 + out2) / 2.0
            preds = torch.argmax(avg_out, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)

def save_confusion_matrix(labels, preds, classes, save_path, title):
    cm = confusion_matrix(labels, preds, labels=range(len(classes)))
    fig, ax = plt.subplots(figsize=(12, 12)) 
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, cmap='Blues', colorbar=False, xticks_rotation=45)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

# ================= MAIN =================
def main():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument("--results_dir", type=str, default="results/cooldown_run", help="Output folder")
    parser.add_argument("--csv_file", type=str, default="labels_masked_inplace.csv")
    parser.add_argument("--img_dir", type=str, default=".")
    
    # Pretrained path from prompt default
    parser.add_argument("--start_weights", type=str, required=True, help="Path to checkpoint")
    
    # Model Architecture (NEW)
    parser.add_argument("--model_name", type=str, default="convnext_small.in12k_ft_in1k_384", 
                        help="Model name (must match weights)")

    # Hyperparams (Cool Down Settings)
    parser.add_argument("--epochs", type=int, default=10) 
    parser.add_argument("--batch_size", type=int, default=128) 
    parser.add_argument("--lr", type=float, default=1e-6) # Very low LR

    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Setup
    os.makedirs(args.results_dir, exist_ok=True)
    sys.stdout = Logger(os.path.join(args.results_dir, "console_cooldown.log"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Output: {args.results_dir}")

    # Data
    print(f"Reading {args.csv_file}...")
    df = pd.read_csv(args.csv_file)
    classes = sorted(df['label'].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']

    # Model
    model = get_model(args.model_name, len(classes), args.start_weights)
    model = model.to(device)

    # Config
    config = resolve_data_config({}, model=model)
    mean = config['mean']
    std = config['std']
    img_size = config['input_size'][1:]

    # COOL DOWN AUGMENTATION (Weak)
    train_tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomCrop(img_size[0], padding=16),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size[0]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Datasets
    train_dataset = EchoDataset(train_df, args.img_dir, train_tf, class_to_idx)
    val_dataset = EchoDataset(val_df, args.img_dir, eval_tf, class_to_idx)
    test_dataset = EchoDataset(test_df, args.img_dir, eval_tf, class_to_idx)

    # Weighted Sampler
    train_labels_idx = train_df['label'].map(class_to_idx).values
    class_counts = np.bincount(train_labels_idx, minlength=len(classes))
    weights = 1.0 / (class_counts + 1e-6)
    sampler = WeightedRandomSampler(weights[train_labels_idx], len(train_labels_idx))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)

    # Optimizer (Low LR)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    early_stopper = EarlyStopping(patience=5, min_delta=0.0001, mode="max")

    # Loop
    best_f1 = 0.0
    print(f"\nStarting Cool Down for {args.epochs} epochs (LR={args.lr})...")
    
    for epoch in range(args.epochs):
        tr_loss, tr_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1, _, _ = validate(model, val_loader, criterion, device)

        save_msg = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(args.results_dir, "best_model_cooldown.pth"))
            save_msg = "--> Best"

        print(f"Ep {epoch+1} | TrL: {tr_loss:.4f} F1: {tr_f1:.3f} | VaL: {val_loss:.4f} F1: {val_f1:.3f} {save_msg}")

        early_stopper(val_f1)
        if early_stopper.early_stop:
            print("Early stopping.")
            break

    # Final Eval
    print("\nGenerating Final Cool Down Report...")
    model.load_state_dict(torch.load(os.path.join(args.results_dir, "best_model_cooldown.pth"), map_location=device))
    model.to(device)

    if len(test_df) > 0:
        test_labels, test_preds = get_all_preds_tta(model, test_loader, device)
        test_f1 = f1_score(test_labels, test_preds, average='macro')
        print(f"Final Test F1 (After Cool Down): {test_f1:.4f}")
        with open(os.path.join(args.results_dir, "test_report.txt"), "w") as f:
            f.write(classification_report(test_labels, test_preds, target_names=classes))
        save_confusion_matrix(test_labels, test_preds, classes, os.path.join(args.results_dir, "confusion_test.png"), "Test CM")

if __name__ == "__main__":
    main()