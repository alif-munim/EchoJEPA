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
from timm.data import resolve_data_config, Mixup
from timm.loss import SoftTargetCrossEntropy
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
    def __init__(self, patience=10, min_delta=0.001, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode  # "min" or "max"
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

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean
    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

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
def get_model(model_name, num_classes, local_weights_path=None, drop_path_rate=0.0):
    print(f"Loading {model_name} (drop_path={drop_path_rate})...")
    # Using 'in12k_ft_in1k_384' tag ensures we get the architecture compatible with 384px
    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
    )

    if local_weights_path and os.path.exists(local_weights_path):
        print(f"Loading weights from: {local_weights_path}")
        checkpoint = torch.load(local_weights_path, map_location="cpu")
        
        # Handle different checkpoint keys (timm vs standard)
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Remove head weights if size mismatch (transfer learning)
        keys_to_remove = [k for k in state_dict.keys() if "head" in k]
        for k in keys_to_remove:
            del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Weights loaded. Missing keys (expected for head): {msg.missing_keys}")
    else:
        print("WARNING: Training from scratch (no local weights found).")

    return model

# ================= TRAINING / EVAL HELPERS =================
def train_one_epoch(model, loader, criterion, optimizer, device, mixup_fn=None):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="Train", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # Apply Mixup if enabled
        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        
        # If mixup is active, we can't easily calculate accuracy/F1 in the loop
        if mixup_fn is None:
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(loader.dataset)
    
    # Return 0.0 F1 if using Mixup
    if mixup_fn is not None:
        epoch_f1 = 0.0 
    else:
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

def get_all_preds(model, loader, device):
    """Standard evaluation prediction."""
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)

def get_all_preds_tta(model, loader, device):
    """
    Test Time Augmentation (TTA):
    Predicts on (Image) and (Flip(Image)) and averages the results.
    """
    model.eval()
    all_labels, all_preds = [], []
    
    # No tqdm here if running inside a larger loop, but useful for final eval
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="TTA Eval"):
            images, labels = images.to(device), labels.to(device)
            
            # 1. Forward pass (Original)
            output_orig = model(images).softmax(dim=1)
            
            # 2. Forward pass (Flipped)
            # Flip width dimension (Batch, C, H, W) -> dim 3
            images_flipped = torch.flip(images, dims=[3]) 
            output_flip = model(images_flipped).softmax(dim=1)
            
            # 3. Average
            avg_output = (output_orig + output_flip) / 2.0
            
            preds = torch.argmax(avg_output, dim=1)
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

def plot_history(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val')
    plt.title('Loss'); plt.legend()
    plt.subplot(1, 2, 2)
    if any(f > 0 for f in history['train_f1']):
        plt.plot(epochs, history['train_f1'], 'b-', label='Train')
    plt.plot(epochs, history['val_f1'], 'r-', label='Val')
    plt.title('F1 Score'); plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)

# ================= MAIN =================
def main():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument("--results_dir", type=str, default="results/convnext_run", help="Output folder")
    parser.add_argument("--csv_file", type=str, default="labels_masked_inplace.csv")
    parser.add_argument("--img_dir", type=str, default=".")
    parser.add_argument("--weights", type=str, default="convnext_base_384.bin")
    parser.add_argument("--resume", type=str, default=None)

    # Model Architecture (NEW ARGUMENT)
    parser.add_argument("--model_name", type=str, default="convnext_base.in12k_ft_in1k_384", 
                        help="timm model name (e.g., convnext_base.in12k_ft_in1k_384)")

    # Hyperparams (DEFAULTS FOR BASE)
    parser.add_argument("--epochs", type=int, default=50) 
    parser.add_argument("--batch_size", type=int, default=64) 
    parser.add_argument("--lr", type=float, default=1e-4) 

    # Regularization & Augmentation
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--drop_path", type=float, default=0.5)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--mixup", type=float, default=0.8)
    parser.add_argument("--cutmix", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=15) 

    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Setup
    os.makedirs(args.results_dir, exist_ok=True)
    sys.stdout = Logger(os.path.join(args.results_dir, "console.log"))
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

    # Model (Using args.model_name now)
    model = get_model(args.model_name, len(classes), args.weights, drop_path_rate=args.drop_path)
    model = model.to(device)

    # Config
    config = resolve_data_config({}, model=model)
    mean = config['mean']
    std = config['std']
    img_size = config['input_size'][1:]  # (H, W)

    # Augmentations (UPDATED WITH GRAYSCALE + JITTER)
    train_tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomResizedCrop(img_size[0], scale=(0.6, 1.0)), 
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), shear=10),
        transforms.RandomHorizontalFlip(),
        
        # --- NEW: DOPPLER/GAIN INVARIANCE ---
        transforms.RandomGrayscale(p=0.2), 
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        # ------------------------------------
        
        transforms.ToTensor(),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),
        transforms.RandomApply([AddGaussianNoise(0., 0.05)], p=0.4),
        transforms.Normalize(mean=mean, std=std),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size[0]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # MIXUP setup
    mixup_fn = None
    if args.mixup > 0 or args.cutmix > 0:
        print("Enabling Mixup/Cutmix...")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, 
            cutmix_alpha=args.cutmix, 
            cutmix_minmax=None,
            prob=1.0, 
            switch_prob=0.5, 
            mode='batch',
            label_smoothing=args.label_smoothing, 
            num_classes=len(classes)
        )

    # Datasets & Loaders
    train_dataset = EchoDataset(train_df, args.img_dir, train_tf, class_to_idx)
    val_dataset = EchoDataset(val_df, args.img_dir, eval_tf, class_to_idx)
    test_dataset = EchoDataset(test_df, args.img_dir, eval_tf, class_to_idx)

    # Balanced sampler
    train_labels_idx = train_df['label'].map(class_to_idx).values
    class_counts = np.bincount(train_labels_idx, minlength=len(classes))
    class_weights_sampler = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights_sampler[train_labels_idx]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # UPDATED WORKERS TO 16
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)
    train_eval_loader = DataLoader(EchoDataset(train_df, args.img_dir, eval_tf, class_to_idx), batch_size=args.batch_size, shuffle=False, num_workers=16)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Loss Functions
    if mixup_fn is not None:
        train_criterion = SoftTargetCrossEntropy()
    else:
        train_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    val_criterion = nn.CrossEntropyLoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True, min_lr=1e-6)
    early_stopper = EarlyStopping(patience=args.patience, min_delta=0.001, mode="max")

    # Loop
    start_epoch = 0
    best_f1 = 0.0
    history = {'train_loss': [], 'train_f1': [], 'val_loss': [], 'val_f1': []}

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_f1 = ckpt.get('best_f1', 0.0)

    print(f"\nStarting training for {args.epochs} epochs with Batch Size {args.batch_size}...")
    for epoch in range(start_epoch, args.epochs):
        # TRAIN
        tr_loss, tr_f1 = train_one_epoch(model, train_loader, train_criterion, optimizer, device, mixup_fn)
        
        # VAL
        val_loss, val_f1, _, _ = validate(model, val_loader, val_criterion, device)

        history['train_loss'].append(tr_loss)
        history['train_f1'].append(tr_f1)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)

        scheduler.step(val_f1)

        save_msg = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(args.results_dir, "best_model.pth"))
            save_msg = "--> Best"

        print(f"Ep {epoch+1} | TrL: {tr_loss:.4f} | VaL: {val_loss:.4f} F1: {val_f1:.3f} {save_msg}")

        early_stopper(val_f1)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

    # ================= FINAL EVAL (USING TTA) =================
    print("\nGenerating final reports with TTA...")
    model.load_state_dict(torch.load(os.path.join(args.results_dir, "best_model.pth"), map_location=device))
    model.to(device)

    # Train Eval (Clean, no mixup) - No TTA needed here usually, but consistent to check
    print("Evaluating on Train set (Clean)...")
    train_labels_all, train_preds_all = get_all_preds(model, train_eval_loader, device)
    
    # Val Eval
    print("Evaluating on Val set...")
    _, _, val_labels, val_preds = validate(model, val_loader, val_criterion, device)
    
    # Test Eval WITH TTA
    if len(test_df) > 0:
        print("Evaluating on Test set (with TTA)...")
        test_labels_all, test_preds_all = get_all_preds_tta(model, test_loader, device)
        test_f1 = f1_score(test_labels_all, test_preds_all, average='macro')
        print(f"Final Test F1 (TTA): {test_f1:.3f}")
        with open(os.path.join(args.results_dir, "test_report.txt"), "w") as f:
            f.write(classification_report(test_labels_all, test_preds_all, target_names=classes))
        
        save_confusion_matrix(test_labels_all, test_preds_all, classes, os.path.join(args.results_dir, "confusion_test.png"), "Test Confusion Matrix")

    # Save Train/Val Matrices
    save_confusion_matrix(train_labels_all, train_preds_all, classes, os.path.join(args.results_dir, "confusion_train.png"), "Train Confusion Matrix")
    save_confusion_matrix(val_labels, val_preds, classes, os.path.join(args.results_dir, "confusion_val.png"), "Validation Confusion Matrix")

    pd.DataFrame(history).to_csv(os.path.join(args.results_dir, "log.csv"), index=False)
    plot_history(history, os.path.join(args.results_dir, "curves.png"))

if __name__ == "__main__":
    main()