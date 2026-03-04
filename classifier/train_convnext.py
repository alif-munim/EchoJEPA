import argparse
import os
import time
import json
import logging
import random
import tempfile
import heapq
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.amp import GradScaler, autocast
import torch.nn.functional as F

import numpy as np
import pandas as pd
import boto3
import decord
from PIL import Image
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score, 
    confusion_matrix, roc_auc_score, classification_report
)
from sklearn.preprocessing import label_binarize

import timm
from timm.data import create_transform
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy

# -----------------------------------------------------------------------------
# CONFIGURATION MAPS
# -----------------------------------------------------------------------------
LABEL_MAP_VIEW = {
    0: 'A2C', 1: 'A3C', 2: 'A4C', 3: 'A5C', 4: 'Exclude', 5: 'PLAX',
    6: 'PSAX-AP', 7: 'PSAX-AV', 8: 'PSAX-MV', 9: 'PSAX-PM', 10: 'SSN',
    11: 'Subcostal', 12: 'TEE'
}

LABEL_MAP_COLOR = {
    0: 'No',
    1: 'Yes'
}

def setup_logger(output_dir, rank):
    if rank != 0:
        return logging.getLogger()
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(output_dir, 'train_log.txt'))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------
class EchoS3Dataset(Dataset):
    def __init__(self, csv_path, transform=None, is_training=True, num_frames_voting=3):
        # Use python engine and regex separator to handle potential spacing issues in CSV
        self.df = pd.read_csv(csv_path, header=None, names=['s3_uri', 'label'], sep=r'\s+', engine='python')
        self.df['label'] = self.df['label'].astype(int)
        self.data = self.df.to_records(index=False)
        self.transform = transform
        self.is_training = is_training
        self.num_frames_voting = num_frames_voting

    def __len__(self):
        return len(self.data)

    def _get_s3_client(self):
        # Lazy loading: Only create the client inside the worker process
        if not hasattr(self, 's3_client'):
            self.s3_client = boto3.client('s3')
        return self.s3_client

    def _download_video(self, s3_uri):
        # Robust URI parsing
        clean_uri = str(s3_uri).strip()
        if clean_uri.startswith("s3://"):
            clean_uri = clean_uri.replace("s3://", "")
        
        bucket, key = clean_uri.split("/", 1)
        
        fd, temp_path = tempfile.mkstemp(suffix='.mp4')
        os.close(fd)
        
        client = self._get_s3_client()
        try:
            client.download_file(bucket, key, temp_path)
            return temp_path
        except Exception as e:
            if os.path.exists(temp_path): os.remove(temp_path)
            raise e

    def __getitem__(self, idx):
        # ITERATIVE RETRY LOOP
        attempt_count = 0
        
        while True:
            uri, label = self.data[idx]
            temp_path = None
            
            try:
                temp_path = self._download_video(uri)
                vr = decord.VideoReader(temp_path, num_threads=1)
                total_frames = len(vr)
                
                if total_frames < 1:
                    raise ValueError("Empty video")
                
                images = []
                if self.is_training:
                    # Random frame for training
                    frame_idx = random.randint(0, total_frames - 1)
                    indices = [frame_idx]
                else:
                    # Voting logic for validation
                    if total_frames < self.num_frames_voting:
                        # If video is too short, grab all available frames
                        indices = np.arange(total_frames).astype(int)
                    else:
                        indices = np.linspace(0, total_frames - 1, self.num_frames_voting).astype(int)
                
                video_data = vr.get_batch(indices).asnumpy()
                for i in range(len(indices)):
                    img = Image.fromarray(video_data[i])
                    if self.transform: img = self.transform(img)
                    images.append(img)
                
                # Cleanup success
                if os.path.exists(temp_path): os.remove(temp_path)
                
                if self.is_training: 
                    return images[0], label
                else: 
                    # --- VALIDATION PADDING LOGIC ---
                    img_stack = torch.stack(images)
                    current_count = img_stack.shape[0]
                    
                    if current_count < self.num_frames_voting:
                        diff = self.num_frames_voting - current_count
                        last_frame = img_stack[-1:] 
                        padding = last_frame.repeat(diff, 1, 1, 1)
                        img_stack = torch.cat([img_stack, padding], dim=0)
                        
                    return img_stack, label

            except Exception as e:
                # Cleanup failure
                if temp_path and os.path.exists(temp_path): os.remove(temp_path)
                
                attempt_count += 1
                is_404 = "404" in str(e) or "Not Found" in str(e)
                
                # Log sparsely to avoid flooding the console
                if attempt_count % 50 == 1: 
                    print(f"[Worker Warning] Failed {uri}: {e}. Retrying...")

                # Pick a new random index and loop again immediately
                idx = random.randint(0, len(self.data) - 1)

# -----------------------------------------------------------------------------
# INFERENCE ENGINE
# -----------------------------------------------------------------------------
def run_inference(model, loader, device, rank, world_size, num_classes):
    model.eval()
    local_probs = []
    local_labels = []
    
    with torch.no_grad():
        for input, target in loader:
            if input.ndim == 5:
                B, N, C, H, W = input.shape
                input = input.view(B * N, C, H, W)
            else:
                B, C, H, W = input.shape
                N = 1
            
            # OPTIMIZATION: channels_last
            input = input.to(device, memory_format=torch.channels_last)
            target = target.to(device)
            
            with autocast('cuda'):
                logits = model(input)
            
            if N > 1:
                probs = F.softmax(logits, dim=1).view(B, N, num_classes).mean(dim=1)
            else:
                probs = F.softmax(logits, dim=1)
                
            local_probs.append(probs)
            local_labels.append(target)
            
    local_probs = torch.cat(local_probs)
    local_labels = torch.cat(local_labels)
    
    gathered_probs = [torch.zeros_like(local_probs) for _ in range(world_size)]
    gathered_labels = [torch.zeros_like(local_labels) for _ in range(world_size)]
    dist.all_gather(gathered_probs, local_probs)
    dist.all_gather(gathered_labels, local_labels)
    
    if rank == 0:
        return torch.cat(gathered_probs).cpu().numpy(), torch.cat(gathered_labels).cpu().numpy()
    return None, None

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--drop_path', type=float, default=0.4)
    parser.add_argument('--model', type=str, default='convnext_small.fb_in1k')
    parser.add_argument('--output_dir', type=str, default='./output/run2_convnext')
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--val_csv', type=str, required=True)
    parser.add_argument('--test_csv', type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", 0)))
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--mode', type=str, default='view', choices=['view', 'color'], help='Task mode: view (13 classes) or color (binary)')
    args = parser.parse_args()

    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    rank = dist.get_rank()
    
    # Select Mapping based on Mode
    if args.mode == 'color':
        current_map = LABEL_MAP_COLOR
    else:
        current_map = LABEL_MAP_VIEW
        
    num_classes = len(current_map)
    class_names = [current_map[i] for i in range(num_classes)]

    # OPTIMIZATION: Benchmark mode
    torch.backends.cudnn.benchmark = True
    
    logger = setup_logger(args.output_dir, rank)
    if rank == 0: 
        logger.info(f"Args: {args}")
        logger.info(f"Running Mode: {args.mode} | Num Classes: {num_classes}")

    model = timm.create_model(args.model, pretrained=True, num_classes=num_classes, drop_path_rate=args.drop_path)
    model = model.to(device, memory_format=torch.channels_last)
    model = DDP(model, device_ids=[args.local_rank])

    train_tf = create_transform(args.img_size, is_training=True, auto_augment='rand-m9-mstd0.5-inc1', 
                                interpolation='bicubic', mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    val_tf = create_transform(args.img_size, is_training=False, interpolation='bicubic',
                              mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    mixup_fn = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5, mode='batch', label_smoothing=0.1, num_classes=num_classes)

    loader_train = DataLoader(EchoS3Dataset(args.train_csv, train_tf, True), batch_size=args.batch_size, 
                              sampler=DistributedSampler(EchoS3Dataset(args.train_csv), shuffle=True),
                              num_workers=4, pin_memory=True, drop_last=True)
    loader_val = DataLoader(EchoS3Dataset(args.val_csv, val_tf, False, 3), batch_size=args.batch_size // 2, 
                            sampler=DistributedSampler(EchoS3Dataset(args.val_csv), shuffle=False),
                            num_workers=4, pin_memory=True)
    loader_test = DataLoader(EchoS3Dataset(args.test_csv, val_tf, False, 3), batch_size=args.batch_size // 2,
                             sampler=DistributedSampler(EchoS3Dataset(args.test_csv), shuffle=False),
                             num_workers=4, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion_train = SoftTargetCrossEntropy()
    scaler = GradScaler('cuda')

    start_epoch = 0
    top_3_models = [] 

    # RESUME LOGIC
    if args.resume and os.path.exists(args.resume):
        if rank == 0: logger.info(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.module.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1

    for epoch in range(start_epoch, args.epochs):
        loader_train.sampler.set_epoch(epoch)
        model.train()
        if rank == 0: logger.info(f"=== Epoch {epoch+1}/{args.epochs} ===")
        
        for i, (input, target) in enumerate(loader_train):
            input = input.to(device, memory_format=torch.channels_last)
            target = target.to(device)
            input, target = mixup_fn(input, target)
            
            with autocast('cuda'):
                loss = criterion_train(model(input), target)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            if i % 100 == 0 and rank == 0: 
                logger.info(f"Iter {i}: Loss {loss.item():.4f}")

        scheduler.step()

        val_probs, val_labels = run_inference(model, loader_val, device, rank, dist.get_world_size(), num_classes)

        if rank == 0:
            val_preds = np.argmax(val_probs, axis=1)
            val_f1 = f1_score(val_labels, val_preds, average='macro')
            val_acc = accuracy_score(val_labels, val_preds)
            logger.info(f"Epoch {epoch+1} Val F1: {val_f1:.4f} | Acc: {val_acc:.4f}")
            
            # Checkpointing
            ckpt_name = f"epoch_{epoch+1}.pt"
            save_path = os.path.join(args.output_dir, ckpt_name)
            save_state = {
                'model': model.module.state_dict(), 
                'optimizer': optimizer.state_dict(), 
                'scheduler': scheduler.state_dict(), 
                'scaler': scaler.state_dict(), 
                'epoch': epoch, 
                'val_f1': val_f1
            }
            torch.save(save_state, save_path)
            
            # Top 3 Logic
            if len(top_3_models) < 3:
                heapq.heappush(top_3_models, (val_f1, epoch+1, ckpt_name))
            else:
                if val_f1 > top_3_models[0][0]:
                    removed = heapq.heappop(top_3_models)
                    removed_ckpt_path = os.path.join(args.output_dir, removed[2])
                    if os.path.exists(removed_ckpt_path): os.remove(removed_ckpt_path)
                    logger.info(f"Removed old checkpoint: {removed[2]}")
                    heapq.heappush(top_3_models, (val_f1, epoch+1, ckpt_name))
                else:
                    if os.path.exists(save_path): os.remove(save_path)
            
            torch.save(save_state, os.path.join(args.output_dir, "last.pt"))

    dist.barrier()
    
    if rank == 0:
        logger.info("\n" + "="*40)
        logger.info(f" TRAINING COMPLETE. EVALUATING TOP {len(top_3_models)} MODELS")
        logger.info("="*40)
    
    # Broadcast top 3 list
    top_3_ckpts = [None] * 3
    if rank == 0:
        sorted_list = sorted(top_3_models, key=lambda x: x[0], reverse=True)
        top_3_ckpts = [x[2] for x in sorted_list]
        while len(top_3_ckpts) < 3: top_3_ckpts.append(None)
    
    object_list = [top_3_ckpts]
    dist.broadcast_object_list(object_list, src=0)
    top_3_ckpts = object_list[0]

    for i, ckpt_name in enumerate(top_3_ckpts):
        if ckpt_name is None: continue
        
        ckpt_path = os.path.join(args.output_dir, ckpt_name)
        map_loc = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
        checkpoint = torch.load(ckpt_path, map_location=map_loc)
        model.module.load_state_dict(checkpoint['model'])
        
        test_probs, test_labels = run_inference(model, loader_test, device, rank, dist.get_world_size(), num_classes)
        
        if rank == 0:
            test_preds = np.argmax(test_probs, axis=1)
            acc = accuracy_score(test_labels, test_preds)
            f1_macro = f1_score(test_labels, test_preds, average='macro')
            
            # --- METRIC SWITCH LOGIC ---
            try:
                if args.mode == 'color':
                    # Binary: use Probability of Positive Class (Index 1)
                    auroc = roc_auc_score(test_labels, test_probs[:, 1])
                else:
                    # Multi-class: One-Hot encode
                    test_labels_onehot = label_binarize(test_labels, classes=range(num_classes))
                    auroc = roc_auc_score(test_labels_onehot, test_probs, multi_class='ovr', average='macro')
            except Exception as e:
                print(f"AUROC Error: {e}")
                auroc = 0.0
            
            report = classification_report(test_labels, test_preds, target_names=class_names, digits=4)
            cm = confusion_matrix(test_labels, test_preds)
            
            try: ep_num = ckpt_name.split('_')[1].split('.')[0]
            except: ep_num = "unknown"

            prefix = f"rank{i+1}_epoch{ep_num}"
            metrics = {"accuracy": acc, "f1_macro": f1_macro, "auroc": auroc}
            
            with open(os.path.join(args.output_dir, f'{prefix}_metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=4)
            with open(os.path.join(args.output_dir, f'{prefix}_report.txt'), 'w') as f:
                f.write(report)
            pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(os.path.join(args.output_dir, f'{prefix}_cm.csv'))
            
            logger.info(f"Rank {i+1} Results -> Acc: {acc:.4f} | F1: {f1_macro:.4f} | AUROC: {auroc:.4f}")

if __name__ == '__main__':
    main()