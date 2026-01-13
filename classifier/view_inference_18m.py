import argparse
import os
import sys
import time
import tempfile
import logging
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.amp import autocast
import torch.nn.functional as F
import pandas as pd
import numpy as np
import boto3
import decord
from PIL import Image
from tqdm import tqdm
import timm
from timm.data import create_transform

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
# Must match the training configuration exactly
LABEL_MAP = {
    0: 'A2C', 1: 'A3C', 2: 'A4C', 3: 'A5C', 4: 'Exclude', 5: 'PLAX',
    6: 'PSAX-AP', 7: 'PSAX-AV', 8: 'PSAX-MV', 9: 'PSAX-PM', 10: 'SSN',
    11: 'Subcostal', 12: 'TEE'
}
NUM_CLASSES = len(LABEL_MAP)
IMG_SIZE = 336 # From run5 config

def setup_logger(rank):
    logger = logging.getLogger(f'inference_rank{rank}')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(f'[Rank {rank}] %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

# -----------------------------------------------------------------------------
# INFERENCE DATASET
# -----------------------------------------------------------------------------
class InferenceDataset(Dataset):
    def __init__(self, df, transform=None, num_frames=3):
        self.data = df.to_records(index=False) # Expecting ['s3_uri'] column
        self.transform = transform
        self.num_frames = num_frames

    def __len__(self):
        return len(self.data)

    def _get_s3_client(self):
        # Lazy loading for fork safety
        if not hasattr(self, 's3_client'):
            # Optimization: distinct sessions might help throughput
            session = boto3.session.Session()
            self.s3_client = session.client('s3')
        return self.s3_client

    def _download_video(self, s3_uri):
        try:
            clean_uri = str(s3_uri).strip()
            if clean_uri.startswith("s3://"):
                clean_uri = clean_uri.replace("s3://", "")
            
            bucket, key = clean_uri.split("/", 1)
            fd, temp_path = tempfile.mkstemp(suffix='.mp4')
            os.close(fd)
            
            self._get_s3_client().download_file(bucket, key, temp_path)
            return temp_path
        except Exception:
            if os.path.exists(temp_path): os.remove(temp_path)
            return None

    def __getitem__(self, idx):
        # Tuple unpacking handles if there are extra cols, we just want the first one
        row = self.data[idx]
        uri = row[0] 
        
        temp_path = self._download_video(uri)
        
        # FAILURE HANDLING
        if temp_path is None:
            # Return dummy tensor and invalid flag
            return torch.zeros((self.num_frames, 3, IMG_SIZE, IMG_SIZE)), uri, False

        try:
            vr = decord.VideoReader(temp_path, num_threads=1)
            total_frames = len(vr)
            
            if total_frames < 1:
                raise ValueError("Empty video")

            # Voting Logic: Uniformly sample frames
            if total_frames < self.num_frames:
                indices = np.arange(total_frames).astype(int)
            else:
                indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
            
            images = []
            video_data = vr.get_batch(indices).asnumpy()
            
            for i in range(len(indices)):
                img = Image.fromarray(video_data[i])
                if self.transform: img = self.transform(img)
                images.append(img)
            
            os.remove(temp_path)
            
            # PADDING LOGIC (Same as training)
            img_stack = torch.stack(images)
            if img_stack.shape[0] < self.num_frames:
                diff = self.num_frames - img_stack.shape[0]
                last_frame = img_stack[-1:]
                padding = last_frame.repeat(diff, 1, 1, 1)
                img_stack = torch.cat([img_stack, padding], dim=0)
            
            return img_stack, uri, True

        except Exception as e:
            if os.path.exists(temp_path): os.remove(temp_path)
            # Return dummy tensor and invalid flag
            return torch.zeros((self.num_frames, 3, IMG_SIZE, IMG_SIZE)), uri, False

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, required=True, help="Path to csv with s3_uri column")
    parser.add_argument('--output_dir', type=str, default="./inference_output", help="Dir to save chunked results")
    parser.add_argument('--checkpoint', type=str, default="/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2/classifier/output/run5_convnext_small_336px/epoch_89.pt")
    parser.add_argument('--limit', type=int, default=None, help="Process only first N rows (for testing)")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size per GPU")
    parser.add_argument('--num_frames', type=int, default=3, help="Number of frames to sample per video")
    parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", 0)))
    args = parser.parse_args()

    # 1. DDP Setup
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    logger = setup_logger(rank)
    
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Starting inference on {world_size} GPUs.")
        logger.info(f"Checkpoint: {args.checkpoint}")
        logger.info(f"Frames per video: {args.num_frames}")

    # 2. Load Model
    # Important: Model definition must match training exactly (convnext_small, 336px)
    model = timm.create_model('convnext_small.fb_in1k', pretrained=False, num_classes=NUM_CLASSES)
    
    # Load Weights
    # Map storage to current device
    ckpt = torch.load(args.checkpoint, map_location=f'cuda:{args.local_rank}')
    # Handle state dict prefix 'module.' if it exists (it usually does from DDP training)
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    
    # Standardizing:
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "") 
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    
    model.to(device)
    # Optimization: Channels Last
    model = model.to(memory_format=torch.channels_last)
    model = DDP(model, device_ids=[args.local_rank])
    model.eval()

    # 3. Prepare Data
    if rank == 0: logger.info("Loading CSV...")
    
    # Read only first column to save memory
    df = pd.read_csv(args.input_csv, usecols=[0], names=['s3_uri'], header=0)
    
    if args.limit:
        df = df.head(args.limit)
    
    if rank == 0: logger.info(f"Total rows to process: {len(df)}")

    # Transform (Must match validation transform)
    val_tf = create_transform(IMG_SIZE, is_training=False, interpolation='bicubic',
                              mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    # Pass dynamic num_frames argument here
    dataset = InferenceDataset(df, transform=val_tf, num_frames=args.num_frames)
    
    # Distributed Sampler splits the 18M files among 8 GPUs
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    # Num workers: 8 per GPU = 64 total workers downloading files. 
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, 
                        num_workers=8, pin_memory=True, drop_last=False)

    # 4. Inference Loop
    results = []
    
    # Output file for this rank
    output_file = os.path.join(args.output_dir, f"predictions_rank{rank}.csv")
    
    # Buffer control to write incrementally (saves RAM)
    buffer_size = 1000
    
    # --- PROGRESS BAR SETUP ---
    # Only show progress bar on Rank 0
    if rank == 0:
        iterator = tqdm(loader, desc="Inference", total=len(loader), unit="batch")
    else:
        iterator = loader
    # --------------------------

    start_time = time.time()
    
    with torch.no_grad():
        for i, (inputs, uris, valids) in enumerate(iterator):
            # Inputs shape: [B, num_frames, 3, 336, 336]
            
            valid_mask = valids.bool()
            batch_results = []
            
            if valid_mask.sum() > 0:
                # Select valid inputs
                valid_inputs = inputs[valid_mask] # [N_valid, num_frames, 3, 336, 336]
                B_valid, Frames, C, H, W = valid_inputs.shape
                
                # Flatten for inference: [N_valid * num_frames, 3, 336, 336]
                valid_inputs = valid_inputs.view(-1, C, H, W).to(device, memory_format=torch.channels_last)
                
                with autocast('cuda'):
                    logits = model(valid_inputs)
                
                # Voting Logic
                # Reshape back to [N_valid, Frames, Num_Classes] and mean
                probs = F.softmax(logits, dim=1).view(B_valid, Frames, NUM_CLASSES).mean(dim=1)
                
                # Get Preds
                confs, preds = torch.max(probs, dim=1)
                
                # Move to CPU
                preds = preds.cpu().numpy()
                confs = confs.cpu().numpy()
                probs = probs.cpu().numpy()
                
                # Map back to original batch index
                valid_idx_ptr = 0
                for j in range(len(uris)):
                    uri = uris[j]
                    if valids[j]:
                        pred_idx = preds[valid_idx_ptr]
                        pred_label = LABEL_MAP[pred_idx]
                        confidence = confs[valid_idx_ptr]
                        
                        batch_results.append({
                            's3_uri': uri,
                            'prediction': pred_label,
                            'confidence': float(confidence),
                            'status': 'OK'
                        })
                        valid_idx_ptr += 1
                    else:
                        batch_results.append({
                            's3_uri': uri,
                            'prediction': None,
                            'confidence': 0.0,
                            'status': 'ERROR_DOWNLOAD'
                        })
            else:
                # All failed in this batch
                for uri in uris:
                    batch_results.append({
                        's3_uri': uri,
                        'prediction': None,
                        'confidence': 0.0,
                        'status': 'ERROR_DOWNLOAD'
                    })

            results.extend(batch_results)
            
            # Incremental Write
            if len(results) >= buffer_size:
                df_res = pd.DataFrame(results)
                header = not os.path.exists(output_file)
                df_res.to_csv(output_file, mode='a', header=header, index=False)
                results = [] 
            
            # --- PROGRESS BAR UPDATE ---
            if rank == 0:
                if i % 5 == 0:
                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        rate = ((i + 1) * args.batch_size * world_size) / elapsed
                        iterator.set_postfix({'Global Speed': f'{rate:.1f} vid/s'})
            # ---------------------------

    # Write remaining
    if results:
        df_res = pd.DataFrame(results)
        header = not os.path.exists(output_file)
        df_res.to_csv(output_file, mode='a', header=header, index=False)

    logger.info("Rank complete.")
    dist.barrier()
    if rank == 0:
        logger.info(f"All ranks finished. Results saved to {args.output_dir}/predictions_rank*.csv")

if __name__ == '__main__':
    main()