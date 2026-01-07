#!/usr/bin/env python3
"""
Improved DICOM pixel extraction with comprehensive error handling and logging.
This is a replacement for the inline Python script in convert_one().
"""
import os
import sys
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import convert_color_space

def to_uint8(img):
    """Convert image array to uint8."""
    if img.dtype == np.uint8:
        return img
    imin, imax = img.min(), img.max()
    if imax <= imin:
        return np.zeros_like(img, dtype=np.uint8)
    return ((img.astype(np.float32) - imin) / (imax - imin) * 255.0).astype(np.uint8)

def to_gray3(fr, pi):
    """Convert frame to grayscale with 3 identical channels"""
    if fr.ndim == 2:
        g = fr
    elif fr.shape[-1] == 3:
        if pi and pi.startswith('YBR'):
            # If YBR, first channel is luminance (Y)
            g = fr[..., 0]
        else:
            # ITU-R BT.709 luma coefficients for RGB
            g = 0.2126 * fr[..., 0] + 0.7152 * fr[..., 1] + 0.0722 * fr[..., 2]
    else:
        g = fr[..., 0]
    g8 = to_uint8(g)
    return np.stack([g8, g8, g8], axis=-1)

def extract_pixels_to_raw(dcm_path, raw_path):
    """
    Extract pixel data from DICOM and write as raw RGB.
    Returns: (success: bool, message: str, stats: dict)
    """
    stats = {'frames': 0, 'bytes_written': 0, 'shape': None, 'dtype': None}
    
    try:
        # Load DICOM
        ds = pydicom.dcmread(dcm_path, force=True)
        sys.stderr.write(f"[INFO] Loaded DICOM: {dcm_path}\n")
        
        # Get pixel array
        arr = ds.pixel_array
        stats['shape'] = arr.shape
        stats['dtype'] = str(arr.dtype)
        
        pi = getattr(ds, 'PhotometricInterpretation', '') or ''
        nf = int(getattr(ds, 'NumberOfFrames', 1) or 1)
        
        sys.stderr.write(f"[INFO] shape={arr.shape} dtype={arr.dtype} PI={pi} NF={nf}\n")
        
        # Heuristic guard: only convert YBR→RGB if it still *looks* like YCbCr (Cb/Cr near 128)
        if pi.startswith('YBR') and arr.ndim >= 3 and arr.shape[-1] == 3:
            try:
                # Check if chroma channels are near neutral (128) - indicates true YBR
                m0, m1, m2 = [float(arr[..., k].mean()) for k in range(3)]
                if abs(m1 - 128) < 40 and abs(m2 - 128) < 40:  # chroma near neutral
                    arr = convert_color_space(arr, pi, 'RGB')
                    sys.stderr.write(f"[INFO] Converted {pi} → RGB (applied)\n")
                else:
                    sys.stderr.write(f"[INFO] {pi} but looks like RGB already (means: {m0:.1f}, {m1:.1f}, {m2:.1f}); skip convert\n")
            except Exception as e:
                sys.stderr.write(f"[WARN] Color space conversion skipped: {e}\n")
        
        # Normalize to [F, H, W, 3] format
        frames = []
        
        if arr.ndim == 2:  # Single grayscale
            frames = [np.stack([arr, arr, arr], axis=-1)]
            sys.stderr.write(f"[INFO] Format: single grayscale [H, W] → [1, H, W, 3]\n")
        
        elif arr.ndim == 3:
            if arr.shape[-1] == 3:  # Single RGB
                frames = [arr]
                sys.stderr.write(f"[INFO] Format: single RGB [H, W, 3]\n")
            elif arr.shape[0] == nf and nf > 1:  # Multi-frame grayscale
                frames = [np.stack([f, f, f], axis=-1) for f in arr]
                sys.stderr.write(f"[INFO] Format: multi-frame grayscale [F, H, W] → [{len(frames)}, H, W, 3]\n")
            else:  # Ambiguous, treat as single grayscale
                frames = [np.stack([arr, arr, arr], axis=-1)]
                sys.stderr.write(f"[INFO] Format: ambiguous 3D, treating as single grayscale\n")
        
        elif arr.ndim == 4:
            if arr.shape[-1] == 3:  # Multi-frame RGB
                frames = [f for f in arr]
                sys.stderr.write(f"[INFO] Format: multi-frame RGB [F, H, W, 3]\n")
            else:  # Multi-frame with channels
                frames = [np.stack([f[..., 0], f[..., 0], f[..., 0]], axis=-1) for f in arr]
                sys.stderr.write(f"[INFO] Format: multi-frame [{arr.shape[0]}, H, W, {arr.shape[-1]}] → using channel 0\n")
        
        else:
            raise RuntimeError(f"Unsupported array dimensionality: {arr.ndim}D with shape {arr.shape}")
        
        if len(frames) == 0:
            raise RuntimeError("No frames extracted from pixel array")
        
        stats['frames'] = len(frames)
        
        # Apply grayscale conversion if FORCE_GRAY is enabled
        force_gray = os.getenv("FORCE_GRAY", "1") == "1"
        if force_gray:
            sys.stderr.write(f"[INFO] Converting to grayscale (FORCE_GRAY=1)\n")
            frames = [to_gray3(f, pi) for f in frames]
        else:
            sys.stderr.write(f"[INFO] Preserving color (FORCE_GRAY=0)\n")
            frames = [to_uint8(f) if f.dtype != np.uint8 else f for f in frames]
        
        # Write raw RGB data
        sys.stderr.write(f"[INFO] Writing {len(frames)} frame(s) to {raw_path}\n")
        
        total_bytes = 0
        with open(raw_path, 'wb') as f:
            for i, frame in enumerate(frames):
                frame_uint8 = to_uint8(frame)
                frame_bytes = np.ascontiguousarray(frame_uint8).tobytes()
                f.write(frame_bytes)
                total_bytes += len(frame_bytes)
                
                if i < 3:  # Log first few frames
                    sys.stderr.write(f"[INFO] Frame {i}: shape={frame_uint8.shape}, bytes={len(frame_bytes)}\n")
            
            # Force flush to OS and sync to disk
            f.flush()
            os.fsync(f.fileno())
        
        # Sync parent directory to ensure directory entry is visible
        import time
        try:
            dir_fd = os.open(os.path.dirname(raw_path), os.O_RDONLY)
            os.fsync(dir_fd)
            os.close(dir_fd)
        except Exception:
            pass  # Best effort
        
        stats['bytes_written'] = total_bytes
        
        # Verify file with retry for GPFS visibility
        max_retries = 50
        for attempt in range(max_retries):
            try:
                file_size = os.stat(raw_path).st_size
                if file_size > 0:
                    break
            except FileNotFoundError:
                if attempt < max_retries - 1:
                    time.sleep(0.05)  # 50ms between retries
                    continue
                else:
                    raise RuntimeError(f"File not visible after {max_retries} retries")
        
        if file_size == 0:
            raise RuntimeError(f"Raw file is empty: {raw_path}")
        
        if file_size != total_bytes:
            sys.stderr.write(f"[WARN] File size mismatch: wrote {total_bytes} bytes, file is {file_size} bytes\n")
        
        sys.stderr.write(f"[SUCCESS] Wrote {total_bytes} bytes to {raw_path} (file size: {file_size})\n")
        
        return True, "Success", stats
    
    except Exception as e:
        error_msg = f"Extraction failed: {type(e).__name__}: {str(e)}"
        sys.stderr.write(f"[ERROR] {error_msg}\n")
        import traceback
        traceback.print_exc(file=sys.stderr)
        return False, error_msg, stats


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 extract_pixels.py <input.dcm> <output.raw>", file=sys.stderr)
        sys.exit(2)
    
    dcm_path = sys.argv[1]
    raw_path = sys.argv[2]
    
    # Set threading environment
    os.environ.setdefault("OPENJPEG_NUM_THREADS", "2")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    
    success, message, stats = extract_pixels_to_raw(dcm_path, raw_path)
    
    if success:
        print(f"SUCCESS: {message}", file=sys.stderr)
        print(f"STATS: frames={stats['frames']}, bytes={stats['bytes_written']}, shape={stats['shape']}, dtype={stats['dtype']}", file=sys.stderr)
        sys.exit(0)
    else:
        print(f"FAILED: {message}", file=sys.stderr)
        sys.exit(1)
