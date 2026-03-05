import argparse
import os
import subprocess
import glob
from multiprocessing import Pool, cpu_count
from pathlib import Path
import tqdm

def resize_video(args):
    """
    Worker function to resize a single video using ffmpeg.
    """
    input_path, output_path, size = args
    
    # Skip if output already exists (resume capability)
    if os.path.exists(output_path):
        return
        
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # FFMPEG Command:
    # -y: Overwrite output
    # -i: Input file
    # -vf: Video filter (scale)
    #      scale=W:H:force_original_aspect_ratio=decrease (fits inside box)
    #      pad=W:H:(ow-iw)/2:(oh-ih)/2 (pads with black to make exact square)
    # -c:v libx264: Use H.264 codec
    # -crf 18: High quality
    # -preset fast: Balance speed/compression
    # -an: Remove audio (echo doesn't usually have audio, saves space)
    
    cmd = [
        'ffmpeg', '-y', '-v', 'error', '-i', input_path,
        '-vf', f'scale={size}:{size}:force_original_aspect_ratio=decrease,pad={size}:{size}:(ow-iw)/2:(oh-ih)/2',
        '-c:v', 'libx264', '-crf', '18', '-preset', 'fast', '-an',
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error processing {input_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Resize video dataset preserving directory structure.")
    parser.add_argument("--input_dir", type=str, required=True, help="Root directory of original videos")
    parser.add_argument("--output_dir", type=str, required=True, help="Root directory for resized videos")
    parser.add_argument("--size", type=int, default=224, help="Target resolution (e.g., 224, 112)")
    parser.add_argument("--workers", type=int, default=cpu_count(), help="Number of parallel workers")
    
    args = parser.parse_args()

    print(f"Scanning files in {args.input_dir}...")
    
    # Recursive search for all mp4 files
    # Using pathlib for robust path handling
    src_path = Path(args.input_dir)
    dest_path = Path(args.output_dir)
    
    video_files = list(src_path.rglob("*.mp4"))
    print(f"Found {len(video_files)} videos.")

    # Prepare tasks
    tasks = []
    for f in video_files:
        # Calculate relative path to maintain structure
        rel_path = f.relative_to(src_path)
        out_f = dest_path / rel_path
        tasks.append((str(f), str(out_f), args.size))

    print(f"Resizing to {args.size}x{args.size} with {args.workers} workers...")
    
    # Run parallel processing
    with Pool(args.workers) as pool:
        # Wrapper for tqdm to show progress bar
        list(tqdm.tqdm(pool.imap_unordered(resize_video, tasks), total=len(tasks)))
        
    print("Done!")

if __name__ == "__main__":
    main()