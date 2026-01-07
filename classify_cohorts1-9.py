#!/usr/bin/env python3
"""
classify_cohorts1-9.py - Classify unmasked PNG thumbnails from cohorts 1-9
Processes unmasked PNGs from CONVERTED_DATA/cohortX_1000/MRN/StudyID/unmasked/png/
Outputs CSV with columns: cohort, mrn, study_id, series_id, image_path, predicted_view, prob_a2c, prob_a3c, ...

Model architecture: EfficientNet-B2 backbone with 3 heads (binary gate, 9-way other, 4-way PSAX)
Output includes probabilities for all 13 view classes
"""

# CRITICAL: Set CPU-only mode BEFORE any imports (especially torch)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import csv

# ---------- Configuration ----------
BASE_CONVERTED_DIR = Path(os.getenv("BASE_CONVERTED_DIR", "/gpfs/data/whitney-lab/echo-FM/CONVERTED_DATA"))
CHECKPOINT_PATH = Path(os.getenv("CHECKPOINT_PATH", "/gpfs/data/whitney-lab/echo-FM/CODE/adapted_cri/p1_best.pth"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "/gpfs/data/whitney-lab/echo-FM/RESULTS/classification"))
PROCESS_MRN = os.getenv("PROCESS_MRN", None)  # If set, process only this MRN (format: cohortX_1000/MRN_ID)
OUTPUT_CSV_ENV = os.getenv("OUTPUT_CSV", None)  # If set, use this output path

COHORTS = ["cohort1_1000", "cohort2_1000", "cohort3_1000", "cohort4_1000", "cohort5_1000",
           "cohort6_1000", "cohort7_1000", "cohort8_1000", "cohort9_1000"]

# ---------- View Labels ----------
# Based on checkpoint: 9 "other" classes + 4 PSAX classes = 13 total views
VIEW = ["a2c", "a3c", "a4c", "a5c", "plax", "plax-d", "tee", "subcostal", "exclude",
        "psax-av", "psax-mv", "psax-ap", "psax-pm"]

N_OTHER = 9      # First 9 views are "other" (not PSAX)
PSAX_OFF = 9     # PSAX views start at index 9

# ---------- Model Definition ----------
class Net(nn.Module):
    """EfficientNet-B2 with binary gate + two classification heads"""
    def __init__(self, n_other=N_OTHER):
        super().__init__()
        base = models.efficientnet_b2(weights=None)
        
        f = base.classifier[1].in_features
        base.classifier = nn.Identity()
        self.b  = base
        self.vb = nn.Linear(f, 2)           # binary gate
        self.vo = nn.Linear(f, n_other)     # N-way "other" (9 in checkpoint)
        self.vs = nn.Linear(f, 4)           # 4-way PSAX subclasses
        
    def forward(self, x):
        f = self.b(x)
        return {"bin": self.vb(f), "oth": self.vo(f), "sub": self.vs(f)}

# ---------- Preprocessing (matching original code) ----------
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

# ---------- Prediction Function with Probabilities ----------
@torch.no_grad()
def predict_with_probs(out):
    """
    Predict view from model output dict and return probabilities for all classes
    
    Returns:
        pred_idx (int): Predicted class index
        probs (list): Probabilities for all 13 view classes
    """
    # Get gate probabilities
    gate_probs = F.softmax(out["bin"], dim=1)  # [batch, 2]
    p_other = gate_probs[:, 0]  # Probability of "other" (not PSAX)
    p_psax = gate_probs[:, 1]   # Probability of PSAX
    
    # Get probabilities for each head
    oth_probs = F.softmax(out["oth"], dim=1)  # [batch, 9]
    sub_probs = F.softmax(out["sub"], dim=1)  # [batch, 4]
    
    # Combine to get full probability distribution over all 13 classes
    # P(view_i) = P(gate) * P(view_i | gate)
    full_probs = torch.zeros(out["bin"].size(0), len(VIEW))
    
    # First 9 classes (other): multiply by p_other
    full_probs[:, :N_OTHER] = p_other.unsqueeze(1) * oth_probs
    
    # Last 4 classes (PSAX): multiply by p_psax
    full_probs[:, PSAX_OFF:] = p_psax.unsqueeze(1) * sub_probs
    
    # Get predicted class
    pred_idx = full_probs.argmax(1)
    
    # Return as lists for single image
    return pred_idx.item(), full_probs[0].tolist()

# ---------- Load model ----------
def load_model(checkpoint_path):
    """Load trained model from checkpoint"""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model = Net()
    
    # Load state dict (checkpoint format from original training code)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    elif isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    # else: assume it's already a direct state dict
    
    # Handle key name mismatches (checkpoint may use different naming)
    # Map: checkpoint keys → model keys
    key_mapping = {
        'backbone.': 'b.',           # backbone → b
        'view_bin.': 'vb.',          # view_bin → vb
        'view_oth.': 'vo.',          # view_oth → vo
        'view_sub.': 'vs.',          # view_sub → vs
    }
    
    # Check if we need to remap keys
    sample_key = next(iter(state_dict.keys()))
    needs_remapping = any(sample_key.startswith(old) for old in key_mapping.keys())
    
    if needs_remapping:
        print(f"Remapping checkpoint keys (original format detected)")
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            for old_prefix, new_prefix in key_mapping.items():
                if key.startswith(old_prefix):
                    new_key = key.replace(old_prefix, new_prefix, 1)
                    break
            # Skip keys that don't match our model (e.g., color, zoom, qual heads)
            if any(new_key.startswith(prefix) for prefix in ['b.', 'vb.', 'vo.', 'vs.']):
                new_state_dict[new_key] = value
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    model.eval()
    return model

# ---------- Classification ----------
def classify_image(model, image_path, device):
    """Classify a single PNG image and return predicted view + all probabilities"""
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            out = model(img_tensor)
            pred_idx, probs = predict_with_probs(out)
            predicted_view = VIEW[pred_idx]
        
        return predicted_view, probs
    
    except Exception as e:
        print(f"Error classifying {image_path}: {e}", file=sys.stderr)
        # Return error with zero probabilities
        return "error", [0.0] * len(VIEW)

# ---------- Main ----------
def main():
    print("=" * 60)
    print("Echo View Classification - Cohorts 1-9 (Unmasked)")
    print("=" * 60)
    print()
    
    # Debug: Show environment variables
    print(f"DEBUG: PROCESS_MRN = '{PROCESS_MRN}'")
    print(f"DEBUG: OUTPUT_CSV_ENV = '{OUTPUT_CSV_ENV}'")
    print()
    
    # Determine output CSV path
    if OUTPUT_CSV_ENV:
        output_csv = Path(OUTPUT_CSV_ENV)
    else:
        output_csv = RESULTS_DIR / "view_predictions_cohorts1-9_unmasked.csv"
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # Validate checkpoint
    if not CHECKPOINT_PATH.exists():
        print(f"ERROR: Checkpoint not found: {CHECKPOINT_PATH}")
        sys.exit(1)
    
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Base directory: {BASE_CONVERTED_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    
    if PROCESS_MRN:
        print(f"Processing single MRN: {PROCESS_MRN}")
    
    print(f"Output CSV: {output_csv}")
    print()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()
    
    # Load model
    print("Loading model...")
    model = load_model(CHECKPOINT_PATH)
    model.to(device)
    print("✓ Model loaded successfully")
    print()
    
    # Collect PNGs - either single MRN or all cohorts
    print("Searching for PNG files in unmasked/png/ directories...")
    print(f"  (This may take a moment on large filesystems...)")
    all_pngs = []
    
    if PROCESS_MRN:
        # Process single MRN (format: cohortX_1000/MRN_ID)
        mrn_path = BASE_CONVERTED_DIR / PROCESS_MRN
        
        if not mrn_path.exists():
            print(f"ERROR: MRN directory not found: {mrn_path}")
            sys.exit(1)
        
        print(f"  Searching in: {mrn_path}")
        # Find all PNGs in this MRN's studies (unmasked/png/)
        mrn_pngs = sorted(mrn_path.glob("*/unmasked/png/*.png"))
        print(f"  ✓ Found {len(mrn_pngs)} PNG files in {PROCESS_MRN}")
        all_pngs.extend(mrn_pngs)
        
    else:
        # Process all cohorts
        for cohort in COHORTS:
            cohort_dir = BASE_CONVERTED_DIR / cohort
            
            if not cohort_dir.exists():
                print(f"WARNING: Cohort directory not found: {cohort_dir}")
                continue
            
            # Find all PNGs in MRN/StudyID/unmasked/png/ directories
            png_pattern = f"{cohort}/*/*/unmasked/png/*.png"
            cohort_pngs = sorted(BASE_CONVERTED_DIR.glob(png_pattern))
            
            print(f"  {cohort}: {len(cohort_pngs)} PNG files")
            all_pngs.extend(cohort_pngs)
    
    print()
    print(f"Total PNG files to classify: {len(all_pngs)}")
    print()
    
    if len(all_pngs) == 0:
        print("ERROR: No PNG files found!")
        print()
        print("Expected structure:")
        print("  CONVERTED_DATA/")
        print("    cohort1_1000/")
        print("      <MRN>/")
        print("        <StudyID>/")
        print("          unmasked/")
        print("            png/")
        print("              *.png")
        sys.exit(1)
    
    # Classify and write to CSV
    print("Starting classification...")
    print(f"  Processing {len(all_pngs)} images...")
    print()
    
    results = []
    errors = 0
    
    for idx, png_path in enumerate(all_pngs, 1):
        if idx % 10 == 0 or idx == 1:
            print(f"  Progress: {idx}/{len(all_pngs)} ({idx*100//len(all_pngs)}%)", flush=True)
        
        # Extract metadata from path
        # Expected: .../CONVERTED_DATA/cohortX_1000/MRN/StudyID/unmasked/png/filename.png
        parts = png_path.parts
        
        try:
            # Find cohort index
            cohort_idx = None
            for i, part in enumerate(parts):
                if part in COHORTS:
                    cohort_idx = i
                    break
            
            if cohort_idx is None:
                print(f"WARNING: Cannot determine cohort for {png_path}", file=sys.stderr)
                cohort = "unknown"
                mrn = "unknown"
                study_id = "unknown"
            else:
                cohort = parts[cohort_idx]
                mrn = parts[cohort_idx + 1]
                study_id = parts[cohort_idx + 2]
            
            # Series ID from filename (e.g., 1.2.3.4.5.png → series_id)
            series_id = png_path.stem  # filename without .png
            
        except IndexError:
            print(f"WARNING: Cannot parse path: {png_path}", file=sys.stderr)
            cohort = mrn = study_id = series_id = "unknown"
        
        # Classify
        predicted_view, probs = classify_image(model, png_path, device)
        
        if predicted_view == "error":
            errors += 1
        
        # Build result row with probabilities
        result = {
            'cohort': cohort,
            'mrn': mrn,
            'study_id': study_id,
            'series_id': series_id,
            'image_path': str(png_path),
            'predicted_view': predicted_view
        }
        
        # Add probability for each view class
        for view_name, prob in zip(VIEW, probs):
            result[f'prob_{view_name}'] = f"{prob:.6f}"
        
        results.append(result)
    
    print()
    print("✓ Classification complete")
    print()
    
    # Write CSV
    print(f"Writing results to {output_csv}...")
    
    # Build fieldnames: base fields + probability columns
    prob_fieldnames = [f'prob_{view}' for view in VIEW]
    fieldnames = ['cohort', 'mrn', 'study_id', 'series_id', 'image_path', 'predicted_view'] + prob_fieldnames
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(results)
    
    print("✓ CSV written successfully")
    print()
    
    # Summary statistics
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print()
    print(f"Total images classified: {len(results)}")
    print(f"Errors encountered: {errors}")
    print()
    
    # View distribution
    from collections import Counter
    view_counts = Counter(r['predicted_view'] for r in results)
    
    print("View Distribution:")
    for view, count in view_counts.most_common():
        percentage = count * 100 / len(results)
        print(f"  {view:15s}: {count:6d} ({percentage:5.1f}%)")
    print()
    
    # Per-cohort distribution
    if not PROCESS_MRN:
        print("Per-Cohort Breakdown:")
        for cohort in COHORTS:
            cohort_results = [r for r in results if r['cohort'] == cohort]
            if cohort_results:
                print(f"  {cohort}: {len(cohort_results)} images")
        print()
    
    print("=" * 60)
    print("Classification complete!")
    print("=" * 60)
    print()
    print(f"Results saved to: {output_csv}")
    print()
    print("CSV includes:")
    print("  - Predicted view (highest probability)")
    print("  - Probabilities for all 13 view classes")
    print()
    print("View classes and probability columns:")
    for i, view in enumerate(VIEW):
        print(f"  {i+1:2d}. {view:12s} → prob_{view}")
    print()

if __name__ == "__main__":
    main()
