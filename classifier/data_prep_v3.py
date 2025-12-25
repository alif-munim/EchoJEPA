#!/usr/bin/env python
# coding: utf-8

# In[3]:


# ====== 1. Imports + config + helpers ======
import os
import glob
import random
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm.notebook import tqdm

# ---------------- CONFIG ----------------
# Root directory containing your job_* folders (unchanged)
SOURCE_ROOT = os.path.join(".", "unmasked")

# Output CSV (unchanged)
OUTPUT_CSV = "labels_masked_inplace.csv"

# Reproducibility
SEED = 42

# Data splits (unchanged)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# Remaining 15% is test

# ------------- CLASS MAPPING -------------
CLASS_MAP = {
    "A2C": "A2C", "A3C": "A3C", "A4C": "A4C", "A5C": "A5C",
    "PLAX": "PLAX", "SSN": "SSN",
    "PSAX-AV": "PSAX-AV", "PSAX-PM": "PSAX-PM",
    "PSAX-MV": "PSAX-MV", "PSAX-AP": "PSAX-AP",
    "Subcostal": "Subcostal", "SBC-4C": "Subcostal",
    "SBC-Vessel": "Subcostal", "SBC-SA": "Subcostal",
    "TEE": "TEE",
    "Exclude": "Exclude",
    # any other labels not listed here will be skipped
}

def map_label(raw_label):
    return CLASS_MAP.get(raw_label, None)

# ------------- CLEANUP OLD MASKS -------------
def cleanup_old_masks(root_dir):
    print(f"Scanning {root_dir} for old masked images to delete...")
    old_masks = glob.glob(os.path.join(root_dir, "**", "*_masked.jpg"),
                          recursive=True)
    if old_masks:
        print(f"Found {len(old_masks)} old masked images. Deleting...")
        for f in tqdm(old_masks, desc="Deleting old masks"):
            try:
                os.remove(f)
            except OSError as e:
                print(f"Error deleting {f}: {e}")
        print("Cleanup complete.")
    else:
        print("No old masked images found. Clean start.")

cleanup_old_masks(SOURCE_ROOT)

# ------------- SECTOR MASK -------------
def apply_mask(img_array):
    """
    Applies the 'Voxel Cone' sector mask.
    Expects RGB input (H, W, 3).
    """
    h, w = img_array.shape[:2]
    mask = np.ones((h, w), dtype=np.uint8)

    def draw_box(x, y, box_w, box_h):
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(w, int(x + box_w)), min(h, int(y + box_h))
        mask[y1:y2, x1:x2] = 0

    # 1. Top & Bottom bars
    draw_box(0, 0, w, h * (35/480))
    draw_box(0, h - h*(90/480), w, h * (120/480))

    # 2. Left-side stack
    draw_box(0, 0, w*(60/640), h)
    draw_box(0, 0, w*(90/640), h*0.55)
    draw_box(0, 0, w*(77/640), h*0.62)
    draw_box(0, 0, w*(130/640), h*0.3)
    draw_box(0, 0, w*(150/640), h*0.26)
    draw_box(0, 0, w*(220/640), h*0.20)
    draw_box(0, h*0.72, w*(105/640), h*0.3)

    # 3. Right-side stack
    draw_box(w - w*(220/640), 0, w*(220/640), h*0.20)
    draw_box(w - w*(145/640), 0, w*(120/640), h*0.49)
    draw_box(w - w*(130/640), 0, w*(120/640), h*0.51)
    draw_box(w - w*(115/640), 0, w*(120/640), h*0.53)
    draw_box(w - w*(90/640), 0, w*(120/640), h)
    draw_box(w - w*(105/640), h*0.68, w*(105/640), h*0.3)
    draw_box(w - w*(115/640), h*0.72, w*(105/640), h*0.3)

    if img_array.ndim == 3:
        return img_array * mask[:, :, None]
    else:
        return img_array * mask

# ------------- SINGLE-IMAGE MASKING -------------
def mask_image(source_path):
    """
    Given an original JPG path, create the corresponding _masked.jpg
    if it does not already exist, and return the masked path.
    """
    base, ext = os.path.splitext(source_path)
    masked_path = f"{base}_masked{ext}"

    if os.path.exists(masked_path):
        return masked_path

    with Image.open(source_path) as img:
        img = img.convert("RGB")
        img_arr = np.array(img)
        masked_arr = apply_mask(img_arr)
        Image.fromarray(masked_arr).save(masked_path)

    return masked_path


# In[4]:


# ====== 2. Parse XML + generate masked images + CSV rows ======

def parse_annotations(xml_path, job_id, job_folder_path):
    if not os.path.exists(xml_path):
        return []

    tree = ET.parse(xml_path)
    root = tree.getroot()
    extracted_data = []

    for image in root.findall('image'):
        rel_path = image.get('name')

        # Locate original file
        full_source_path = os.path.join(job_folder_path, "images", rel_path)
        if not os.path.exists(full_source_path):
            full_source_path = os.path.join(job_folder_path, rel_path)
            if not os.path.exists(full_source_path):
                continue

        target_label = None
        
        for tag in image.findall('tag'):
            tag_label = tag.get('label')  # This gets "TTE", "TEE", or "Exclude"

            if tag_label == 'TTE':
                # Look for the specific View attribute for TTE
                for attr in tag.findall('attribute'):
                    if attr.get('name') == 'View':
                        target_label = map_label(attr.text)
                        break
            
            elif tag_label in ['TEE', 'Exclude']:
                # For TEE and Exclude, the label itself is the target
                target_label = map_label(tag_label)

            if target_label:
                break # Stop looking once we find a valid label

        if target_label:
            extracted_data.append({
                "source_path": full_source_path,
                "label": target_label,
                "job_id": job_id,
            })

    return extracted_data


# --- Gather annotated source files (unchanged behaviour) ---
print("Parsing XML annotations...")
all_source_files = []
job_folders = sorted(glob.glob(os.path.join(SOURCE_ROOT, "job_*")))

for job_folder in tqdm(job_folders, desc="Scanning jobs"):
    job_id = job_folder.split("_")[-1]
    xml_file = os.path.join(job_folder, "annotations.xml")
    all_source_files.extend(parse_annotations(xml_file, job_id, job_folder))

print(f"Found {len(all_source_files)} valid annotated images.")

# --- Mask annotated images and build CSV rows (unchanged semantics) ---
final_csv_data = []
print("Generating RGB masked images for annotated files...")

for item in tqdm(all_source_files, desc="Masking annotated"):
    try:
        masked_path = mask_image(item["source_path"])
        final_csv_data.append({
            "filename": masked_path,
            "label": item["label"],
            "job_id": item["job_id"],
        })
    except Exception as e:
        print(f"Error processing {item['source_path']}: {e}")

print(f"Done! Created/verified {len(final_csv_data)} masked images for annotated files.")

# --- NEW: ensure *all* JPGs in SOURCE_ROOT have a _masked copy ---
print("\nEnsuring all JPGs under SOURCE_ROOT have a corresponding _masked.jpg...")
all_jpgs = glob.glob(os.path.join(SOURCE_ROOT, "**", "*.jpg"), recursive=True)

to_process = [p for p in all_jpgs if not p.endswith("_masked.jpg")]
print(f"Found {len(to_process)} original JPGs (excluding existing _masked).")

for p in tqdm(to_process, desc="Masking all JPGs"):
    try:
        mask_image(p)
    except Exception as e:
        print(f"Error masking {p}: {e}")

print("All JPGs now have corresponding _masked.jpg files.")


# In[5]:


# ====== 3. Train/val/test split by job_id + save CSV ======

# 1. Unique jobs
unique_jobs = sorted(set(x["job_id"] for x in final_csv_data))
random.seed(SEED)
random.shuffle(unique_jobs)

# 2. Calculate job-level splits
n_jobs = len(unique_jobs)
n_train = int(n_jobs * TRAIN_RATIO)
n_val = int(n_jobs * VAL_RATIO)

train_jobs = set(unique_jobs[:n_train])
val_jobs = set(unique_jobs[n_train:n_train + n_val])
# remaining jobs -> test

# 3. Assign split to each row
for row in final_csv_data:
    if row["job_id"] in train_jobs:
        row["split"] = "train"
    elif row["job_id"] in val_jobs:
        row["split"] = "val"
    else:
        row["split"] = "test"

# 4. Save CSV
df = pd.DataFrame(final_csv_data)
df.to_csv(OUTPUT_CSV, index=False)

print(f"Success! Labels saved to {OUTPUT_CSV}")
print("-" * 30)
print(df["split"].value_counts())
print("-" * 30)
print("Example filename:", df.iloc[0]["filename"])


# In[6]:


# ====== 4. Quick visual check of masked images ======
import matplotlib.pyplot as plt

OUTPUT_CSV = "labels_masked_inplace.csv"
NUM_TO_VISUALIZE = 8

print("\n--- Final masking verification ---")

if not os.path.exists(OUTPUT_CSV):
    print(f"Error: Final CSV '{OUTPUT_CSV}' not found.")
else:
    df = pd.read_csv(OUTPUT_CSV)

    if len(df) <= NUM_TO_VISUALIZE:
        sample_df = df
        print(f"Only {len(df)} rows in CSV. Displaying all.")
    else:
        # different random subset each time
        sample_df = df.sample(n=NUM_TO_VISUALIZE)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, (_, row) in enumerate(sample_df.iterrows()):
        img_path = row["filename"]
        label = row["label"]

        try:
            img = Image.open(img_path)
            img_data = np.array(img)

            cmap = "gray" if img_data.ndim == 2 else None
            axes[i].imshow(img_data, cmap=cmap)
            ch = img_data.shape[2] if img_data.ndim == 3 else 1
            axes[i].set_title(f"{label} ({ch} ch)")
            axes[i].axis("off")
        except Exception as e:
            axes[i].set_title("Load error")
            axes[i].axis("off")
            print(f"Could not load image {img_path}: {e}")

    for j in range(len(sample_df), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
    print("Masking visualization complete.")


# In[7]:


# ====== Class distribution across splits ======
import matplotlib.pyplot as plt

OUTPUT_CSV = "labels_masked_inplace.csv"

df = pd.read_csv(OUTPUT_CSV)

# Raw counts by split + label
print("Counts by label and split:")
counts = pd.crosstab(df["label"], df["split"])
display(counts)

# Percentage within each split
print("\nRow-normalized (percentage within each split):")
percent = counts.div(counts.sum(axis=0), axis=1) * 100
display(percent.round(2))

# --- Bar plot: counts per class per split ---
plt.figure(figsize=(12, 6))
counts.plot(kind="bar")
plt.xlabel("Label")
plt.ylabel("Count")
plt.title("Class distribution by split")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# --- Optional: stacked bar of split composition per class ---
plt.figure(figsize=(12, 6))
counts.plot(kind="bar", stacked=True)
plt.xlabel("Label")
plt.ylabel("Count")
plt.title("Class distribution (stacked by split)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# In[8]:


import pandas as pd

p = "labels_patient_split_mp4.csv"
df = pd.read_csv(p)

# Your header has duplicate "split". Rename duplicates to make them selectable.
cols = []
seen = {}
for c in df.columns:
    k = c
    if k in seen:
        seen[k] += 1
        k = f"{c}_{seen[c]}"
    else:
        seen[k] = 0
    cols.append(k)
df.columns = cols

# pick the first split column (usually the one you want)
split_col = "split" if "split" in df.columns else [c for c in df.columns if c.startswith("split")][0]

counts = (
    df.groupby([split_col, "label"])
      .size()
      .rename("count")
      .reset_index()
      .sort_values([split_col, "count"], ascending=[True, False])
)

# add per-split percentages
counts["pct_within_split"] = counts["count"] / counts.groupby(split_col)["count"].transform("sum") * 100

print("Columns:", list(df.columns))
print("\nCounts:")
print(counts.to_string(index=False))

print("\nTotals per split:")
print(df.groupby(split_col).size().rename("total").to_string())


# # Color

# In[ ]:




