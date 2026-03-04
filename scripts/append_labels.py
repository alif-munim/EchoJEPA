import os
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_FILE = "indices/master_index_18M.csv"
OUTPUT_FILE = "data/csv/inference_18m_vjepa_labeled.csv"
LABEL = "0"

def main():
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    print(f"Reading from: {INPUT_FILE}")
    print(f"Writing to:   {OUTPUT_FILE}")
    print(f"Action:       Filter masks/unmasked -> Append label '{LABEL}'")

    written_count = 0
    skipped_masks = 0
    skipped_unmasked = 0

    with open(INPUT_FILE, "r") as fin, open(OUTPUT_FILE, "w") as fout:
        # 1. Skip the CSV header ("s3_uri")
        header = fin.readline()
        
        # 2. Process line by line
        for line in tqdm(fin, desc="Processing", unit="lines"):
            uri = line.strip()
            
            if not uri:
                continue

            # --- FILTERS ---
            if uri.endswith("mask_visualization.mp4"):
                skipped_masks += 1
                continue
            
            if "/unmasked/" in uri:
                skipped_unmasked += 1
                continue

            # --- WRITE ---
            # Format: Exact URI + Space + Label
            fout.write(f"{uri} {LABEL}\n")
            written_count += 1

    print("\n" + "="*40)
    print("COMPLETE")
    print(f"✅ Lines Written:   {written_count}")
    print(f"🚫 Skipped Masks:    {skipped_masks}")
    print(f"🚫 Skipped Unmasked: {skipped_unmasked}")
    print(f"📂 Output File:      {OUTPUT_FILE}")
    print("="*40)

    # Preview
    print("\n--- Preview of Output ---")
    with open(OUTPUT_FILE, "r") as f:
        for _ in range(5):
            print(f.readline().strip())

if __name__ == "__main__":
    main()