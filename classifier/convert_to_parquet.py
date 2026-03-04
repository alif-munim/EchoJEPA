import pandas as pd
import os

def csv_to_parquet(csv_path, parquet_path, chunksize=1_000_000):
    print(f"Converting {csv_path}...")
    
    # If the file exists, delete it first to avoid appending to old data
    if os.path.exists(parquet_path):
        os.remove(parquet_path)
        
    # Read and write in chunks
    for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize)):
        # Append to parquet file
        # engine='pyarrow' is generally faster
        if i == 0:
            chunk.to_parquet(parquet_path, engine='pyarrow', index=False)
        else:
            # Append mode requires existing file and tracking schema
            # It's often easier to write separate files or just list accumulation if memory allows.
            # But for simplicity, let's use fastparquet for appending OR just accumulate only if RAM permits.
            
            # SAFE METHOD: Append to a list if you have >32GB RAM, 
            # otherwise write to separate parquet files and read as a dataset.
            pass 
            
    # actually, for 18M rows, standard chunking append is tricky in pandas without pyarrow.
    # BETTER APPROACH:
    
    import pyarrow.csv as pv
    import pyarrow.parquet as pq

    # PyArrow reads CSVs much faster and with less memory overhead than Pandas
    table = pv.read_csv(csv_path)
    pq.write_table(table, parquet_path)
    print(f"Saved to {parquet_path}")

# Run conversion
base_path = "output"
# Convert Color
csv_to_parquet(f"{base_path}/color_inference_18m/master_predictions.csv", 
               f"{base_path}/color_inference_18m/predictions.parquet")

# Convert View
csv_to_parquet(f"{base_path}/view_inference_18m/master_predictions.csv", 
               f"{base_path}/view_inference_18m/predictions.parquet")