import os
import csv

def create_annotations_csv(root_dir, output_file):
    """
    Traverses a directory to find all .mp4 files, creating a CSV file
    that maps each video to a dummy label. This version uses a SPACE
    as the delimiter to match the V-JEPA2 data loader's expectations.

    This function specifically avoids descending into directories named
    'unmasked' or 'png', and ignores files named 'mask_visualization.mp4'.

    Args:
        root_dir (str): The path to the root directory containing the video files.
        output_file (str): The path where the output CSV file will be saved.
    """
    # This list will store the rows for the CSV file.
    video_records = []
    print(f"Starting search in the directory: {os.path.abspath(root_dir)}")

    # os.walk is a generator that traverses a directory tree.
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # To prevent os.walk from going into 'unmasked' or 'png' directories,
        # we remove them from the 'dirnames' list in-place.
        if 'unmasked' in dirnames:
            dirnames.remove('unmasked')
        if 'png' in dirnames:
            dirnames.remove('png')

        # Now, iterate over the files in the current directory
        for filename in filenames:
            # Check if the file is an mp4 video and not the one to be ignored.
            if filename.endswith(".mp4") and filename != "mask_visualization.mp4":
                # Construct the full, absolute path to the video file.
                absolute_path = os.path.abspath(os.path.join(dirpath, filename))
                
                # Append the record to our list. The label is always 0.
                video_records.append([absolute_path, 0])

    # Check if any videos were found before trying to write the file.
    if not video_records:
        print("Warning: No .mp4 files were found in the specified directory.")
        return

    print(f"Found {len(video_records)} video files. Writing to {output_file}...")

    # Write the collected data to the CSV file.
    try:
        with open(output_file, 'w', newline='') as csvfile:
            # Create a CSV writer object, specifying a SPACE as the delimiter.
            writer = csv.writer(csvfile, delimiter=' ')
            # Write all the video records.
            writer.writerows(video_records)
        print(f"Successfully created annotations file: {os.path.abspath(output_file)}")
    except IOError as e:
        print(f"Error: Could not write to file {output_file}. Reason: {e}")


# This block allows the script to be run directly from the command line.
if __name__ == "__main__":
    # --- Configuration ---
    # This script assumes it is located in your 'vjepa2' project root,
    # and the video data is in a subdirectory named 'data'.
    data_directory = 'data'
    csv_output_file = 'annotations.csv' # The CSV will be saved in the same directory as the script.

    # --- Execution ---
    # Check if the data directory actually exists before running.
    if not os.path.isdir(data_directory):
        print(f"Error: The directory '{data_directory}' was not found.")
        print("Please make sure you are running this script from your 'vjepa2' project root,")
        print("and your video files are located inside the 'vjepa2/data/' directory.")
    else:
        create_annotations_csv(data_directory, csv_output_file)
