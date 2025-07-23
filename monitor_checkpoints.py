import os
import time
import boto3
import logging
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

# --- Configuration ---
# The local directory where your training script saves checkpoints.
CHECKPOINT_DIR = "/home/sagemaker-user/user-default-efs/vjepa2/checkpoints/anneal/64.8.vitg16-384px-64f"
# The base S3 URI where the checkpoints should be uploaded.
S3_URI = "s3://echodata25/vjepa2/checkpoints-0702/"
# How often to check for new files (in seconds).
POLL_INTERVAL_SECONDS = 60
# The maximum number of epoch-specific checkpoints (e*.pt) to keep locally.
MAX_LOCAL_CHECKPOINTS = 2
# --- End Configuration ---

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def get_epoch_from_filename(filename):
    """Extracts the epoch number from a checkpoint filename like 'e42.pt'."""
    if filename.startswith('e') and filename.endswith('.pt'):
        try:
            return int(filename[1:-3])
        except (ValueError, IndexError):
            return -1
    return -1

def upload_to_s3(local_path, s3_uri):
    """
    Uploads a single file to S3 using a managed transfer for large files.

    Args:
        local_path (str): The full path to the local file.
        s3_uri (str): The full S3 URI for the destination.

    Returns:
        bool: True if upload was successful, False otherwise.
    """
    try:
        s3_client = boto3.client("s3")
        bucket, key = s3_uri.replace("s3://", "").split("/", 1)
        
        logger.info(f"Uploading '{os.path.basename(local_path)}' to s3://{bucket}/{key}...")
        s3_client.upload_file(local_path, bucket, key)
        logger.info(f"Successfully uploaded to s3://{bucket}/{key}")
        return True
    except (NoCredentialsError, PartialCredentialsError):
        logger.error("S3 credentials not found. Please configure your AWS credentials.")
        return False
    except Exception as e:
        logger.error(f"Failed to upload {os.path.basename(local_path)} to S3. Error: {e}")
        return False

def prune_local_checkpoints(directory, max_to_keep):
    """
    Deletes the oldest epoch-specific checkpoints in a folder.
    
    Args:
        directory (str): The path to the checkpoint folder.
        max_to_keep (int): The number of recent checkpoints to keep.
    """
    try:
        # 1. Get all epoch-specific checkpoints (e.g., 'e50.pt')
        epoch_checkpoints = [f for f in os.listdir(directory) if get_epoch_from_filename(f) != -1]

        # 2. If we have more checkpoints than we want to keep...
        if len(epoch_checkpoints) > max_to_keep:
            # 3. Sort them by epoch number (oldest first)
            epoch_checkpoints.sort(key=get_epoch_from_filename)

            # 4. Determine which ones to delete
            checkpoints_to_delete = epoch_checkpoints[:-max_to_keep]
            logger.info(f"Pruning local checkpoints. Keeping {max_to_keep}, deleting {len(checkpoints_to_delete)}.")

            for ckpt_name in checkpoints_to_delete:
                full_path = os.path.join(directory, ckpt_name)
                try:
                    os.remove(full_path)
                    logger.info(f"Deleted old checkpoint: {ckpt_name}")
                except Exception as e:
                    logger.error(f"Failed to delete old checkpoint {full_path}: {e}")
    except Exception as e:
        logger.error(f"Failed to prune checkpoints in folder {directory}: {e}")

def main():
    """
    Main monitoring loop.
    """
    if not os.path.isdir(CHECKPOINT_DIR):
        logger.error(f"Checkpoint directory not found: {CHECKPOINT_DIR}")
        return

    logger.info(f"Monitoring directory: {CHECKPOINT_DIR}")
    logger.info(f"Uploading new checkpoints to: {S3_URI}")
    logger.info(f"Keeping the last {MAX_LOCAL_CHECKPOINTS} local checkpoints.")

    # A set to keep track of files that have already been uploaded
    uploaded_files = set()

    while True:
        try:
            # Get the current list of files in the directory
            current_files = set(os.listdir(CHECKPOINT_DIR))
            
            # Find new files that haven't been uploaded yet
            new_files = current_files - uploaded_files

            if new_files:
                # Sort to process in a predictable order, though not strictly necessary
                for filename in sorted(list(new_files)):
                    if filename.endswith(".pt"):
                        local_path = os.path.join(CHECKPOINT_DIR, filename)
                        
                        # Construct the full destination S3 path
                        s3_destination_uri = os.path.join(S3_URI, os.path.basename(CHECKPOINT_DIR), filename)
                        
                        # Upload the new file
                        success = upload_to_s3(local_path, s3_destination_uri)
                        
                        if success:
                            # If upload was successful, add to our set of processed files
                            uploaded_files.add(filename)
                            
                            # After uploading, prune the old checkpoints
                            if get_epoch_from_filename(filename) != -1: # Prune only after an epoch save
                                prune_local_checkpoints(CHECKPOINT_DIR, MAX_LOCAL_CHECKPOINTS)

            else:
                logger.debug("No new checkpoints found. Waiting...")

            # Wait for the next poll interval
            time.sleep(POLL_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            logger.info("Monitor script stopped by user.")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
