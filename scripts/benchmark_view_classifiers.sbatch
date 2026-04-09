#!/bin/bash
#SBATCH --job-name=bench-view-cls
#SBATCH --partition=ml-p5-48xlarge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/opt/vjepa2/benchmark_view_cls_%j.out
#SBATCH --nodelist=ip-10-0-50-184
#SBATCH --exclusive

set -eo pipefail

cd /opt/vjepa2
source /opt/vjepa2-312/bin/activate
set -u

# cuBLAS bf16 fix for H100
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Download classifiers from S3
mkdir -p /tmp/classifiers
echo "Downloading classifiers from S3..."
aws s3 cp s3://echodata25/neurips/classifiers/view_convnext_small_336px.pt /tmp/classifiers/ --quiet
aws s3 cp s3://echodata25/neurips/classifiers/convnext_view_finetuned_chicago_best.pt /tmp/classifiers/ --quiet
echo "Downloads complete."

# Run benchmark
python3 classifier/benchmark_view.py \
  --checkpoints /tmp/classifiers/view_convnext_small_336px.pt /tmp/classifiers/convnext_view_finetuned_chicago_best.pt \
  --names "UHN-only" "Chicago-finetuned" \
  --test_csv data/csv/uhn_views_22k_test.csv \
  --img_size 336 --batch_size 64 --num_workers 16

echo "Benchmark complete."
