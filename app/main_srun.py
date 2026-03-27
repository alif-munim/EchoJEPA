# Multi-node launcher for srun. Each srun task = one GPU = one DDP rank.
#
# Usage (in sbatch script):
#   export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1)
#   export MASTER_PORT=37129
#   srun --ntasks-per-node=8 python -m app.main_srun --fname config.yaml
#
# Differences from app.main (single-node):
#   - main.py spawns mp.Process per GPU and sets CUDA_VISIBLE_DEVICES itself
#   - main_srun.py is called once per srun task; SLURM manages processes
#   - GPU assignment via SLURM_LOCALID, rank/world_size via SLURM_PROCID/SLURM_NTASKS

import argparse
import logging
import os
import pprint
from pathlib import Path

import yaml

from app.scaffold import main as app_main
from src.utils.distributed import init_distributed
from src.utils.logging import get_logger

parser = argparse.ArgumentParser()
parser.add_argument("--fname", type=str, required=True, help="path to config YAML")


def main():
    args = parser.parse_args()

    # Each srun task gets one GPU via SLURM_LOCALID (0-7 on each node)
    local_rank = os.environ.get("SLURM_LOCALID", "0")
    os.environ["CUDA_VISIBLE_DEVICES"] = local_rank

    logger = get_logger(force=True)
    global_rank = int(os.environ.get("SLURM_PROCID", "0"))
    if global_rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f"main_srun: global_rank={global_rank}, local_rank={local_rank}, "
                f"MASTER_ADDR={os.environ.get('MASTER_ADDR')}, "
                f"MASTER_PORT={os.environ.get('MASTER_PORT')}")

    # Load config
    with open(args.fname, "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    if global_rank == 0:
        pprint.PrettyPrinter(indent=4).pprint(params)
        folder = params.get("folder", ".")
        Path(folder).mkdir(parents=True, exist_ok=True)
        params_path = os.path.join(folder, "params-pretrain.yaml")
        with open(params_path, "w") as f:
            yaml.dump(params, f)

    # Init distributed — reads SLURM_NTASKS and SLURM_PROCID
    world_size, rank = init_distributed()
    logger.info(f"Running... (rank: {rank}/{world_size})")

    app_main(params["app"], args=params)


if __name__ == "__main__":
    main()
