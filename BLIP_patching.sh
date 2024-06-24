#!/bin/bash
#SBATCH --time=64:00:00
#SBATCH -p 3090-gcondo --gres=gpu:1 
#SBATCH -n 16
#SBATCH --mem=120G
#SBATCH -J BLIP_patching_image_catt_head9
#SBATCH -o logs/BLIP_patching_image_catt_head9.out
#SBATCH -e logs/BLIP_patching_image_catt_head9.err

# Activate venv
# Set MASTER_PORT and MASTER_ADDRESS 
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR 
export TORCH_DISTRIBUTED_DEBUG=DETAIL

module load python/3.9.16s-x3wdtvt
source /users/mgolovan/data/mgolovan/llava/bin/activate

#CUDA_LAUNCH_BLOCKING=1  
# Run script. See run_model.py file for options. 

CUDA_LAUNCH_BLOCKING=1 python3 BLIP_patching.py --samples full --block_name text_encoder --kind crossattention_block --mode image --attn_head 9