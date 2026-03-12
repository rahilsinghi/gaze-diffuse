#!/bin/bash -e
# ============================================================
# Experiment 1: MDLM Baseline Perplexity Evaluation
# Goal: Reproduce MDLM Gen. PPL on OpenWebText (~22.1)
# GPU: 1x RTX8000 (48GB) | Time: ~2 hours
# Owner: Rahil | Week 1
# ============================================================
#SBATCH --job-name=exp1-mdlm-ppl
#SBATCH --output=logs/exp1_%j.out
#SBATCH --error=logs/exp1_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rs9174@nyu.edu
#SBATCH --account=torch_pr_111_general

module purge && module load anaconda3/2024.02
conda activate $SCRATCH/envs/gazediffuse

cd $SCRATCH/gaze-diffuse
mkdir -p logs results

echo "=== Experiment 1: MDLM Baseline PPL ==="
echo "Start: $(date)"
nvidia-smi

python submodules/mdlm/main.py mode=ppl_eval \
    eval.checkpoint_path=checkpoints/mdlm-owt \
    data=openwebtext-split \
    parameterization=subs \
    backbone=dit \
    model.length=1024 \
    loader.eval_batch_size=4 \
    +wandb.offline=true

echo "End: $(date)"
echo "=== Experiment 1 Complete ==="
