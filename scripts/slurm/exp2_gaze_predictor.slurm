#!/bin/bash -e
# ============================================================
# Experiment 2: Train Gaze Fixation Duration Predictor
# Goal: BERT-base fine-tuned on GECO, Spearman r ~ 0.35-0.50
# GPU: 1x H100 (80GB) | Time: ~30 minutes
# Owner: Siddhant | Week 1-2
# ============================================================
#SBATCH --job-name=exp2-gaze-pred
#SBATCH --output=logs/exp2_%j.out
#SBATCH --error=logs/exp2_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB
#SBATCH --partition=h100_tandon
#SBATCH --gres=gpu:h100:1
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rs9174@nyu.edu
#SBATCH --account=torch_pr_xxx_yyy

module purge && module load anaconda3/2025.06
conda activate $SCRATCH/envs/gazediffuse

cd $SCRATCH/gaze-diffuse
mkdir -p logs checkpoints/gaze_predictor

echo "=== Experiment 2: Gaze Predictor Training ==="
echo "Start: $(date)"
nvidia-smi

python -m src.gaze_predictor \
    --data_dir data/geco \
    --output_dir checkpoints/gaze_predictor \
    --epochs 3 \
    --batch_size 32 \
    --lr 2e-5 \
    --bert_model bert-base-uncased

echo "End: $(date)"
echo "=== Experiment 2 Complete ==="
