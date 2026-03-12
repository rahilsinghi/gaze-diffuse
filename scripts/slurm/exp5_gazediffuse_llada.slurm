#!/bin/bash -e
# ============================================================
# Experiment 5: GazeDiffuse on LLaDA 8B
# Goal: Show method scales to 8B model. lambda={-1,0,+1} subset.
# GPU: 1x A100 (80GB) or 2x RTX8000 (with 4-bit) | Time: ~8 hours
# Owner: Both | Week 4-5
# ============================================================
#SBATCH --job-name=exp5-llada-gaze
#SBATCH --output=logs/exp5_%j.out
#SBATCH --error=logs/exp5_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rs9174@nyu.edu
#SBATCH --account=torch_pr_111_general

module purge && module load anaconda3/2024.02
conda activate $SCRATCH/envs/gazediffuse

cd $SCRATCH/gaze-diffuse
mkdir -p logs results

echo "=== Experiment 5: GazeDiffuse on LLaDA 8B ==="
echo "Start: $(date)"
nvidia-smi

# Subset conditions (smaller due to cost)
for LAM in -1.0 0.0 1.0; do
    echo "--- Lambda=$LAM ---"
    python -m src.gaze_guidance \
        --mdlm_checkpoint checkpoints/llada-8b \
        --gaze_checkpoint checkpoints/gaze_predictor/gaze_predictor_best.pt \
        --lam "$LAM" \
        --steps 128 \
        --gen_length 128 \
        --top_k 100 \
        --n_samples 100 \
        --n_prompts 50 \
        --output "results/llada_gaze_lam${LAM}.jsonl"
    echo "Lambda=$LAM done at $(date)"
done

# Evaluate
for LAM in -1.0 0.0 1.0; do
    python -m src.metrics \
        --input "results/llada_gaze_lam${LAM}.jsonl" \
        --reference "results/llada_gaze_lam0.0.jsonl" \
        --output "results/llada_gaze_metrics_lam${LAM}.json" \
        --ppl_model gpt2-medium
done

echo "End: $(date)"
echo "=== Experiment 5 Complete ==="
