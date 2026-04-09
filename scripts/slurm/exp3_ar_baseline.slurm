#!/bin/bash -e
# ============================================================
# Experiment 3: AR Gaze Guidance Baseline (Sauberli Replication)
# Goal: GPT-2-medium + gaze guidance at lambda={-2,-1,0,+1,+2}
# GPU: 1x H100 (80GB) | Time: ~3-4 hours per lambda
# Owner: Siddhant | Week 2-3
# ============================================================
#SBATCH --job-name=exp3-ar-gaze
#SBATCH --output=logs/exp3_%j.out
#SBATCH --error=logs/exp3_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB
#SBATCH --partition=h100_tandon
#SBATCH --gres=gpu:h100:1
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rs9174@nyu.edu
#SBATCH --account=torch_pr_xxx_yyy

module purge && module load anaconda3/2025.06
conda activate $SCRATCH/envs/gazediffuse

cd $SCRATCH/gaze-diffuse
mkdir -p logs results

echo "=== Experiment 3: AR Gaze Guidance Baseline ==="
echo "Start: $(date)"
nvidia-smi

# Run for each lambda value
for LAM in -2.0 -1.0 0.0 1.0 2.0; do
    echo "--- Lambda = $LAM ---"
    python -m src.ar_baseline \
        --model_name gpt2-medium \
        --gaze_checkpoint checkpoints/gaze_predictor/gaze_predictor_best.pt \
        --lam "$LAM" \
        --top_k 50 \
        --max_new_tokens 128 \
        --n_samples 200 \
        --n_prompts 50 \
        --output "results/ar_baseline_lam${LAM}.jsonl"
    echo "Lambda $LAM done at $(date)"
done

# Evaluate all conditions
for LAM in -2.0 -1.0 0.0 1.0 2.0; do
    echo "--- Evaluating Lambda = $LAM ---"
    python -m src.metrics \
        --input "results/ar_baseline_lam${LAM}.jsonl" \
        --reference "results/ar_baseline_lam0.0.jsonl" \
        --output "results/ar_baseline_metrics_lam${LAM}.json" \
        --ppl_model gpt2-medium
done

echo "End: $(date)"
echo "=== Experiment 3 Complete ==="
