#!/bin/bash -e
# ============================================================
# Experiment 4: GazeDiffuse on MDLM ★ PRIMARY CONTRIBUTION ★
# Goal: Gaze-guided MDLM at lambda={-2,-1,0,+1,+2} x T={32,64,128,256}
# GPU: 1x RTX8000 (48GB) | Time: ~6-8 hours
# Owner: Rahil | Week 3-4
# ============================================================
#SBATCH --job-name=exp4-gazediffuse
#SBATCH --output=logs/exp4_%j.out
#SBATCH --error=logs/exp4_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rs9174@nyu.edu
#SBATCH --account=torch_pr_111_general

module purge && module load anaconda3/2024.02
conda activate $SCRATCH/envs/gazediffuse

cd $SCRATCH/gaze-diffuse
mkdir -p logs results

echo "=== Experiment 4: GazeDiffuse on MDLM ==="
echo "Start: $(date)"
nvidia-smi

# Full lambda x steps grid
for LAM in -2.0 -1.0 0.0 1.0 2.0; do
    for STEPS in 32 64 128 256; do
        echo "--- Lambda=$LAM, Steps=$STEPS ---"
        python -m src.gaze_guidance \
            --mdlm_checkpoint checkpoints/mdlm-owt \
            --gaze_checkpoint checkpoints/gaze_predictor/gaze_predictor_best.pt \
            --lam "$LAM" \
            --steps "$STEPS" \
            --gen_length 128 \
            --top_k 100 \
            --n_samples 200 \
            --n_prompts 50 \
            --output "results/gazediffuse_lam${LAM}_steps${STEPS}.jsonl"
        echo "Lambda=$LAM Steps=$STEPS done at $(date)"
    done
done

# Evaluate all conditions
for LAM in -2.0 -1.0 0.0 1.0 2.0; do
    for STEPS in 32 64 128 256; do
        echo "--- Evaluating Lambda=$LAM, Steps=$STEPS ---"
        python -m src.metrics \
            --input "results/gazediffuse_lam${LAM}_steps${STEPS}.jsonl" \
            --reference "results/gazediffuse_lam0.0_steps${STEPS}.jsonl" \
            --output "results/gazediffuse_metrics_lam${LAM}_steps${STEPS}.json" \
            --ppl_model gpt2-medium
    done
done

echo "End: $(date)"
echo "=== Experiment 4 Complete ==="
