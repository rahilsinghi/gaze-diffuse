# CLAUDE.md — GazeDiffuse

## Project Overview

**GazeDiffuse**: Gaze-guided readability control for masked text diffusion language models.
First application of eye-tracking gaze signals to steer generation in MDMs (parallel decoding).

- **Paper target**: EMNLP 2026 (ARR ~May/June) / NeurIPS 2026 (stretch, May 15)
- **Team**: Rahil Singhi, Siddhant, Prof. Sai Qian Zhang (NYU SAI Lab)
- **HPC**: NYU Torch (RTX8000 48GB, A100 80GB — GPU types TBC for Torch, keeping for now)

## Architecture

```
src/
├── gaze_predictor.py    # BERT-base fine-tuned on GECO for fixation duration prediction
├── gaze_guidance.py     # GazeDiffuse sampler — gaze-guided parallel denoising for MDLM
├── ar_baseline.py       # AR gaze guidance baseline (Sauberli replication on GPT-2)
├── metrics.py           # FKGL, ARI, MAUVE, self-PPL, sentence-level FK variance
├── data/
│   ├── geco.py          # GECO eye-tracking corpus loader
│   └── prompts.py       # Shared prompt seeds for generation experiments (50 diverse prompts)
└── models/
    └── mdlm_wrapper.py  # Wrappers for MDLM and LLaDA inference APIs
tests/                   # pytest, 38 unit tests passing
scripts/
├── setup_hpc.sh         # One-time HPC environment setup
├── download_data.sh     # Download GECO + UCL eye-tracking data
├── download_checkpoints.sh  # Download MDLM-OWT and LLaDA-8B
└── slurm/               # SLURM batch templates for Experiments 1-5
submodules/
├── mdlm/                # MDLM codebase (NeurIPS 2024)
└── llada/               # LLaDA codebase (NeurIPS 2025)
```

## Key Commands

```bash
# Run unit tests (no GPU needed)
python3 -m pytest tests/ -m unit -v

# Run all tests (slow tests need BERT download)
python3 -m pytest tests/ -v

# Format + lint
black src/ tests/ && isort src/ tests/ && ruff check src/ tests/

# Train gaze predictor (GPU required)
python -m src.gaze_predictor --data_dir data/geco --epochs 3

# Run GazeDiffuse sampling (GPU required)
python -m src.gaze_guidance --mdlm_checkpoint checkpoints/mdlm-owt \
    --gaze_checkpoint checkpoints/gaze_predictor/gaze_predictor_best.pt \
    --lam -1.0 --steps 64

# Evaluate outputs
python -m src.metrics --input results/generations.jsonl
```

## HPC Connection (NYU Torch)

```bash
# Connect via VPN (direct — no gateway hop needed)
ssh rs9174@login.torch.hpc.nyu.edu
# Or with SSH config alias:
ssh torch

# Auth: Microsoft device login (no SSH keys supported on Torch)
# Data transfer node: dtn.torch.hpc.nyu.edu

# Interactive GPU session
srun --pty --gres=gpu:rtx8000:1 --mem=40GB --cpus-per-task=8 --time=4:00:00 /bin/bash

# A100 for LLaDA (Experiment 5)
srun --pty --gres=gpu:a100:1 --mem=80GB --cpus-per-task=8 --time=4:00:00 /bin/bash

# NOTE: GPU types (RTX8000, A100) may differ on Torch — verify with `sinfo` after first login

# Submit batch jobs
sbatch scripts/slurm/exp1_mdlm_baseline.slurm

# Monitor jobs
squeue -u $USER
sacct -j <JOBID> --format=State,Elapsed
```

### HPC First-Time Setup

```bash
# Connect to Torch (VPN required, Microsoft device login)
ssh rs9174@login.torch.hpc.nyu.edu
cd $SCRATCH
git clone <repo-url> gaze-diffuse && cd gaze-diffuse
bash scripts/setup_hpc.sh          # Creates conda env in $SCRATCH/envs/
bash scripts/download_data.sh      # Downloads GECO corpus
bash scripts/download_checkpoints.sh  # Downloads MDLM-OWT (~1.2GB)
bash scripts/download_checkpoints.sh --llada  # Also downloads LLaDA-8B (~16GB)
```

### HPC Environment

```bash
module purge && module load anaconda3/2024.02
conda activate $SCRATCH/envs/gazediffuse
```

## MDLM API (Key Integration Points)

- Class: `diffusion.Diffusion` (LightningModule) in `submodules/mdlm/diffusion.py`
- `model.forward(x, sigma)` → log probs, shape (B, L, V) — already in log space
- `model.mask_index` → mask token ID (103 for BERT, or vocab_size for custom)
- `model.noise(t)` → (sigma_t, sigma_rate_t), t is (B, 1) in [0, 1]
- Timesteps: 1.0 (fully masked) → eps (~1e-5, clean)
- Parameterization: 'subs' (substitution-based, default)
- Backbone: DIT (Diffusion Transformer) — same architecture class as image DiTs

## LLaDA API (Key Integration Points)

- Load: `AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)`
- mask_id = 126336
- `generate()` in `submodules/llada/generate.py` with steps, gen_length, block_length
- Gaze injection point: after model forward, modify logits before confidence scoring

## Conventions

- Python 3.9+ locally, 3.10 on HPC
- Type annotations on all functions
- Immutable dataclasses (`frozen=True`) for configs
- `black` + `isort` + `ruff` for formatting/linting
- pytest with markers: @pytest.mark.unit, @pytest.mark.integration, @pytest.mark.slow
- No hardcoded paths — use CLI args
- Checkpoints, data, results are NOT committed (see .gitignore)

## Key Papers

| Paper | Venue | Role in Project |
|-------|-------|----------------|
| Sauberli et al. | EACL 2026 | AR gaze guidance — our baseline |
| MDLM (Sahoo et al.) | NeurIPS 2024 | Primary diffusion model we extend |
| LLaDA (Nie et al.) | NeurIPS 2025 | Scale-up experiment (8B) |
| DiTAS (Zhang Lab) | WACV 2025 | Quantization angle for future work |
| SEDD (Lou et al.) | ICML 2024 | Best discrete sampler, used by MDLM |
| D3PM (Austin et al.) | NeurIPS 2021 | Mathematical foundation |

## Experiment Status

| Exp | Description | Owner | Status |
|-----|-------------|-------|--------|
| 1 | MDLM baseline PPL on OWT | Rahil | Code ready, needs HPC run |
| 2 | Gaze predictor (BERT on GECO) | Siddhant | Code ready, needs GECO data + HPC |
| 3 | AR gaze baseline (GPT-2) | Siddhant | Code ready, needs gaze predictor |
| 4 | GazeDiffuse on MDLM | Rahil | Code ready, needs gaze predictor + MDLM |
| 5 | GazeDiffuse on LLaDA 8B | Both | Code ready, needs A100 |
