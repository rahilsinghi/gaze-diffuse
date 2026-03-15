# GazeDiffuse

**Controllable Readability in Text Diffusion Models via Gaze-Guided Parallel Decoding**

Rahil Singhi &middot; Siddhant &middot; Prof. Sai Qian Zhang
NYU Tandon / Courant &middot; SAI Lab &middot; 2026

---

## TL;DR

We apply eye-tracking gaze signals to steer text generation in **masked diffusion language models** — guiding *all* token positions simultaneously at each denoising step. Unlike autoregressive gaze guidance (Sauberli et al., EACL 2026), which operates one token at a time, GazeDiffuse produces more globally coherent readability changes by leveraging diffusion's parallel decoding.

**Key result**: GazeDiffuse is the first method to combine gaze-guided generation with text diffusion models. It is training-free — requiring only a lightweight gaze predictor atop a frozen pretrained MDLM.

## Method

<p align="center">
<img src="docs/method_overview.png" alt="GazeDiffuse Method" width="700">
</p>

### Core Algorithm

1. **Initialize** a sequence as `[prompt tokens] + [MASK] * gen_length`
2. **At each denoising step** (parallel across all positions):
   - Get log-probabilities from MDLM (bidirectional attention)
   - Score top-k candidates per position with a gaze predictor (predicted fixation duration)
   - Combine: `score = log P_LM(token) + λ · gaze(token)`
   - Reveal highest-confidence tokens this step
3. **Control direction** with λ:
   - `λ < 0` → easier text (favor low-fixation words)
   - `λ > 0` → harder text (favor high-fixation words)
   - `λ = 0` → unguided baseline

### Gaze Predictor

BERT-base fine-tuned on the [GECO](https://expsy.ugent.be/geco/) eye-tracking corpus (5,031 sentences, 14 participants). Predicts mean fixation duration per word-in-context using a context window of 5 tokens on each side.

## Results

| Method | FKGL (λ=-1) | FKGL (λ=0) | FKGL (λ=+1) | MAUVE | Self-PPL | FK Var. |
|--------|:-----------:|:----------:|:-----------:|:-----:|:--------:|:-------:|
| Unguided MDLM | — | *baseline* | — | — | — | — |
| AR + Gaze (GPT-2) | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **GazeDiffuse (MDLM)** | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **GazeDiffuse (LLaDA 8B)** | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |

*Results will be filled as experiments complete. See [experiment plan](#experiments) below.*

## Installation

### Local Development (macOS/Linux)

```bash
git clone https://github.com/<your-org>/gaze-diffuse.git
cd gaze-diffuse
git submodule update --init --recursive
pip install -e ".[dev]"
python3 -m pytest tests/ -v  # Verify 69 tests pass
```

### NYU Torch HPC

```bash
# Connect via VPN (Microsoft device login, no SSH keys)
ssh rs9174@login.torch.hpc.nyu.edu   # or: ssh torch
cd $SCRATCH
git clone <repo-url> gaze-diffuse && cd gaze-diffuse
bash scripts/setup_hpc.sh              # Create conda env + install all deps
bash scripts/download_data.sh          # Download GECO eye-tracking corpus
bash scripts/download_checkpoints.sh   # Download MDLM-OWT checkpoint (~1.2GB)
```

## Project Structure

```
gaze-diffuse/
├── src/
│   ├── gaze_predictor.py      # BERT-based fixation duration predictor (Exp 2)
│   ├── gaze_guidance.py       # GazeDiffuse sampler — core contribution (Exp 4)
│   ├── ar_baseline.py         # AR gaze guidance baseline (Exp 3)
│   ├── metrics.py             # FKGL, ARI, MAUVE, self-PPL, FK variance
│   ├── data/
│   │   ├── geco.py            # GECO corpus loader + subject-level CV splits
│   │   └── prompts.py         # 50 shared prompt seeds
│   └── models/
│       └── mdlm_wrapper.py    # MDLM + LLaDA inference wrappers
├── tests/                     # 69 tests (unit + integration + smoke), pytest
├── scripts/
│   ├── setup_hpc.sh           # One-time HPC environment setup
│   ├── download_data.sh       # Download eye-tracking datasets
│   ├── download_checkpoints.sh # Download model checkpoints
│   └── slurm/                 # SLURM batch templates (Experiments 1-5)
├── submodules/
│   ├── mdlm/                  # MDLM (Sahoo et al., NeurIPS 2024)
│   └── llada/                 # LLaDA 8B (Nie et al., NeurIPS 2025)
├── configs/                   # Hydra experiment configs
├── CLAUDE.md                  # AI assistant context
└── pyproject.toml             # Project metadata + dev dependencies
```

## Experiments

All experiments use the same 50 prompt seeds and evaluation metrics for fair comparison.

| # | Experiment | GPU | Time | Owner | Week |
|---|-----------|-----|------|-------|------|
| 1 | MDLM baseline PPL on OpenWebText | 1× RTX8000 | ~2h | Rahil | 1 |
| 2 | Gaze predictor training (BERT on GECO) | 1× RTX8000 | ~30min | Siddhant | 1-2 |
| 3 | AR gaze guidance baseline (GPT-2, λ sweep) | 1× RTX8000 | ~8h | Siddhant | 2-3 |
| 4 | **GazeDiffuse on MDLM** (λ × steps grid) | 1× RTX8000 | ~12h | Rahil | 3-4 |
| 5 | GazeDiffuse on LLaDA 8B (λ subset) | 1× A100 | ~12h | Both | 4-5 |

### Running Experiments

```bash
# On Torch HPC:
sbatch scripts/slurm/exp1_mdlm_baseline.slurm
sbatch scripts/slurm/exp2_gaze_predictor.slurm
sbatch scripts/slurm/exp3_ar_baseline.slurm
sbatch scripts/slurm/exp4_gazediffuse_mdlm.slurm
sbatch scripts/slurm/exp5_gazediffuse_llada.slurm
```

### Evaluation

```bash
python -m src.metrics \
    --input results/gazediffuse_lam-1.0_steps64.jsonl \
    --reference results/gazediffuse_lam0.0_steps64.jsonl \
    --output results/metrics.json
```

## Metrics

| Metric | What it Measures | Paper Role |
|--------|-----------------|------------|
| **FKGL** (Flesch-Kincaid Grade Level) | Readability level | Primary — shows guidance shifts reading difficulty |
| **ARI** (Automated Readability Index) | Readability (corroborates FKGL) | Secondary confirmation |
| **MAUVE** | Distributional similarity to unguided text | Fluency preservation |
| **Self-PPL** | Perplexity under base LM | Coherence preservation |
| **FK Sentence Variance** | Readability consistency across sentences | **Key claim**: parallel < sequential variance |

## Related Work

| Paper | Venue | Key Idea | Our Difference |
|-------|-------|----------|----------------|
| Sauberli et al. | EACL 2026 | Gaze guidance for AR LLMs (GPT-2) | We apply to **masked diffusion** — all tokens guided in parallel |
| MDLM (Sahoo et al.) | NeurIPS 2024 | Masked diffusion LM, SOTA PPL | We **add** gaze-guided readability control |
| LLaDA (Nie et al.) | NeurIPS 2025 | 8B masked diffusion LLM | We extend to show method scales |
| Diffusion-LM (Li et al.) | NeurIPS 2022 | Classifier guidance for continuous diffusion | We work in **discrete** space with gaze, not classifiers |
| DiTAS (Zhang Lab) | WACV 2025 | PTQ for image DiTs | Architecture connection — future quantization work |

## Timeline

- **Mar 2026**: Project setup, MDLM baseline, gaze predictor training
- **Apr 2026**: Full experiment grid, LLaDA scale-up, paper writing
- **May 2026**: EMNLP 2026 submission via ARR (primary target)
- **May 15, 2026**: NeurIPS 2026 submission (stretch target)

## Citation

```bibtex
@inproceedings{singhi2026gazediffuse,
  title={GazeDiffuse: Controllable Readability in Text Diffusion Models
         via Gaze-Guided Parallel Decoding},
  author={Singhi, Rahil and Siddhant and Zhang, Sai Qian},
  booktitle={Proceedings of EMNLP},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Prof. Sai Qian Zhang and NYU SAI Lab for GPU cluster access and advising
- NYU HPC team for Torch cluster support
- Kuleshov Group (Cornell) for the MDLM codebase
- Sauberli et al. for the gaze-guided generation framework
