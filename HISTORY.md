# GazeDiffuse — Project History and Session Log

> This file is the single source of truth for anyone picking up this project.
> It tracks every substantial change, known issues, debugging discoveries,
> architectural decisions, and pending work. Update after every session.

---

## Project Overview

**GazeDiffuse** is the first method applying eye-tracking gaze signals to steer
text generation in masked diffusion language models (MDLM, LLaDA). Unlike
autoregressive gaze guidance (Sauberli et al., EACL 2026), it guides ALL token
positions simultaneously at each denoising step, producing more globally
coherent readability changes.

- **Authors**: Rahil Singhi, Siddhant, Prof. Sai Qian Zhang
- **Lab**: SAI Lab, NYU Tandon / Courant
- **Targets**: EMNLP 2026 (primary, ARR ~May/June), NeurIPS 2026 (stretch, May 15)
- **Repo**: https://github.com/rahilsinghi/gaze-diffuse
- **HPC**: NYU Torch cluster (`login.torch.hpc.nyu.edu`), account `torch_pr_111_general`

---

## Session Log

### Session 1 — 2026-03-12 (Initial Scaffolding)

**What was done:**
1. Cloned `everything-claude-code` as a submodule for best practices/skills
2. Read the research guide docx thoroughly for experiment plan context
3. Scaffolded both research tracks:
   - **Rahil's track**: MDLM baseline (Exp 1) + GazeDiffuse sampler (Exp 4)
   - **Siddhant's track**: Gaze predictor training (Exp 2) + AR baseline (Exp 3)
4. Created all source modules:
   - `src/gaze_guidance.py` — Core GazeDiffuse algorithm (339 lines)
   - `src/gaze_predictor.py` — BERT-based fixation predictor (412 lines)
   - `src/ar_baseline.py` — Sauberli et al. replication on GPT-2 (220 lines)
   - `src/metrics.py` — FKGL, ARI, MAUVE, self-PPL, FK variance (332 lines)
   - `src/models/mdlm_wrapper.py` — MDLM + LLaDA wrappers (314 lines)
   - `src/data/geco.py` — GECO corpus loader with CV splits (256 lines)
   - `src/data/prompts.py` — 50 diverse prompt seeds (77 lines)
5. Created 38 unit tests (all passing locally)
6. Created HPC scripts:
   - `scripts/setup_hpc.sh` — Conda env in $SCRATCH, PyTorch CUDA 12.1
   - `scripts/download_data.sh` — GECO corpus download
   - `scripts/download_checkpoints.sh` — MDLM-OWT + optional LLaDA 8B
   - `scripts/slurm/exp{1-5}_*.slurm` — All 5 experiment SLURM templates
7. Created comprehensive README.md
8. Set up SSH config for Torch HPC
9. Pushed 8 structured commits to GitHub

**HPC Access Investigation:**
- Discovered user's cluster is Torch (NOT Greene) — migrated all configs
- Torch uses Microsoft device login (not Duo), no SSH keys supported
- SSH auth fails after completing Microsoft device login — server rejects after 3 attempts
- Open OnDemand (`ood.torch.hpc.nyu.edu`) returns: "User rs9174 authenticated with identity provider nyu does not exist"
- **Root cause identified**: HPC account for rs9174 has NOT been provisioned on Torch
- Prof. Zhang needs to add rs9174 as a member via `projects.rit.nyu.edu` portal
- Email draft created in Gmail explaining the exact steps needed

**Bugs Found and Fixed:**
- `pyproject.toml` had `requires-python >= 3.10` but local Python was 3.9.6 → changed to `>= 3.9`
- `textstat.textstatistics().text_sentences()` removed in textstat >= 0.7 → switched to `re.split(r"(?<=[.!?])\s+", text.strip())`
- Test case "Cat sat." too short for FKGL → changed to longer sentence
- SSH stale host keys for `gw.hpc.nyu.edu` and `greene.hpc.nyu.edu` → cleared with `ssh-keygen -R`
- `.gitignore` pattern `data/` was catching `src/data/` → changed to `/data/` (root-only)

**Key Decisions:**
- All config classes use `@dataclass(frozen=True)` for immutability
- No Hydra YAML configs — using Python dataclasses directly (simpler for research)
- CLI entry points via `python -m src.module_name` (no setup.py scripts)
- MockMDLM in conftest returns random logits for GPU-free testing
- BERT vocab_size (30522) used in mocks to match gaze predictor tokenizer

### Session 2 — 2026-03-15 (Testing, Data Pipeline, Paper Scaffold)

**What was done:**
1. Downloaded GECO eye-tracking corpus (220MB, 774K rows, 57 columns)
   - Correct URL: `https://expsy.ugent.be/downloads/geco/files/MonolingualReadingData.xlsx`
   - DO NOT use `https://expsy.ugent.be/geco/` (redirects to HTML page)
2. Fixed GECO data parser for real data format:
   - WORD_ID format is `PART-TRIAL-WORDPOS` (e.g., `1-5-3`), NOT `w1.1` as assumed
   - sentence_id = PART * 10000 + TRIAL (composite key)
   - Fixation column uses `"."` string for missing values → added `pd.to_numeric(errors="coerce")`
   - Some WORD_ID entries may not match regex → added NaN row filtering
3. Created 21 new tests (5 smoke, 11 integration, 5 GECO validation):
   - `tests/test_training_smoke.py` — Training loop, checkpoint save/load, score_tokens, score_vocabulary, Spearman r
   - `tests/test_integration.py` — Data pipeline chain, predictor→guidance wiring, AR baseline, metrics, full E2E
   - `tests/test_geco_data.py` — Real GECO loading, fixation ranges, CV splits, tokenization
4. Expanded AR baseline tests from 3 → 10 (config validation, result format)
5. Created visualization script (`scripts/plot_results.py`):
   - 4 publication-quality plots: FKGL vs lambda, FK variance bars, MAUVE preservation, radar chart
   - Demo mode with synthetic data (`--demo` flag)
   - Colorblind-friendly palette (Wong 2011)
6. Created LaTeX paper scaffold (`paper/`):
   - `main.tex` — EMNLP 2023 template, 7 sections with TODOs
   - `references.bib` — 10 BibTeX entries (Sauberli, MDLM, LLaDA, GECO, MAUVE, etc.)
   - `tables/results.tex` — Results table template
   - `figures/method_overview.tex` — TikZ placeholder with layout description
7. Created Makefile with 11 targets
8. Fixed README test count (38 → 69)
9. Removed CLAUDE.md and everything-claude-code from git (kept local, gitignored)

**Bugs Found and Fixed:**
- GECO WORD_ID regex mismatch: `r"w(\d+)\.\d+"` → `r"(\d+)-(\d+)-(\d+)"`
- GECO fixation column has `"."` strings instead of NaN → added `pd.to_numeric(errors="coerce")`
- Training loss test too strict with 20 examples and 2 epochs at LR=5e-4 → reduced LR to 1e-4, increased to 4 epochs, check first-vs-last instead of consecutive
- `num_workers=0` required in all test DataLoaders (macOS `fork` hangs)

**Key Numbers:**
- 69 total tests, all passing
- 51% code coverage (up from 35%)
- GECO: 774,015 rows, 57 columns, 14 participants, fixation median ~200-300ms
- Test suite runs in ~28 seconds (excluding GECO xlsx load which takes ~3 min)

### Session 3 — 2026-03-26 (Local Environment Setup + Training Launch)

**What was done:**
1. Set up local development environment:
   - Created `.venv` with Python 3.14.3 (Homebrew)
   - Installed all deps via `pip install -e ".[dev]"` — torch 2.11, transformers 5.4, etc.
   - Added `openpyxl` (missing dep for GECO xlsx parsing)
   - Confirmed PyTorch MPS available (Apple Silicon GPU acceleration)
2. Downloaded GECO eye-tracking corpus (220MB) to `data/geco/`
   - Processed CSV cache created automatically (loads in <1s after first parse)
   - Validated: 473,116 rows, 14 participants, mean fixation 274.3ms, median 230ms
3. Ran full test suite: 69/69 passing (including 5 GECO data tests that were skipped before)
4. Fixed training code for local execution:
   - Added MPS device support (CUDA > MPS > CPU fallback) in `gaze_predictor.py`
   - Fixed `num_workers=2` → `num_workers=0` in training DataLoaders (macOS fork hang)
5. Started Experiment 2 (gaze predictor training) locally on MPS:
   - BERT-base-uncased, 3 epochs, batch_size=32, lr=2e-5
   - ~44K steps total, estimated 30-45 min on Apple Silicon
6. Created `CLAUDE.md` with project context for AI assistants
7. Updated `HISTORY.md` with Session 3 findings

**Bugs Found and Fixed:**
- `openpyxl` not in `pyproject.toml` dependencies — needed for `pd.read_excel()` on GECO xlsx
- `num_workers=2` in training DataLoaders causes macOS fork hangs — changed to 0
- Training code only checked CUDA, not MPS — added MPS fallback for Apple Silicon
- GECO data has 12 rows with NaN words (sentence 20028, position 46) — added `dropna(subset=["word"])` in loader
- GECO CSV reload parses numeric-looking words (e.g., "1984") as floats — added `astype(str)` on word column in both xlsx processing and CSV reload paths. 205 words affected.
- Checkpoint pickle portability: `torch.save` pickles `GazePredictorConfig` as `__main__.GazePredictorConfig` when training runs via `python -m src.gaze_predictor`. Loading from another module (e.g., `ar_baseline`) fails with `AttributeError: module '__main__' has no attribute 'GazePredictorConfig'`. Fix: save config as `asdict(config)` dict, reconstruct on load. Re-saved both checkpoints.

**Key Numbers:**
- GECO data: 473,116 rows, 14 participants, fixation range 0-1998ms
- After filtering: 472,945 training examples
- 69/69 tests passing, 51% coverage
- Local env: Python 3.14.3, PyTorch 2.11.0, MPS enabled

**Experiment 2 Results (Gaze Predictor — Local MPS):**
- Best Spearman r: 0.241 (epoch 2)
- Train loss: 0.942 → 0.909 → 0.881 (steadily decreasing)
- Val loss: 0.934 → 0.931 → 0.936 (slight overfit by epoch 3)
- Best checkpoint: `checkpoints/gaze_predictor/gaze_predictor_best.pt` (418MB)
- Total training time: ~3.5 hours on MPS (Apple Silicon), num_workers=0
- Note: Spearman r=0.24 is below Sauberli et al. (~0.3-0.4). Possible improvements:
  - More epochs (try 5-10)
  - Pre-tokenize dataset to speed up training
  - Tune LR or try warmup schedule
  - Run on GPU for faster iteration

**Experiment 3 Preliminary Results (AR Baseline — GPT-2 small, 5 samples each, 64 tokens):**
- Lambda=+1 (harder): FKGL=11.22, ARI=11.69, FK Var=3.54 — guidance working, higher readability + low variance
- Lambda=0 (unguided): FKGL=8.56, ARI=6.65, FK Var=10.87 — baseline
- Lambda=-1 (easier): FKGL=10.91, ARI=10.54, FK Var=15.57 — NOT working as expected (higher than unguided, should be lower)
- Possible cause: gaze predictor r=0.24 too weak for reliable easier-direction guidance
- Need more samples (200+) and stronger predictor for conclusive results

**Experiment 3 Larger Run (AR Baseline — GPT-2 small, 20 samples, 128 tokens):**
- Lambda=+1: FKGL=12.57, ARI=13.64, FK Var=21.19 — slight upward shift (correct direction)
- Lambda=0:  FKGL=11.44, ARI=11.79, FK Var=13.23 — baseline (very high std=9.86)
- Lambda=-1: FKGL=11.66, ARI=11.79, FK Var=22.98 — not working (indistinguishable from unguided)
- Conclusion: gaze predictor r=0.24 too weak to reliably steer. Need more epochs or better training
- GPT-2 small produces highly variable readability (std ~10-13), masking any guidance signal
- GPT-2 tokenizer has no pad_token — added `pad_token = eos_token` fallback in ar_baseline.py

**Bugs Found and Fixed (continued):**
- GPT-2 tokenizer missing pad_token causes ValueError in `score_vocabulary` — added `tokenizer.pad_token = tokenizer.eos_token` in `ar_baseline.py`
- Checkpoint pickle portability: config saved as `__main__.GazePredictorConfig` — re-saved as dict, fixed `train_gaze_predictor` to use `asdict(config)` going forward

**HPC Access Update (2026-03-27):**
- SSH to Torch works: `ssh rs9174@login.torch.hpc.nyu.edu` (Microsoft device login)
- `$SCRATCH` exists at `/scratch/rs9174`
- **Cannot submit jobs**: need a `torch_pr_xxx_yyy` project account, currently only have default `users`
- Portal URL changed: now `https://projects.hpc.nyu.edu` (NOT `projects.rit.nyu.edu` as assumed in Session 1)
- **GPU hardware is different than assumed**:
  - NO rtx8000 or a100 partitions on Torch
  - Available: H200 (8x per node), H100 (4x per node), L40S (4x per node)
  - Public partitions: `h200_public`, `l40s_public`
  - All SLURM scripts need updating for new partition names and GPU types
- Conda module: `anaconda3/2025.06` (not 2024.02 as assumed)
- Follow-up email drafted to Prof. Zhang requesting project account registration

### Session 4 — 2026-04-06 (HPC Verification + Retraining + Test Coverage)

**What was done:**
1. Connected to NYU VPN and SSHed into Torch HPC
2. Verified HPC project account status — still only `users`, no `torch_pr_*` account
   - `sacctmgr show associations user=rs9174` shows only `users` account with `normal` QOS
   - Cannot submit GPU jobs until Prof. Zhang registers project at https://projects.hpc.nyu.edu
3. Ran `sinfo` to get full partition/GPU inventory — **corrected previous assumptions**:
   - A100 GPUs DO exist (4x/node, 43 nodes, partitions: a100_cilvr, a100_cds, a100_chemistry)
   - RTX6000 GPUs exist (8x/node, 2 nodes)
   - Tandon-specific partitions: `h200_tandon`, `h100_tandon`
4. Updated all 5 SLURM scripts:
   - `rtx8000` → `h100_tandon` (exp 1-4), `a100` → `h200_tandon` (exp 5)
   - Added `--partition=` lines to all scripts
   - Updated `--account=torch_pr_xxx_yyy` placeholder
5. Updated `setup_hpc.sh`: conda module `anaconda3/2024.02` → `anaconda3/2025.06`
6. Verified all 69 tests pass (27.89s)
7. Kicked off gaze predictor retraining: 10 epochs (up from 3), targeting r=0.3-0.4
   - Same hyperparams: BERT-base, batch_size=32, lr=2e-5
   - Running on local MPS, estimated ~10-12 hours
8. Boosted test coverage from 51% toward 80%:
   - Added tests for gaze_guidance.py: generate_samples, save_generations, MDLM wrapper version
   - Added tests for metrics.py: load_generations, print_results_table, perplexity/MAUVE mocking
   - Added new test file for mdlm_wrapper.py: MDLMWrapper and LLaDAWrapper methods
9. Updated CLAUDE.md and HISTORY.md with corrected GPU/partition info
10. Created persistent memory files for future session continuity

**HPC Partition Discovery (via `sinfo` on 2026-04-06):**
- h200_tandon: gpu:h200:8 (7 on gh126), nodes gh101-130
- h100_tandon: gpu:h100:4, nodes gh001-015
- h200_public: same nodes as h200_tandon
- l40s_public: gpu:l40s:4, nodes gl001-068
- a100_cilvr/a100_cds/a100_chemistry: gpu:a100:4, nodes ga001-043
- rtx6000: gpu:rtx6000:8, nodes gr101-102

**Key Numbers:**
- 69 tests passing (pre-coverage-boost), 51% coverage
- Gaze predictor retraining: 10 epochs on MPS, ~10-12 hours
- HPC account: still `users` only (blocked)

**Known Issues:**
- Multi-line sbatch commands break in Torch terminal — use script files instead
- SSH host key changes periodically — `ssh-keygen -R login.torch.hpc.nyu.edu` to fix

---

## Architecture and Design Decisions

### Core Algorithm (gaze_guidance.py)
```
score = log P_LM(token) + lambda * gaze(token)
```
- lambda < 0 → easier text (favor low-fixation words)
- lambda > 0 → harder text (favor high-fixation words)
- lambda = 0 → unguided baseline
- Confidence-based unmasking: reveal highest-confidence tokens first
- Linear reveal schedule (equal tokens per step)

### Gaze Predictor (gaze_predictor.py)
- BERT-base-uncased fine-tuned on GECO regression task
- Input format: `[CLS] left_5 [SEP] target [SEP] right_5 [SEP]`
- Output: scalar fixation duration prediction via [CLS] pooling + linear head
- Two inference modes:
  - `score_tokens()` — score all positions in a sequence (for GazeDiffuse)
  - `score_vocabulary()` — score candidate tokens at one position (for AR baseline)

### Model Wrappers (mdlm_wrapper.py)
- `MDLMWrapper`: loads MDLM from checkpoint via submodule API, provides `get_log_probs(x, t)`
- `LLaDAWrapper`: loads LLaDA 8B from HuggingFace, mask_id=126336, optional 4-bit quantization
- Both share interface: `create_masked_input()`, `decode()`

### Data Format
- GECO processed CSV: `word, sentence_id, word_position, participant, mean_fixation_ms`
- Generation output JSONL: `{prompt, generation, full_text, lam, steps, ...}`
- Metrics output JSON: `{fkgl_mean, fkgl_std, ari_mean, ari_std, self_ppl, fk_sentence_variance, mauve_score, n_samples}`

---

## Known Issues and Pitfalls

### DO NOT:
- Use `https://expsy.ugent.be/geco/` for GECO download (redirects to HTML). Use `https://expsy.ugent.be/downloads/geco/files/MonolingualReadingData.xlsx`
- Assume GECO WORD_ID format is `w1.1`. It is `PART-TRIAL-WORDPOS` (e.g., `1-5-3`)
- Use `num_workers > 0` in DataLoaders on macOS (causes fork hangs)
- Use `textstat.textstatistics().text_sentences()` (removed in textstat >= 0.7)
- Expect SSH keys to work on Torch HPC (uses Microsoft device login only)
- Try to SSH to Torch without NYU VPN active
- Assume Torch only has H200/H100 — it also has A100 (4x/node), L40S (4x/node), RTX6000 (8x/node)
- Use `projects.rit.nyu.edu` for HPC projects — new URL is `projects.hpc.nyu.edu`
- Assume conda module is `anaconda3/2024.02` — it's `anaconda3/2025.06`
- Assume training loss will monotonically decrease with high LR on tiny synthetic data

### WATCH OUT FOR:
- GECO fixation column (`WORD_TOTAL_READING_TIME`) uses `"."` string for missing values, not NaN
- The `position` argument in `score_vocabulary()` can equal `len(sequence)` (one past end) — this is valid Python slicing behavior, not a bug
- `MockMDLM.forward()` expects 2D input `[batch, seq_len]` — `gaze_guided_diffuse` adds the batch dim with `.unsqueeze(0)`
- GECO xlsx is 220MB and takes ~3 minutes to parse. After first load, it saves a processed CSV that loads in ~2 seconds
- `pip install` on macOS may need `--break-system-packages` flag (PEP 668)

---

## File Inventory

### Source Code (src/)
| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `gaze_guidance.py` | 390 | Core GazeDiffuse algorithm | Complete |
| `gaze_predictor.py` | 412 | BERT gaze predictor + training | Complete |
| `ar_baseline.py` | 220 | Sauberli replication (GPT-2) | Complete |
| `metrics.py` | 332 | All evaluation metrics | Complete |
| `models/mdlm_wrapper.py` | 314 | MDLM + LLaDA wrappers | Complete, untested on real models |
| `data/geco.py` | ~115 | GECO corpus loader + CV splits | Complete, validated on real data |
| `data/prompts.py` | 77 | 50 prompt seeds | Complete |

### Tests (tests/)
| File | Tests | What it covers |
|------|-------|----------------|
| `test_data.py` | 9 | Prompts, GazeExample, GazeDataConfig |
| `test_gaze_predictor.py` | 7 | Config, forward, gradient, dataset |
| `test_gaze_guidance.py` | 11 | Confidence schedule, config, sampling |
| `test_ar_baseline.py` | 10 | Config, result format, multi-prompt |
| `test_metrics.py` | 11 | FKGL, ARI, FK variance, pipeline |
| `test_training_smoke.py` | 5 | Training loop, checkpoint, inference |
| `test_integration.py` | 11 | Data→predictor→guidance→metrics E2E |
| `test_geco_data.py` | 5 | Real GECO validation (conditional) |
| **Total** | **69** | |

### Scripts and Infrastructure
| File | Purpose |
|------|---------|
| `scripts/setup_hpc.sh` | One-time Torch HPC conda env setup |
| `scripts/download_data.sh` | GECO corpus download |
| `scripts/download_checkpoints.sh` | MDLM + LLaDA checkpoint download |
| `scripts/plot_results.py` | 4 publication-quality result plots |
| `scripts/slurm/exp{1-5}*.slurm` | All 5 experiment SLURM templates |
| `Makefile` | 11 targets for common tasks |

### Paper (paper/)
| File | Purpose |
|------|---------|
| `main.tex` | EMNLP 2023 template, 7 sections with TODOs |
| `references.bib` | 10 BibTeX entries |
| `tables/results.tex` | Results table template |
| `figures/method_overview.tex` | TikZ diagram placeholder |

---

## Experiment Plan

| # | Experiment | GPU | Time | Owner | Status |
|---|-----------|-----|------|-------|--------|
| 1 | MDLM baseline PPL on OpenWebText | 1x H100/L40S | ~2h | Rahil | BLOCKED (need project account) |
| 2 | Gaze predictor training (BERT on GECO) | 1x H100/L40S | ~30min | Siddhant | DONE (r=0.241). 10-epoch retraining launched Session 4 |
| 3 | AR gaze guidance baseline (GPT-2, lambda sweep) | 1x H100/L40S | ~4h | Siddhant | IN PROGRESS — needs stronger predictor from retraining |
| 4 | GazeDiffuse on MDLM (lambda x steps grid) | 1x H100/L40S | ~6h | Rahil | BLOCKED (need project account) |
| 5 | GazeDiffuse on LLaDA 8B (lambda subset) | 1x H200 | ~6h | Both | BLOCKED (need project account) |

**Blocked on**: Prof. Zhang registering HPC project at `https://projects.hpc.nyu.edu` and adding rs9174

---

## Pending Tasks

### Critical (Blocking Experiments)
- [ ] **HPC access**: Follow up with Prof. Zhang on account provisioning email
- [x] **Verify GPU types**: Confirmed via `sinfo` (Session 4) — H200, H100, A100, L40S, RTX6000
- [x] **Verify conda modules**: Confirmed `anaconda3/2025.06` (Session 4)
- [ ] **Run setup_hpc.sh**: Create conda env on Torch once access works
- [ ] **Download data on HPC**: Run `download_data.sh` and `download_checkpoints.sh` on Torch

### High Priority (Paper Progress)
- [ ] **Run Experiment 1**: MDLM baseline PPL — validates the diffusion model works
- [x] **Run Experiment 2**: Gaze predictor training — running locally on MPS (Session 3)
- [ ] **Run Experiment 3**: AR baseline lambda sweep — the comparison target
- [ ] **Run Experiment 4**: GazeDiffuse MDLM grid — the main result
- [ ] **Fill paper results table**: Once experiments produce data
- [ ] **Create method overview TikZ diagram**: Replace placeholder in paper/figures/

### Medium Priority (Code Quality)
- [ ] **Increase test coverage**: Currently 51%, target 80%
- [ ] **Add Hydra configs**: Optional YAML configs for experiment reproducibility
- [ ] **WandB integration**: Add experiment tracking to training and generation scripts
- [ ] **Add type checking**: Run `mypy src/` and fix any issues

### Low Priority (Nice to Have)
- [ ] **Run Experiment 5**: LLaDA 8B scale-up (requires A100)
- [ ] **Qualitative examples**: Generate example outputs for paper appendix
- [ ] **Ablation studies**: Steps grid, top-k sensitivity, context window size
- [ ] **Open-source release**: Clean up repo for public release after submission

---

## Suggestions for Next Session

1. **Check gaze predictor retraining results** (launched Session 4, 10 epochs):
   - Look at `checkpoints/gaze_predictor/gaze_predictor_best.pt` — target Spearman r >= 0.30
   - If improved, re-run AR baseline (Exp 3) with 200+ samples for paper-ready numbers

2. **Check HPC access**: SSH in, run `sacctmgr show associations user=rs9174 format=account,user,partition`
   - If project account exists: run `scripts/setup_hpc.sh`, then experiments 1 and 4
   - If still blocked: follow up with Prof. Zhang

3. **If HPC still blocked**:
   - Run full AR baseline sweep locally (Exp 3) with retrained predictor
   - Start writing paper Introduction and Method sections (don't need results)
   - Create TikZ method overview diagram
   - Add WandB tracking to training scripts

4. **Once first results arrive**:
   - Run `python scripts/plot_results.py --results_dir results/ --output_dir results/figures/`
   - Fill in `paper/tables/results.tex` with actual numbers
   - Write Results and Analysis sections

---

## Contact and Resources

- **Rahil**: rs9174@nyu.edu
- **Prof. Zhang**: sai.qian.zhang@nyu.edu (advisor, HPC allocation owner)
- **Torch HPC docs**: https://services.rt.nyu.edu/docs/hpc/
- **Torch Open OnDemand**: https://ood.torch.hpc.nyu.edu (browser-based shell)
- **RIT Projects portal**: https://projects.rit.nyu.edu (manage HPC allocations)
- **GECO corpus**: https://expsy.ugent.be/downloads/geco/
- **MDLM paper**: https://arxiv.org/abs/2406.07524
- **LLaDA paper**: https://arxiv.org/abs/2502.09992
- **Sauberli et al.**: EACL 2026 (gaze-guided AR generation)
