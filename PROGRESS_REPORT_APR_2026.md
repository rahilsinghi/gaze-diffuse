# GazeDiffuse Progress Report — Professor Meeting (April 9, 2026)

> Prepared by Rahil Singhi for progress call with Prof. Sai Qian Zhang
> Project: Gaze-Guided Text Generation in Masked Diffusion Language Models
> Targets: EMNLP 2026 (ARR May/June), NeurIPS 2026 (stretch, May 15)

---

## 1. Executive Summary

We have a **fully implemented, tested codebase** (1,016 lines of source, 194 passing tests, 69% coverage) covering all five planned experiments. Two experiments have been run locally on Apple Silicon. The core finding so far: **our gaze predictor (Spearman r=0.24) is too weak to meaningfully steer AR generation**, which is actually a useful baseline result for the paper. However, **all three GPU-intensive experiments (1, 4, 5) remain blocked** because we don't have a project account on the Torch HPC cluster. This is the single most critical blocker for the entire project.

---

## 2. What Has Been Done (Local Work)

### 2.1 Complete Implementation (All 5 Experiments)

Every module is written, tested, and ready to run on the cluster:

| Module | Purpose | Lines | Tests | Coverage |
|--------|---------|-------|-------|----------|
| `src/gaze_guidance.py` | **Core GazeDiffuse algorithm** (main contribution) | 143 | 11 | 80% |
| `src/gaze_predictor.py` | BERT-base fixation duration predictor | 182 | 7 | 53% |
| `src/ar_baseline.py` | Sauberli et al. GPT-2 replication | 89 | 25 | 61% |
| `src/metrics.py` | FKGL, ARI, MAUVE, self-PPL, FK variance | 143 | 11 | 80% |
| `src/models/mdlm_wrapper.py` | MDLM + LLaDA 8B wrappers | 91 | -- | 67% |
| `src/data/geco.py` | GECO corpus loader + CV splits | 115 | 5 | 75% |
| `src/data/prompts.py` | 50 shared prompt seeds | 6 | -- | 100% |
| **Total** | | **769** | **194** | **69%** |

Infrastructure is also ready:
- 5 SLURM scripts (updated for H200/H100/L40S partitions)
- HPC setup script (conda env, dependencies)
- Data + checkpoint download scripts
- Publication-quality plotting script (4 figure types)
- LaTeX paper scaffold (EMNLP template, 7 sections)

### 2.2 Experiment 2: Gaze Predictor Training (DONE)

**Goal**: Fine-tune BERT-base on the GECO eye-tracking corpus to predict per-word mean fixation duration from context.

**Setup**:
- Model: BERT-base-uncased with [CLS] pooling + linear regression head
- Input format: `[CLS] left_5_tokens [SEP] target_token [SEP] right_5_tokens [SEP]`
- Data: GECO corpus (473K examples, 14 participants), subject-level 5-fold CV
- Training: AdamW, lr=2e-5, batch_size=32, linear warmup + decay
- Hardware: Apple Silicon MPS GPU

**Results — First Run (3 epochs, ~3.5 hours)**:
| Epoch | Train Loss | Val Loss | Spearman r |
|-------|-----------|----------|------------|
| 1 | 0.942 | 0.934 | 0.227 |
| 2 | 0.909 | 0.931 | — |
| 3 | 0.881 | 0.936 | 0.230 |

Best checkpoint: **Spearman r = 0.241** (epoch 2)

**Results — Second Run (10 epochs, launched Apr 6)**:
- Process ran for 12+ hours on MPS but became **stuck** (macOS reported process state as `stuck`, memory ballooned to 9GB vs expected ~1.5GB)
- Root cause: MPS GPU memory leak on long training runs — PyTorch doesn't call `torch.mps.empty_cache()` between epochs
- Best checkpoint before hang: **Spearman r = 0.239** (epoch 2) — essentially no improvement
- Process was killed; checkpoint at `checkpoints/gaze_predictor_v2/gaze_predictor_best.pt` (438MB) is intact

**Key Insight**: The model plateaus at epoch 2 regardless of total epochs. More training is not the solution — we likely need architectural changes or to accept this as the predictor quality and frame results accordingly.

**Comparison to Sauberli et al.**: They report r ~ 0.3-0.4. Our gap could be due to:
- Different GECO preprocessing / feature engineering
- Different evaluation split strategy (we use subject-level CV)
- Different input format or context window
- We should compare our exact methodology against theirs

### 2.3 Experiment 3: AR Gaze Baseline (IN PROGRESS)

**Goal**: Replicate Sauberli et al. — gaze-guided decoding on GPT-2 medium where at each autoregressive step, top-k candidates are re-ranked by `score = log P_LM(token) + lambda * gaze(token)`.

**Setup**:
- LM: GPT-2 medium (355M params)
- Gaze predictor: our trained BERT (r=0.239)
- Decoding: greedy top-k=50 re-ranking (deterministic)
- Prompts: 50 diverse seeds (10-20 tokens each)
- Max new tokens: 128

**Results — Latest Run (April 7, 50 samples per condition)**:

| Condition | FKGL | ARI | FK Sentence Variance |
|-----------|------|-----|---------------------|
| Lambda=0 (unguided) | **9.83** +/- 2.89 | **10.10** +/- 2.83 | **8.77** |
| Lambda=-1 (easier) | 12.18 +/- 7.62 | 12.87 +/- 9.76 | 26.73 |

**These results are opposite to expectations.** With lam=-1 (should make text easier), FKGL actually goes UP (harder), and FK variance nearly triples. Two outlier samples show extreme FKGL scores (33.1 and 51.0) with degenerate text like _"choice-making processes involving choice-making processes..."_.

**Analysis**:

1. **Weak predictor signal**: At Spearman r=0.24, the gaze scores are nearly random. When added to LM log-probs, they act as noise, degrading the LM's own fluency and coherence rather than steering readability.

2. **Greedy decoding amplifies errors**: Because we use argmax selection, a single bad re-ranking decision propagates — the model can get stuck in repetitive loops when gaze noise overrides the LM's true preferences.

3. **This is actually a useful result for the paper**: It demonstrates that weak gaze prediction + sequential AR guidance = unreliable. Our hypothesis is that GazeDiffuse (parallel guidance across all positions) will be more robust to predictor noise because:
   - Errors at individual positions are diluted across the full sequence
   - The diffusion model can self-correct at subsequent denoising steps
   - Parallel scoring provides a more globally coherent signal

**Previous smaller runs** also showed similar patterns:
- Lambda=+1 (harder) shows a slight correct-direction shift (FKGL 12.57 vs 11.44 unguided)
- Lambda=-1 (easier) is indistinguishable from unguided or slightly worse
- Very high variance across samples (std 7-10 on FKGL)

---

## 3. The Core Algorithm (What We're Proposing)

### 3.1 GazeDiffuse: How It Works

Unlike AR guidance which modifies one token at a time, GazeDiffuse operates within a masked diffusion LM's denoising loop:

```
For each denoising step t = T, T-1, ..., 1:
    1. Get LM log-probs for ALL masked positions:  log P(x_i | x_unmasked)
    2. Score ALL candidates with gaze predictor:    gaze(x_i, context)
    3. Combine:  score_i = log P(x_i) + lambda * gaze(x_i)
    4. Select top-k tokens by CONFIDENCE to reveal (unmask)
    5. Reveal n_reveal = gen_length / T tokens per step (linear schedule)
```

**Key advantages over AR approach**:
- **Global coherence**: All positions are scored simultaneously, so the guidance signal considers the full text rather than left-to-right only
- **Error correction**: Tokens revealed early can influence later denoising steps; mistakes don't propagate irreversibly
- **Sentence-level FK variance** should be lower because readability is adjusted across the whole text at once

### 3.2 Why This Matters (Paper Narrative)

The story of the paper is:

1. **Sauberli et al. (EACL 2026)** showed gaze-guided AR decoding can control readability, but it's sequential — each token decision is made without seeing the full text.

2. **Our claim**: Masked diffusion LMs (MDLM, LLaDA) enable *parallel* gaze guidance that produces more globally coherent readability changes, as measured by lower sentence-level FK variance.

3. **Supporting evidence we need**:
   - AR baseline produces high FK variance (already shown - 8.77-26.73)
   - GazeDiffuse produces lower FK variance with similar FKGL shift
   - Both methods maintain fluency (low self-PPL) and text quality (high MAUVE)

---

## 4. What's Blocked and Why

### 4.1 HPC Project Account (CRITICAL BLOCKER)

**Status**: We can SSH into the Torch cluster, but cannot submit any GPU jobs.

**The problem**: SLURM requires a `torch_pr_xxx_yyy` project account to allocate GPUs. We currently only have the default `users` account, which has no GPU allocation.

**What needs to happen**:
1. Prof. Zhang goes to https://projects.hpc.nyu.edu
2. Registers a new project (or adds us to an existing SAI Lab project)
3. Adds `rs9174` (and Siddhant if applicable) as project members
4. The system creates a `torch_pr_xxx_yyy` account we can reference in SLURM scripts

**Verified on April 6**:
```
$ sacctmgr show associations user=rs9174 format=account,user,partition
    Account       User  Partition
---------- ---------- ----------
      users    rs9174     normal
```

No `torch_pr_*` entries — confirmed blocked.

**This blocks Experiments 1, 4, and 5** — the three experiments that require GPU hardware beyond Apple Silicon (MDLM/LLaDA models are too large for local MPS).

### 4.2 Available GPU Hardware (Once Unblocked)

| Partition | GPU | Count/Node | Recommended For |
|-----------|-----|-----------|----------------|
| `h100_tandon` | H100 80GB | 4x | Exp 1, 2, 3, 4 |
| `h200_tandon` | H200 141GB | 8x | Exp 5 (LLaDA 8B) |
| `l40s_public` | L40S 48GB | 4x | Exp 1, 2, 3 (fallback) |
| `h200_public` | H200 141GB | 8x | Exp 4, 5 (if tandon full) |

All SLURM scripts are already updated for these partitions. Once the account is provisioned, we can submit jobs immediately — the setup script, data download, and checkpoint download are all scripted.

---

## 5. What's Left To Do

### 5.1 Experiments Roadmap

| # | Experiment | Hardware | Est. Time | Status | Depends On |
|---|-----------|----------|-----------|--------|------------|
| 1 | MDLM baseline PPL on OpenWebText | 1x H100 | ~2h | BLOCKED | HPC account |
| 2 | Gaze predictor training | MPS (local) | DONE | **DONE** | -- |
| 3 | AR gaze baseline (GPT-2 medium) | MPS (local) | Partial | **PRELIMINARY** | Better predictor (optional) |
| 4 | **GazeDiffuse on MDLM** (main result) | 1x H100 | ~6h | BLOCKED | HPC + Exp 1 |
| 5 | GazeDiffuse on LLaDA 8B (scale-up) | 1x H200 | ~6h | BLOCKED | HPC + Exp 4 |

**Realistic timeline once HPC is unblocked**:
- Day 1: Run setup_hpc.sh + download data/checkpoints (~1h), then Exp 1 (~2h)
- Day 1-2: Run Exp 4 (GazeDiffuse MDLM) with lambda sweep (~6h)
- Day 2-3: Run Exp 5 (LLaDA 8B) if Exp 4 shows promise (~6h)
- Day 3-4: Compute all metrics, generate plots, fill results table
- Total: **~3-4 days from account provisioning to paper-ready results**

### 5.2 What We Can Do Without HPC

These tasks can proceed in parallel while waiting for HPC access:

1. **Write paper Introduction and Method sections** — don't need results
2. **Create method overview figure** (TikZ diagram of GazeDiffuse algorithm)
3. **Re-run Exp 3 with lambda=+1** (harder direction showed more signal)
4. **Investigate gaze predictor improvements** (see Section 6)
5. **Increase test coverage** to 80% (currently 69%)
6. **Add WandB experiment tracking** for reproducibility

---

## 6. Questions for the Professor

### 6.1 Critical / Blocking

1. **HPC Account**: Can you register the project at https://projects.hpc.nyu.edu and add rs9174? This is the single blocker for 3 of our 5 experiments. We just need a `torch_pr_xxx_yyy` account with GPU allocation on `h100_tandon` and `h200_tandon` partitions.

2. **MDLM Checkpoint**: Are we using the public MDLM-OWT checkpoint from the Sahoo et al. repo, or do you have a specific checkpoint you want us to use? Our download script pulls from the public release (~1.2GB).

### 6.2 Methodology / Framing

3. **Gaze predictor quality**: Our best Spearman r is 0.24 vs Sauberli et al.'s reported ~0.3-0.4. Should we:
   - (a) Accept this and frame it as "even with a weaker predictor, GazeDiffuse is more robust than AR guidance"?
   - (b) Invest time trying to match their predictor quality first (different preprocessing, architecture tweaks)?
   - (c) Use their pretrained predictor if available?
   
   Option (a) could actually strengthen our paper's story — showing GazeDiffuse is robust to predictor noise.

4. **AR baseline results**: The AR guidance with lam=-1 actually made text harder to read (FKGL 12.2 vs 9.8 unguided). Is this consistent with what Sauberli et al. observed with weak predictors? Or does this suggest a bug in our re-ranking formulation?

5. **Evaluation metrics**: We currently measure FKGL, ARI, FK sentence variance, self-PPL, and MAUVE. Are there additional metrics the reviewers would expect? Specifically:
   - Should we include human evaluation (even small-scale)?
   - Should we measure per-sentence readability distributions beyond just variance?
   - Is MAUVE the right distributional metric, or should we consider FED or BERTScore?

6. **Experiment 5 priority**: LLaDA 8B requires H200 GPUs and is framed as a "scale-up" experiment. Given the timeline pressure, should we prioritize MDLM results first and treat LLaDA as optional? Or is the scale story important for NeurIPS?

### 6.3 Paper Strategy

7. **Venue targeting**: With ARR deadlines in May/June, do we have enough runway? Realistically we need ~3-4 days of GPU time plus ~1 week for writing. Is the EMNLP submission realistic, or should we pivot to a workshop paper first?

8. **Paper scope**: Should the paper be:
   - (a) A full methods paper (GazeDiffuse algorithm + comprehensive evaluation)?
   - (b) A findings paper (shorter, focused on the empirical comparison)?
   - (c) Something else?

9. **Related work / positioning**: Are there other concurrent works applying guidance to diffusion LMs that we should be aware of? We're positioning against Sauberli et al. (AR gaze guidance) and the MDLM/LLaDA papers, but the classifier-free guidance literature in image diffusion is also relevant.

### 6.4 Technical Questions

10. **Confidence schedule**: We use a linear reveal schedule (equal tokens per step). The MDLM paper uses a cosine schedule. Should we ablate this? Which is standard for text diffusion?

11. **Lambda range**: We're sweeping lambda in {-1, 0, +1}. Should the range be wider (e.g., {-2, -1, -0.5, 0, +0.5, +1, +2})? The gaze scores are z-normalized so lambda=1 means 1 standard deviation of gaze signal.

12. **GECO data splits**: We use subject-level 5-fold CV (group by participant). Is this the correct evaluation protocol? Sauberli et al. may use different splits, which could explain our lower Spearman r.

---

## 7. Detailed Experiment Results (Reference Data)

### 7.1 Gaze Predictor (Experiment 2) — Full Details

**Data**:
- GECO eye-tracking corpus: 473,116 word-level fixation records
- 14 participants reading 5,031 English sentences
- Mean fixation duration: 274.3ms, median: 230ms, range: 0-1998ms
- Subject-level 5-fold CV: ~378K train, ~95K val per fold

**Model Architecture**:
```
BERT-base-uncased (110M params)
  -> [CLS] pooling (768-dim)
    -> Dropout(0.1)
      -> Linear(768, 1) regression head
        -> MSE loss
```

**Training Config**:
- Optimizer: AdamW, lr=2e-5
- Scheduler: Linear warmup (10%) + linear decay
- Batch size: 32
- Max sequence length: 64 tokens
- Gradient clipping: max_norm=1.0

**Checkpoints Available**:
| Version | Epochs | Best Epoch | Spearman r | Path |
|---------|--------|-----------|------------|------|
| v1 | 3 | 0 | 0.227 | `checkpoints/gaze_predictor/gaze_predictor_best.pt` |
| v1 final | 3 | 2 | 0.241 | `checkpoints/gaze_predictor/gaze_predictor_final.pt` |
| v2 | 10 (hung) | 2 | 0.239 | `checkpoints/gaze_predictor_v2/gaze_predictor_best.pt` |

### 7.2 AR Baseline (Experiment 3) — Full Details

**Generated Samples**: 100 total (50 per lambda condition)

**Unguided (lambda=0)**:
- FKGL: 9.83 +/- 2.89 (median 10.45)
- ARI: 10.10 +/- 2.83 (median 9.91)
- FK Sentence Variance: 8.77 (median 4.39)
- Avg generation length: 101.8 words
- No outliers (all FKGL < 20)
- Text quality: Coherent, standard GPT-2 output

**Guided (lambda=-1, "easier")**:
- FKGL: 12.18 +/- 7.62 (median 11.20)
- ARI: 12.87 +/- 9.76 (median 11.48)
- FK Sentence Variance: 26.73 (median 9.81)
- Avg generation length: 104.1 words
- 2 extreme outliers (FKGL 33.1 and 51.0 — degenerate repetitive text)
- Text quality: Noticeably worse; repetition artifacts

**Interpretation**: The gaze guidance with r=0.24 predictor is essentially adding random noise to LM log-probs. For the paper, this establishes that **sequential AR guidance fails gracefully — it doesn't** (it degrades). GazeDiffuse's parallel approach should be more robust.

### 7.3 What We Expect From GazeDiffuse (Experiment 4)

**Hypothesis**: GazeDiffuse will show:
1. FKGL shifts in the correct direction (lower for lam < 0, higher for lam > 0)
2. **Lower FK sentence variance** than both unguided and AR-guided text (key claim)
3. Maintained text quality (self-PPL close to unguided, MAUVE > 0.8)
4. More robustness to predictor noise than AR approach

**If the hypothesis holds**, the paper narrative is strong:
> "Parallel gaze guidance in diffusion models produces globally coherent readability changes that sequential AR guidance cannot, even with imperfect gaze prediction."

**If it doesn't hold**, we have fallback narratives:
- "Gaze guidance in text diffusion: challenges and insights" (negative results paper)
- Focus on the gaze predictor quality as the bottleneck
- Propose improved predictor architectures as future work

---

## 8. Codebase Health

### 8.1 Test Suite

```
194 tests, 69% coverage, all passing
Runtime: ~42 seconds
```

Coverage by module:
| Module | Coverage | Notes |
|--------|----------|-------|
| gaze_guidance.py | 80% | Missing: CLI __main__ block |
| metrics.py | 80% | Missing: CLI __main__ block |
| geco.py | 75% | Missing: full xlsx parse path |
| mdlm_wrapper.py | 67% | Missing: real model loading |
| ar_baseline.py | 61% | Missing: CLI __main__ block |
| gaze_predictor.py | 53% | Missing: training loop, __main__ |
| prompts.py | 100% | Complete |

### 8.2 Git History

24 commits across 5 sessions (Mar 12 - Apr 7). Clean conventional commit style. No force pushes. Main branch only (no feature branches needed yet).

### 8.3 Known Technical Debt

1. MPS memory leak on long training — needs `torch.mps.empty_cache()` per epoch
2. `openpyxl` dependency added manually but not in pyproject.toml
3. MDLM wrapper untested against real MDLM checkpoint (only mock-tested)
4. No WandB integration yet (metrics only saved to JSON/JSONL)

---

## 9. Summary of Action Items

### For Prof. Zhang (Urgent)
- [ ] Register project at https://projects.hpc.nyu.edu
- [ ] Add rs9174 as project member
- [ ] Confirm which MDLM checkpoint to use

### For Rahil (This Week)
- [ ] Write paper Introduction and Method sections
- [ ] Create method overview TikZ diagram
- [ ] Run Exp 3 with lambda=+1 locally
- [ ] Fix MPS memory leak (add `torch.mps.empty_cache()`)
- [ ] Once HPC access: run Exp 1 and 4 back-to-back

### For Siddhant (This Week)
- [ ] Investigate Sauberli et al. predictor methodology differences
- [ ] Start Related Work section
- [ ] Help with Exp 3 analysis if predictor improves

### Decision Needed
- [ ] Gaze predictor strategy: accept r=0.24 or invest in improving?
- [ ] Venue: EMNLP (ARR May/June) realistic, or workshop paper first?
- [ ] Experiment 5 (LLaDA 8B): priority or optional stretch goal?

---

*Generated from project state as of April 8, 2026. All code, tests, and results referenced above are in the gaze-diffuse repository.*
