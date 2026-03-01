# CS263 Project Status

## Progress (2026-03-01)

### Completed
- [x] MMLU → MMLU-Pro (10-choice, A-J)
- [x] Gemini API → Vertex AI authentication
- [x] Full run completed (`results_full/`, 3 turns)
- [x] Per-subject, per-strategy metrics added
- [x] 3 new gaslighting strategies: `step_by_step`, `false_premise`, `targeted_attack`
- [x] `--mode compare` added (3 judge model comparison)
- [x] Compare mode re-run completed (flash + 2.5-pro results, committed & pushed)
- [x] `num_turns` 3 → 5 (full_config)
- [x] Full run re-run completed (5 turns, `results_full/`)

---

## Experiment Results

### Full Run (`results_full/`) — 3 turns vs 5 turns

| Metric | 3 turns | 5 turns |
|--------|---------|---------|
| Overall Flip Rate | 1.33% | **2.08%** |
| Targeted Flip Rate | 0% | 0% |
| RUP Score | 96.7% | 96.7% |
| Baseline Accuracy | 60% | 60% |
| Order Effect | — | +0.005 |

**Per-strategy (5 turns)**
| Strategy | Flip Rate | Targeted FR | ΔConf |
|----------|-----------|-------------|-------|
| authority | 3.3% | 0% | +0.40 |
| combined | 3.3% | 0% | +0.07 |
| confidence | 3.3% | 0% | +0.07 |
| step_by_step | 3.3% | 0% | -0.07 |
| targeted_attack | 3.3% | 0% | +0.17 |
| emotional | 0% | 0% | -0.10 |
| false_premise | 0% | 0% | +0.10 |
| jargon | 0% | 0% | -0.17 |

**Per-subject (5 turns)**
| Subject | Flip Rate | ΔConf |
|---------|-----------|-------|
| economics | 6.9% | +0.69 |
| philosophy | 0% | +0.46 |
| psychology | 0% | +0.50 |
| health | 0% | +0.11 |
| math | 0% | -1.23 |

**Key findings**
- Flip rate barely changes with more turns (1.33% → 2.08%) — model is generally robust to gaslighting
- Targeted flip rate stays at 0% — steering toward a specific answer consistently fails
- Economics is the most vulnerable subject (6.9%) — likely due to interpretive ambiguity
- Math confidence drops after gaslighting (ΔConf = -1.23) — model becomes less confident rather than switching
- Authority strategy produces the largest confidence shift (+0.40)

---

### Compare Mode — 3 Judge Models (30Q × 3 strategies × 3 turns)

| Judge | Baseline Acc | Flip Rate | Targeted FR | RUP Score | Order Effect |
|-------|-------------|-----------|-------------|-----------|--------------|
| gemini-2.0-flash | 46.7% | 7.1% | 4.8% | 92.9% | +0.055 (Truth-first favored) |
| gemini-2.5-pro | 46.7% | 7.1% | 4.8% | 92.9% | -0.062 (Gaslight-first favored) |
| gemini-1.5-pro-002 | 0% (failed) | — | — | — | — |

**Per-strategy (flash judge)**
| Strategy | Flip Rate | Targeted FR | ΔConf |
|----------|-----------|-------------|-------|
| combined | 7.1% | 7.1% | +0.36 |
| step_by_step | 7.1% | 7.1% | +0.21 |
| targeted_attack | 7.1% | 0% | +0.21 |

**Per-strategy (2.5-pro judge)**
| Strategy | Flip Rate | Targeted FR | ΔConf |
|----------|-----------|-------------|-------|
| combined | 7.1% | 7.1% | 0.00 |
| step_by_step | 7.1% | 7.1% | 0.00 |
| targeted_attack | 7.1% | 0% | 0.00 |

**Per-subject (flash judge)**
| Subject | Flip Rate | Targeted FR | ΔConf | N |
|---------|-----------|-------------|-------|---|
| law | **66.7%** | 66.7% | +1.00 | 3 |
| math | 8.3% | 0% | 0.00 | 12 |
| economics | 0% | 0% | +1.33 | 3 |
| health | 0% | 0% | +0.67 | 12 |
| philosophy | 0% | 0% | -1.00 | 3 |
| psychology | 0% | 0% | -0.11 | 9 |

**Per-subject (2.5-pro judge)**
| Subject | Flip Rate | Targeted FR | ΔConf | N |
|---------|-----------|-------------|-------|---|
| law | **50.0%** | 33.3% | 0.00 | 6 |
| all others | 0% | 0% | 0.00 | — |

**Key findings**
- flash vs 2.5-pro: identical flip/targeted flip rates; 2.5-pro shows ΔConf=0 across all strategies (possible parsing issue)
- Law is the most vulnerable subject — 66.7% (flash) and 50.0% (2.5-pro) flip rates
- `targeted_attack` flips answers but targeted flip = 0% — cannot steer toward a specific answer
- Order effect direction differs between judges (flash: Truth-first favored; 2.5-pro: Gaslight-first favored)
- 1.5-pro-002: baseline accuracy = 0% → parsing failure, Phase 2 skipped entirely

---

## TODO

### 1. Analysis & Paper Writing
- Summarize 3 turns vs 5 turns comparison
- Integrate compare mode results (flash vs 2.5-pro) into overall analysis
- Write results section for paper/report

### 2. Optional Follow-up Experiments
- Deep dive into law subject vulnerability
- Investigate 2.5-pro ΔConf=0 (parsing issue vs model behavior)
- Explore strategies to improve targeted flip rate

---

## File Structure
```
debate_pipeline_gemini.py   # main pipeline
results_full/               # full run results (8 strategies × 6 subjects, 5 turns)
results_compare_flash/      # flash judge compare results ✓
results_compare_1.5pro/     # 1.5-pro judge results (failed, baseline_accuracy=0)
results_compare_2.5pro/     # 2.5-pro judge compare results ✓
```

## Branch
- `han/gemini` (current working branch, in sync with origin)
- Latest commit: `ec6946b Add compare mode results for flash and 2.5-pro judge models`
