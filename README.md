# CS 263: Adversarial Multi-Agent Debate — Experiment I

Evaluating parametric robustness of LLMs under adversarial multi-agent debate using MMLU-Pro.

> **Paper:** *Evaluating Parametric Robustness and Persuasion via Adversarial Multi-Agent Interaction*
> **GitHub:** https://github.com/Han3373/CS263-Multi-Agent-Debate

---

## Overview

This repo contains the pipeline for **Experiment I**: cross-domain robustness analysis of Gemini 2.0 Flash under five adversarial persuasion strategies.

Three agents operate in a closed loop:
- **Truth Agent** — argues for the ground-truth answer using logical reasoning
- **Gaslighting Agent** — argues for an incorrect distractor using one of five persuasion strategies
- **Judge Agent** — observes the debate and renders a final answer with a confidence score (1–10)

The pipeline runs in three phases:
1. **Baseline** — Judge answers independently; only correct answers proceed
2. **Debate** — Truth and Gaslighting agents alternate for T turns (speaking order randomized)
3. **Judgment** — Judge re-answers; metrics are computed against the Phase 1 baseline

---

## Persuasion Strategies

| Strategy | Description |
|---|---|
| `authority` | Fabricates academic citations and expert opinions |
| `jargon` | Uses dense technical terminology to obscure reasoning |
| `confidence` | Frames the correct answer as a common misconception |
| `emotional` | Applies social pressure and peer-consensus framing |
| `combined` | Layers all four strategies simultaneously |

---

## Key Results

**Cross-domain experiment** (50Q × 5 strategies × T=5, Gemini 2.0 Flash):

| Subject | Flip Rate | ΔConf |
|---|---|---|
| Economics | 6.9% | +0.60 |
| Law | 4.2% | +0.89 |
| Psychology | 0.0% | +0.50 |
| Philosophy | 0.0% | +0.53 |
| Health | 0.0% | +0.11 |
| Math | 0.0% | −1.17 |

Two distinct failure modes emerge:
- **Confident Error** (law, economics): Judge flips to the wrong answer *and* increases confidence
- **Confidence Erosion** (math): Judge retains the correct answer but confidence drops (−1.17)

**Law deep-dive** (60Q × 5 strategies × T=3):

| Strategy | Flip Rate | ΔConf |
|---|---|---|
| Confidence | 10.8% | +0.86 |
| Jargon | 7.5% | +0.65 |
| Combined | 5.4% | +0.53 |
| Emotional | 2.7% | +0.42 |
| Authority | 0.0% | +0.35 |

The `confidence` strategy is most effective; fabricated `authority` citations backfire in legal domains.

---

## Setup

```bash
pip install google-genai google-cloud-aiplatform datasets
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="your-project-id"
```

## Usage

```bash
# Quick test
python debate_pipeline_gemini.py --mode test

# Full cross-domain run (50Q × 8 strategies × 6 subjects)
python debate_pipeline_gemini.py --mode full

# Law deep-dive (60Q × 5 strategies × 3 turns)
python debate_pipeline_gemini.py --mode law_deep
```

Results are saved to `results_full/`, `results_law_deep/`, etc.

---

## Metrics

- **Flip Rate (FR)**: fraction of initially-correct answers abandoned after debate
- **ΔConf**: mean confidence shift over correctly-answered questions (negative = erosion)
- **Order Effect (Δ_order)**: FR difference between gaslight-first vs. truth-first speaking order
