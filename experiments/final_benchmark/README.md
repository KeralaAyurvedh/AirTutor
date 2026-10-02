# AirTutor Authoritative Final Benchmark Record

This directory records the reproducible, verified final benchmark execution for AirTutor, evaluating tabular Q-learning against standard pedagogical baselines on a strictly held-out simulated learner population.

## 1. Provenance & Execution Environment

- **Benchmark Command:** `python src/evaluation/compare_teachers.py`
- **Execution Date:** 2026-10-02 21:46:52 UTC+05:30
- **Python Version:** 3.13.14 (CPython)
- **Operating System:** Windows 11 (AMD64)
- **Key Dependencies:**
  - `numpy==2.2.6`
  - `pandas==2.3.3`
  - `matplotlib==3.10.7`
  - `pytest==9.0.2`
  - `torch==2.9.1` (optional DQN extension)
- **Primary Source Artifacts:** `experiments/stage3_benchmark_20261002_214652/`

---

## 2. Experimental Configuration

- **Curriculum:** `data/curriculum/gate_programming_fundamentals.json` (6 concepts with prerequisite DAG)
- **Training Population:** $N_{\text{train}} = 300$ simulated learners, seeds $[0, 300)$
- **Evaluation Population:** $N_{\text{eval}} = 100$ held-out simulated learners, seeds $[10000, 10100)$
  - *Strict Invariant:* $\text{train\_seeds} \cap \text{eval\_seeds} = \emptyset$ (enforced via assertions)
- **RL Training Seeds:** $[42, 101, 2024, 777, 999]$ (5 independent random seeds)
- **Max Steps per Episode:** 30 steps
- **Reward Formulation:** Reward v1 ($r_t = \hat{M}(c, t+1) - \hat{M}(c, t)$)
- **State Representation:** 5 observable features ($|S| = 320$ states, 1,600 Q-values)
  - Current concept mastery bucket (5 buckets)
  - Last response correctness (2 buckets)
  - Recent accuracy over last 3 attempts (4 buckets)
  - Consecutive error streak (4 buckets)
  - Prior step hint indicator (2 buckets)
- **Action Space:** 5 discrete pedagogical actions (`EXPLAIN`, `EXAMPLE`, `HINT`, `PRACTICE`, `REVIEW_PREREQUISITE`)

---

## 3. Evaluated Teachers & Baselines

1. **Random Teacher:** Uniform random action selection over $|A|=5$ at every step.
2. **Fixed Teacher:** Static deterministic curriculum traversal alternating `EXPLAIN` $\rightarrow$ `PRACTICE` per concept.
3. **Rule Teacher:** Cognitive-heuristic baseline implementing expert pedagogical rules:
   - If weakest prerequisite mastery $< 0.40$ $\rightarrow$ `REVIEW_PREREQUISITE`
   - If current concept mastery $< 0.40$ $\rightarrow$ `EXPLAIN`
   - If previous response was incorrect $\rightarrow$ `HINT`
   - If current concept mastery in $[0.40, 0.70)$ $\rightarrow$ `EXAMPLE`
   - If current concept mastery $\ge 0.70$ $\rightarrow$ `PRACTICE`
4. **RL (Tabular Q-Agent):** Off-policy TD control ($Q(s, a)$ updated via Bellman equation, $\alpha=0.10, \gamma=0.95$). Evaluated greedily ($\epsilon=0$) across 5 trained seeds ($5 \times 100 = 500$ evaluation episodes).

---

## 4. Benchmark Results

Evaluated across the 100 held-out learner configurations:

| Metric | Random | Fixed (Curriculum) | Rule Baseline | RL (Tabular Q-Agent) |
|---|---|---|---|---|
| **Observable Gain ($\Delta \hat{M}$)** | +0.019 $\pm$ 0.055 [$\pm$0.011] | -0.057 $\pm$ 0.034 [$\pm$0.007] | +0.052 $\pm$ 0.104 [$\pm$0.020] | **+0.115 $\pm$ 0.077 [$\pm$0.007]** |
| **True Latent Gain ($\Delta M$)** | +0.084 $\pm$ 0.110 [$\pm$0.022] | +0.069 $\pm$ 0.131 [$\pm$0.026] | **+0.263 $\pm$ 0.182 [$\pm$0.036]** | +0.228 $\pm$ 0.171 [$\pm$0.015] |
| **Final Mastery** | 0.119 $\pm$ 0.055 [$\pm$0.011] | 0.043 $\pm$ 0.034 [$\pm$0.007] | 0.152 $\pm$ 0.104 [$\pm$0.020] | **0.215 $\pm$ 0.077 [$\pm$0.007]** |
| **Steps to Mastery** | 30.0 $\pm$ 0.0 [$\pm$0.0] | 30.0 $\pm$ 0.0 [$\pm$0.0] | 30.0 $\pm$ 0.0 [$\pm$0.0] | 30.0 $\pm$ 0.0 [$\pm$0.0] |
| **Efficiency (Gain / Step)** | 0.0006 [$\pm$0.0004] | -0.0019 [$\pm$0.0002] | 0.0017 [$\pm$0.0007] | **0.0038 [$\pm$0.0002]** |
| **Hint Usage Rate** | 21.8% $\pm$ 7.2% | 0.0% $\pm$ 0.0% | 6.3% $\pm$ 3.7% | 13.3% $\pm$ 8.4% |
| **Action Entropy (bits)** | 2.230 $\pm$ 0.069 | 1.000 $\pm$ 0.000 | 1.077 $\pm$ 0.251 | 1.512 $\pm$ 0.406 |

*Values represent Mean $\pm$ Standard Deviation [95% Confidence Interval half-width].*

---

## 5. Scientific Findings & Interpretation

1. **Observable Gain Advantage:** The tabular Q-learning agent achieves $+0.115$ observable learning gain, more than double the Rule baseline ($+0.052$) and substantially outperforming the Fixed ($-0.057$) and Random ($+0.019$) baselines.
2. **Intervention Efficiency:** The RL policy achieves $0.0038$ gain per step, compared to $0.0017$ for Rule and $-0.0019$ for Fixed, demonstrating superior targeting of interventions to student states.
3. **The Observable vs. True Latent Tradeoff:**
   - Under Reward v1, the RL agent optimizes strictly on the observable estimator $\hat{M}$. Under the POMDP information barrier, the RL agent never observes the student's true continuous latent mastery $M(c, t)$ or latent parameter vector $\theta$.
   - The Rule teacher achieves higher true latent gain ($+0.263$ vs $+0.228$) because its hand-crafted rules enforce frequent `PRACTICE` and remedial reviews that happen to align well with the simulator's latent dynamics.
   - This contrast demonstrates an authentic POMDP finding: optimizing observable metrics in partially observable environments does not guarantee identical rank ordering on unobservable latent variables.
