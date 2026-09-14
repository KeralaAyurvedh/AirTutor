# AirTutor — Experimental Methodology

Authoritative Spec Cross-Reference: [`docs/research_roadmap_v2.md`](research_roadmap_v2.md)

This document provides a concise prose walkthrough of the experimental design, evaluation methodology, and empirical findings of AirTutor. Per project standards, all theoretical formalisms, mathematical derivations, and architecture choices cross-reference specific sections of `docs/research_roadmap_v2.md` rather than re-explaining them.

---

## Roadmap Section Cross-Reference Matrix

| Topic Area | Roadmap Section | Implementation File | Primary Invariant / Research Function |
|---|---|---|---|
| **Mathematical POMDP Formulation** | [Sec. 7](research_roadmap_v2.md#7-mathematical-rl-formulation) | `src/rl/q_learning.py` | State $S$, Action $A$, Transition $T$, Reward $R$, Discount $\gamma=0.90$. |
| **Observable Learner Model** | [Sec. 8](research_roadmap_v2.md#8-learner-model) | `src/learner_model/mastery.py` | Empirical $\hat{M}(c, t) \in [0, 1]$; updates via $\eta=0.20$, $\delta=0.01$. |
| **Parametric Learner Population** | [Sec. 9](research_roadmap_v2.md#9-parametric-learner-population) | `src/simulator/population.py` | Hidden $\theta = [\alpha, \beta, \gamma_c, h, \tau, \sigma]$; truncated normals. |
| **Hidden vs. Observable Split** | [Sec. 10](research_roadmap_v2.md#10-hidden-vs-observable-learner-state) | `src/environment/state_encoder.py` | Policy conditions exclusively on $O_t$; $\theta$ is strictly inaccessible. |
| **Simulator Dynamics & Sanity** | [Sec. 11–12](research_roadmap_v2.md#11-simulator-dynamics) | `src/simulator/dynamics.py` | Continuous $M(c, t)$; 5 automated mathematical sanity checks. |
| **Observable State Representation** | [Sec. 13](research_roadmap_v2.md#13-observable-state-representation) | `src/environment/state_encoder.py` | 5 toggleable features: $5 \times 2 \times 4 \times 4 \times 2 = 320$ discrete states. |
| **Pedagogical Action Space** | [Sec. 14](research_roadmap_v2.md#14-pedagogical-action-space) | `src/environment/actions.py` | 5 discrete interventions: `EXPLAIN`, `EXAMPLE`, `HINT`, `PRACTICE`, `REVIEW`. |
| **Reward Formulations (v1, v2, v3)** | [Sec. 15](research_roadmap_v2.md#15-reward-design) | `src/environment/rewards.py` | v1 (mastery gain), v2 (+retention, -cost), v3 (+transfer, -penalties). |
| **Baseline Pedagogical Teachers** | [Sec. 16](research_roadmap_v2.md#16-baseline-teachers) | `src/teachers/` | `RandomTeacher`, `FixedCurriculumTeacher`, `RuleBasedTeacher`. |
| **Disjoint Training / Evaluation** | [Sec. 17](research_roadmap_v2.md#17-training--evaluation-protocol) | `src/rl/train.py`, `src/evaluation/` | Train seeds $[0, 300)$; Held-out eval seeds $[10000, 10100)$. |
| **Research Metrics & Statistics** | [Sec. 19–20](research_roadmap_v2.md#19-metrics) | `src/evaluation/metrics.py`, `src/evaluation/stats.py` | 6 core metrics, Welch's $t$-test, Wilcoxon, Cohen's $d$, 95% bootstrap CIs. |
| **Robustness & Sensitivity Sweep** | [Sec. 21](research_roadmap_v2.md#21-robustness-and-sensitivity-analysis) | `src/evaluation/sensitivity.py` | 12-axis perturbation testing across $\pm 20\%$ parameter shifts. |
| **Systematic Ablation Studies** | [Sec. 22](research_roadmap_v2.md#22-ablation-studies) | `src/evaluation/ablations.py` | Reward ablations, state feature ablations, population diversity. |
| **Decoupled LLM Layer** | [Sec. 24–27](research_roadmap_v2.md#24-llm-abstraction-architecture) | `src/llm/` | `LLMProvider` interface; Gemini, OpenAI-compatible, Offline template. |
| **Extensible Curriculum Schema** | [Sec. 28](research_roadmap_v2.md#28-curriculum-architecture) | `src/environment/curriculum.py` | JSON-driven curriculum; dynamic validation via `pydantic`/schema. |
| **System Architectural Decoupling** | [Sec. 29](research_roadmap_v2.md#29-research-core-vs-delivery-vs-demo-layer) | `GEMINI.md` | Research Core $\perp$ Delivery Layer $\perp$ Demo Layer (`app/main.py`). |
| **Terminology Standards** | [Sec. 39](research_roadmap_v2.md#39-terminology-standards) | Entire repository | "simulated learner", "transparent policy inspection", "preliminary evidence". |
| **Limitations & Future Work** | [Sec. 40](research_roadmap_v2.md#40-future-research-extensions) | `README.md`, Section 4 below | BKT knowledge tracing, personalization, unseen adaptation, real pilot. |

## 1. System Architecture & Core Research Identity

AirTutor investigates whether reinforcement learning can learn adaptive pedagogical intervention policies that outperform structured heuristic teaching baselines under partial observability.

```
+-------------------------------------------------------------------------+
| SIMULATOR (Ground Truth)                                                |
| Latent traits theta = [alpha, beta, gamma, h, tau, sigma]                |
| True continuous mastery M(c, t) in [0.0, 1.0]                           |
+------------------------------------+------------------------------------+
                                     | Response Correctness
                                     v
+-------------------------------------------------------------------------+
| OBSERVABLE BOUNDARY (GEMINI.md Non-Negotiable Constraint)               |
| Observable Learner Model: M_hat(c, t) updated via observable delta      |
| Modular State Encoder: 320 discrete states (no theta or true M visible) |
+------------------------------------+------------------------------------+
                                     | Observable State Index
                                     v
+-------------------------------------------------------------------------+
| TABULAR Q-LEARNING AGENT                                                |
| Policy pi(a | O_t) selects action a in {EXPLAIN, EXAMPLE, HINT,        |
|                                         PRACTICE, REVIEW_PREREQUISITE}  |
+------------------------------------+------------------------------------+
                                     | Pedagogical Action
                                     v
+-------------------------------------------------------------------------+
| LLM DELIVERY LAYER (Replaceable renderer, strictly no decision-making)  |
+-------------------------------------------------------------------------+
```

---

## 2. Experimental Benchmark Protocol (Stage 3)

1. **Disjoint Population Partitioning (Roadmap Sec. 17):**
   - Training cohort: $N_{\text{train}} = 300$ simulated learners sampled from seed range $[0, 300)$.
   - Evaluation cohort: $N_{\text{eval}} = 100$ unseen simulated learners sampled from disjoint seed range $[10000, 10100)$.
   - Data leakage between training and evaluation is strictly prevented by assertion.
2. **Multi-Seed Training (Roadmap Sec. 20):**
   - RL agents are trained across independent random seeds (affecting epsilon-greedy exploration and action tie-breaking) to report distributions (mean $\pm$ std, 95% confidence intervals).
3. **Standing Rule on Honest Reporting (`GEMINI.md`):**
   - Results are reported plainly as observed. Baseline strength, reward formulations, and simulator dynamics are never tuned post-hoc to manufacture an artificial win.

---

## 3. Ablations & Robustness (Stage 4 Findings)

### 3.1 Reward Formulation Ablation (Roadmap Sec. 15)

Reward was evaluated across three formal formulations:
- **Reward v1:** $r_t = \hat{M}(c, t+1) - \hat{M}(c, t)$ (immediate observable mastery gain on current concept).
- **Reward v2:** $r_t = (\hat{M}(c, t+1) - \hat{M}(c, t)) + \lambda_{\text{ret}} \cdot \text{retention\_signal} - \lambda_{\text{cost}} \cdot \text{intervention\_cost}$ ($\lambda_{\text{ret}} = 0.50$, $\lambda_{\text{cost}} = 0.015$).
- **Reward v3:** $r_t = r_{v2} + \text{transfer\_term} - \text{excessive\_hint\_penalty} - \text{wasted\_step\_penalty}$.

#### Quantitative Outcomes on Held-Out Population:

| Reward Version | Observable Learning Gain | True Latent Learning Gain | Intervention Efficiency |
|---|---|---|---|
| **Reward v1** | **+0.089 $\pm$ 0.074** | **+0.276 $\pm$ 0.146** | 0.0030 |
| **Reward v2** | +0.067 $\pm$ 0.070 | +0.246 $\pm$ 0.177 | 0.0022 |
| **Reward v3** | +0.078 $\pm$ 0.078 | +0.254 $\pm$ 0.184 | 0.0026 |

#### Qualitative Action Preference Shifts:

Under Reward v1, the agent optimizes purely for immediate observable delta, causing it to rely heavily on `EXPLAIN` (61.4%) and `EXAMPLE` (26.1%), with almost negligible `HINT` (1.1%) and `PRACTICE` (1.7%).

Under Reward v2 and v3, penalizing intervention cost and drilling on unmastered concepts produces a marked qualitative shift:
- **`PRACTICE`** frequency increases six-fold from 1.7% in v1 to **10.8%** in v2 and **7.3%** in v3.
- **`HINT`** frequency rises from 1.1% in v1 to **4.0%** in v2 and **6.2%** in v3, deployed judiciously when students struggle.
- **`EXPLAIN`** drops from 61.4% down to **51.6%** in v3, yielding a far more balanced and pedagogically sound intervention profile across the curriculum.

---

### 3.2 State Representation Ablation (Roadmap Sec. 22)

To identify which observable features are load-bearing, three state representations were compared under identical training and held-out evaluation conditions:

| State Configuration | State Count | Observable Gain | True Latent Gain | Efficiency | $\Delta$ Obs Gain vs Full |
|---|---|---|---|---|---|
| **Full State (v1)** | 320 | +0.089 $\pm$ 0.074 | +0.276 $\pm$ 0.146 | 0.0030 | Baseline |
| **No-Hint** (`use_hint_used=False`) | 160 | +0.102 $\pm$ 0.074 | +0.251 $\pm$ 0.158 | 0.0034 | **+0.013 (Obs) / -0.025 (True)** |
| **No-Accuracy** (`use_recent_accuracy=False`) | 80 | +0.066 $\pm$ 0.075 | +0.264 $\pm$ 0.150 | 0.0022 | **-0.023 (Obs) / -0.011 (True)** |

#### Key Insights:
1. **Recent Accuracy is Load-Bearing:** Removing recent accuracy causes observable learning gain to plummet by $-0.023$ and efficiency to drop from 0.0030 to 0.0022. The agent loses the ability to distinguish temporary slips from entrenched misconceptions.
2. **Hint Awareness Prevents False Optimism:** Removing the hint feature creates an apparent increase in observable gain (+0.102) but a substantial decrease in true ground-truth gain (+0.251 vs +0.276). Without knowing a hint was provided, the agent overestimates learner independence, validating the inclusion of `hint_used_last_step` in the state tuple.

---

### 3.3 Learner Population Diversity Ablation (Roadmap Sec. 22)

We evaluated whether policy generalization breaks when trained on an artificially narrow population distribution versus the full diverse distribution:

| Training Population | Evaluated On | Observable Gain | True Latent Gain | Generalization Gap |
|---|---|---|---|---|
| **Wide Population (Default)** | Unseen Wide Learners | **+0.089 $\pm$ 0.074** | **+0.276 $\pm$ 0.146** | Baseline |
| **Narrow Population ($\pm 10\%$)** | Unseen Wide Learners | +0.076 $\pm$ 0.072 | +0.255 $\pm$ 0.160 | **-0.013 (Obs) / -0.021 (True)** |

**Finding:** An agent trained on homogeneous learners suffers a measurable performance drop on heterogeneous populations. Exposing the tabular Q-agent to diverse parametric learners during training is critical for robust policy learning.

---

### 3.4 Robustness & Sensitivity Sweep (Roadmap Sec. 21)

We perturbed 12 distinct simulator parameter axes one at a time and evaluated both RL and the competitive Rule baseline on 100 held-out learners:

| Perturbation Axis | Description | RL Gain | Rule Gain | RL Advantage ($\Delta$) | Status |
|---|---|---|---|---|---|
| **Baseline** | Default distributions | +0.089 | -0.037 | **+0.126** | **Holds** |
| **Alpha -20%** | Slower learning rates | +0.079 | -0.043 | **+0.121** | **Holds** |
| **Alpha +20%** | Faster learning rates | +0.103 | -0.032 | **+0.134** | **Holds** |
| **Beta -20%** | Slower forgetting | +0.092 | -0.036 | **+0.128** | **Holds** |
| **Beta +20%** | Faster forgetting | +0.088 | -0.038 | **+0.126** | **Holds** |
| **Noise Sigma -50%** | Lower response noise | +0.091 | -0.037 | **+0.129** | **Holds** |
| **Noise Sigma +50%** | Higher response noise | +0.090 | -0.040 | **+0.130** | **Holds** |
| **A_effect -20%** | Weaker pedagogical effect | +0.076 | -0.043 | **+0.119** | **Holds** |
| **A_effect +20%** | Stronger pedagogical effect | +0.106 | -0.031 | **+0.137** | **Holds** |
| **High-h Skew** | Hint dependence in [0.5, 1.0] | +0.089 | -0.032 | **+0.121** | **Holds** |
| **Curriculum Harder** | Difficulty offset +1.0 | +0.088 | -0.015 | **+0.104** | **Holds** |
| **Curriculum Easier** | Difficulty offset -1.0 | +0.095 | -0.032 | **+0.127** | **Holds** |

#### Robustness Finding:
The RL pedagogical advantage over the Rule baseline **reliably holds across all 12 perturbation axes** ($\Delta \ge +0.104$, substantially above the $+0.020$ threshold).
The single condition that most tightened the RL advantage was **Curriculum Harder (+1.0 difficulty offset)**, where headroom narrowed to $+0.104$. Under elevated difficulty, all teachers face higher failure rates, constraining the margin for adaptive differentiation.

---

## 4. Threats to Validity & Limitations

1. **Simulated vs. Human Learners:** The six-trait parametric simulation is designed for mathematical smoothness and monotonicity, not empirical cognitive fidelity. Findings should be understood as preliminary simulated evidence.
2. **State Discretization Boundary:** Tabular Q-learning is bounded by the discrete bucket resolution (320 states). Scaling to multi-course curricula will require function approximation (DQN) as future work.
3. **Decoupled Delivery Independence:** Content rendering by the LLM layer is decoupled from policy selection. Imperfect natural language rendering does not impact the mathematical validity of the RL intervention decisions.
