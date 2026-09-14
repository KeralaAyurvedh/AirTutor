# AirTutor — An RL-Based Adaptive Pedagogical Decision System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: Pytest](https://img.shields.io/badge/tests-68%20passed-brightgreen.svg)](tests/)
[![Architecture: Decoupled](https://img.shields.io/badge/RL%20Core-Zero%20LLM%20Dependency-informational.svg)](src/)

AirTutor is a research artifact investigating reinforcement learning (RL) for adaptive pedagogical intervention selection. The central contribution is a policy that decides **which** pedagogical intervention to deploy based on an observable student history, while a decoupled Large Language Model (LLM) acts purely as a delivery layer to render that decision into natural-language dialogue.

Authoritative Specification: [`docs/research_roadmap_v2.md`](docs/research_roadmap_v2.md)  
Experimental Methodology: [`docs/methodology.md`](docs/methodology.md)

---

## 1. Problem Statement

Most adaptive learning platforms focus either on *predicting* student attrition or determining *which curriculum topic* to present next. AirTutor investigates a distinct, narrower decision layer:

> **Given an observable representation of a student's current learning state, which pedagogical intervention (`EXPLAIN`, `EXAMPLE`, `HINT`, `PRACTICE`, or `REVIEW_PREREQUISITE`) should be selected, and can that selection be learned via reinforcement learning rather than hand-authored by static heuristics?**

The LLM is strictly a delivery mechanism—it is never the pedagogical decision-maker.

---

## 2. Decoupled System Architecture

The codebase enforces strict modular boundaries (`GEMINI.md` & Roadmap Sec. 29):

```
+-----------------------------------------------------------------------------------+
| SIMULATOR (Ground Truth)                                                          |
| Latent parameters: theta = [alpha, beta, gamma_c, h, tau, sigma]                   |
| True continuous mastery: M(c, t) in [0.0, 1.0]                                    |
+----------------------------------------+------------------------------------------+
                                         | Response Correctness
                                         v
+-----------------------------------------------------------------------------------+
| OBSERVABLE BOUNDARY (GEMINI.md Invariant: theta & true M are strictly hidden)     |
| Observable Learner Model: M_hat(c, t) in [0.0, 1.0] (eta=0.20, delta=0.01)        |
| State Encoder: 5 observable features -> 320 discrete states                       |
+----------------------------------------+------------------------------------------+
                                         | Observable State Index s in [0, 319]
                                         v
+-----------------------------------------------------------------------------------+
| TABULAR Q-LEARNING AGENT (Research Core)                                          |
| Q(s, a) updated via Bellman TD error: Q <- Q + alpha [r + gamma max Q - Q]        |
| Policy pi(s) selects action a in {EXPLAIN, EXAMPLE, HINT, PRACTICE, REVIEW}       |
+----------------------------------------+------------------------------------------+
                                         | Selected Action + Context
                                         v
+-----------------------------------------------------------------------------------+
| PROVIDER-AGNOSTIC LLM LAYER (Delivery Layer)                                      |
| Adapters: Google Gemini, OpenAI-Compatible, Offline Template (Zero-Config)        |
| Renders pedagogical action into structured natural language tutoring dialogue     |
+----------------------------------------+------------------------------------------+
                                         | Natural Language Output
                                         v
+-----------------------------------------------------------------------------------+
| STREAMLIT USER INTERFACE (Demo Layer - app/main.py)                               |
| Interactive tutoring, transparent policy inspection, benchmark comparisons        |
+-----------------------------------------------------------------------------------+
```

---

## 3. Formal System Specification

### 3.1 Observable State Space ($S$)
Per Roadmap Sec. 13, the policy conditions **exclusively** on 5 observable interaction features:
1. **Current Concept Mastery Bucket ($b_m$):** 5 discrete intervals ($[0.0, 0.2), \dots, [0.8, 1.0]$).
2. **Last Response Correctness ($c_{\text{last}}$):** Binary ($\{0, 1\}$).
3. **Recent Accuracy ($acc_3$):** Correct responses over last 3 attempts ($\{0, 1, 2, 3\}$).
4. **Consecutive Error Streak ($s_{\text{err}}$):** Capped error count ($\{0, 1, 2, 3+\}$).
5. **Hint Used on Prior Step ($h$):** Binary ($\{0, 1\}$).

$$\text{Total States } |S| = 5 \times 2 \times 4 \times 4 \times 2 = 320 \text{ states}$$

### 3.2 Pedagogical Action Space ($A$)
Per Roadmap Sec. 14, the policy selects among 5 discrete interventions:
- `EXPLAIN`: Conceptual exposition addressing underlying principles.
- `EXAMPLE`: Concrete, annotated code or application example.
- `HINT`: Immediate scaffolding hint for the active question.
- `PRACTICE`: Formative assessment question drilling current concept.
- `REVIEW_PREREQUISITE`: Remedial review of the weakest foundational prerequisite.

### 3.3 Reward Formulations ($R$)
Per Roadmap Sec. 15, reward formulations evaluate immediate versus balanced pedagogical efficiency:
- **Reward v1 (Mastery Gain):**
  $$r_t = \hat{M}(c, t+1) - \hat{M}(c, t)$$
- **Reward v2 (Retention & Cost Aware):**
  $$r_t = (\hat{M}(c, t+1) - \hat{M}(c, t)) + \lambda_{\text{ret}} \cdot \text{retention\_signal} - \lambda_{\text{cost}} \cdot \text{intervention\_cost}$$
  where $\lambda_{\text{ret}} = 0.50$ and $\lambda_{\text{cost}} = 0.015$.
- **Reward v3 (Transfer & Balanced Penalties):**
  $$r_t = r_{v2} + \text{transfer\_term} - \text{excessive\_hint\_penalty} - \text{wasted\_step\_penalty}$$

---

## 4. Setup & Execution

AirTutor is built for CPU-only execution with an 8 GB RAM target. No PyTorch, no GPU, and no external databases are required.

### 4.1 Installation
```bash
# Clone the repository
git clone https://github.com/your-username/AirTutor.git
cd AirTutor

# Create and activate virtual environment (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install pinned dependencies
pip install -r requirements.txt
```

*(On Linux / macOS, activate via `source .venv/bin/activate`)*

### 4.2 Running the Streamlit Application
Launch the interactive demo interface:
```bash
streamlit run app/main.py
```

The app runs out-of-the-box in two operational modes:
1. **Offline Mode (Zero API Keys):** If no `.env` exists or no API key is provided, the system seamlessly defaults to `OfflineTemplateAdapter`. Every tutoring interaction, question, policy rationale, and benchmark visualization functions completely headlessly with zero external network calls.
2. **Online LLM Mode:** To enable dynamic natural-language generation:
   - Create a `.env` file (see `.env.example`):
     ```bash
     cp .env.example .env
     ```
   - Add your API key:
     ```env
     LLM_PROVIDER=gemini
     GEMINI_API_KEY=your_gemini_api_key_here
     ```
     *(Or set `LLM_PROVIDER=openai_compatible` with `OPENAI_API_BASE` and `OPENAI_API_KEY`).*

### 4.3 Running Tests & Headless Experiments
```bash
# Run complete test suite (68 unit/integration/UI tests)
python -m pytest

# Run Stage 3 baseline evaluation (Random vs Fixed vs Rule vs RL)
python src/evaluation/compare_teachers.py

# Run Stage 4 ablations & sensitivity sweeps
python run_stage4_experiments.py
```

---

## 5. Empirical Results

Evaluated across $N_{\text{eval}} = 100$ held-out simulated learner configurations sampled from seed range $[10000, 10100)$, strictly disjoint from the training population ($[0, 300)$):

### 5.1 Held-Out Teacher Comparison (Stage 3 Benchmark)

| Pedagogical Teacher | Observable Gain $\Delta \hat{M}$ | True Latent Gain $\Delta M$ | Intervention Efficiency | Hint Rate | Action Entropy |
|---|---|---|---|---|---|
| **Random** | +0.0115 $\pm$ 0.0119 | +0.1657 $\pm$ 0.0296 | 0.0004 | 21.8% | 2.23 |
| **Fixed (Curriculum)** | +0.0179 $\pm$ 0.0126 | +0.2152 $\pm$ 0.0411 | 0.0006 | 0.0% | 1.00 |
| **Rule Baseline** | +0.0280 $\pm$ 0.0139 | +0.2339 $\pm$ 0.0308 | 0.0009 | 8.0% | 1.40 |
| **RL (Tabular Q-Agent)** | **+0.1309 $\pm$ 0.0067** | **+0.2989 $\pm$ 0.0167** | **0.0044** | **0.5%** | **1.37** |

*Statistical significance: RL vs. Rule Baseline $p < 10^{-15}$ (Welch's $t$-test & Wilcoxon signed-rank test), Cohen's $d = 1.396$.*

### 5.2 Reward Formulation Ablation (Stage 4)
- **Reward v1** optimizes purely for immediate observable delta, causing the policy to rely predominantly on `EXPLAIN` (61.4%) and `EXAMPLE` (26.1%), with almost negligible `HINT` (1.1%) and `PRACTICE` (1.7%).
- **Reward v2 & v3** penalize intervention costs and excessive hints, increasing `PRACTICE` to **10.8%** and `HINT` to **6.2%**, producing a more balanced pedagogical distribution.

### 5.3 12-Axis Robustness Sweep (Stage 4)
When systematically perturbing 12 simulator parameters ($\pm 20\%$ learning/forgetting rates, $\pm 50\%$ response noise, $\pm 20\%$ intervention effects, student hint skew, and curriculum difficulty offsets), the RL policy's advantage over the Rule baseline **reliably holds across all 12 axes** ($\Delta \ge +0.104$, well above the $+0.020$ threshold).

---

## 6. Limitations & Future Work

Per Roadmap Sec. 40, several research extensions are explicitly identified:

1. **Knowledge Tracing (Level 2):** The current observable learner model uses an empirical exponential moving average ($\eta=0.20, \delta=0.01$). Future work will replace this heuristic with Bayesian Knowledge Tracing (BKT) or Deep Knowledge Tracing (DKT) for calibrated mastery estimates.
2. **Personalized Policy Learning (Level 3):** The current policy learns a population-average strategy. Extending the state representation to incorporate observable learner profiles will enable personalized intervention routing.
3. **Unseen-Learner Adaptation (Level 4):** Rapid online meta-learning (e.g., contextual bandits or in-session belief updating) to adapt to an outlier student profile within 5–10 interaction steps.
4. **Counterfactual Evaluation (Level 5):** Off-policy evaluation (OPE) via doubly robust estimators on logged educational trajectory data.
5. **Real-User Pilot (Level 6):** Controlled human-subject trials under appropriate institutional review and informed consent.

---

## 7. Scientific Validity & Terminology Standards

Per Roadmap Sec. 39, AirTutor enforces strict terminology standards to avoid academic overclaiming:
- **"Simulated Learner"** is used consistently over "student" or "learner" to prevent any implication of human-subject empirical validation.
- **"Transparent Policy Inspection"** is used instead of "explainable AI", reflecting that tabular Q-learning decisions are inspectable table lookups rather than opaque black-box inferences.
- **"Preliminary Simulated Evidence"** is used rather than "proof" or "validation", reflecting that results are bounded by the parametric simulation environment.
