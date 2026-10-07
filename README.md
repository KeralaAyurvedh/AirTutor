# AirTutor — An RL-Based Adaptive Pedagogical Decision System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: Pytest](https://img.shields.io/badge/tests-86%20passed-brightgreen.svg)](tests/)
[![Architecture: Decoupled](https://img.shields.io/badge/RL%20Core-Zero%20LLM%20Dependency-informational.svg)](src/)

AirTutor is a research artifact investigating reinforcement learning (RL) for adaptive pedagogical intervention selection in a simulated learner environment. The central contribution is a policy that decides **which** pedagogical intervention to deploy based on an observable student history, while a decoupled Large Language Model (LLM) acts purely as a delivery layer to render that decision into natural-language dialogue.

**Authoritative Specification:** [`docs/research_roadmap_v2.md`](docs/research_roadmap_v2.md)  
**Experimental Methodology:** [`docs/methodology.md`](docs/methodology.md)  
**Final Benchmark Record:** [`experiments/final_benchmark/`](experiments/final_benchmark/)

---

## 1. Problem Statement & Research Scope

Most adaptive learning platforms focus either on *predicting* student attrition or determining *which curriculum topic* to present next. AirTutor investigates a distinct, narrower decision layer:

> **Given an observable representation of a student's current learning state, which pedagogical intervention (`EXPLAIN`, `EXAMPLE`, `HINT`, `PRACTICE`, or `REVIEW_PREREQUISITE`) should be selected, and can that selection be learned via reinforcement learning rather than hand-authored by static heuristics?**

The LLM is strictly a delivery mechanism—it never decides the pedagogical action.

---

## 2. Decoupled System Architecture & POMDP Information Barrier

The codebase enforces strict modular boundaries (`GEMINI.md` & Roadmap Sec. 29):

```
+-----------------------------------------------------------------------------------+
| SIMULATOR (Ground Truth - Latent Mechanics)                                       |
| Latent parameters: theta = [alpha, beta, gamma_c, h, tau, sigma]                   |
| True continuous mastery: M(c, t) in [0.0, 1.0]                                    |
+----------------------------------------+------------------------------------------+
                                         | Response Correctness (Binary outcome)
                                         v
+-----------------------------------------------------------------------------------+
| OBSERVABLE BOUNDARY (POMDP Barrier: theta & true M are strictly hidden)           |
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
+----------------------------------------+------------------------------------------+
```

### Strict Research Invariants
1. **POMDP Information Barrier:** The implemented information barrier is behaviorally tested for the verified execution paths, with dedicated tests preventing latent-state leakage into the tabular RL policy. The policy's state input **never** contains the latent parameter vector $\theta$ (learning rate $\alpha$, forgetting rate $\beta$, difficulty sensitivity $\gamma$, hint dependence $h$, transfer ability $\tau$, response noise $\sigma$) or true simulator-internal mastery $M(c, t)$. Only the observable feature vector is passed to the agent. Dedicated behavioral tests in `tests/test_pomdp_barrier.py` verify that differing latent states produce strictly identical observable states and policy action selections for both the tabular agent and the optional continuous 14-D DQN agent.
2. **Core Independence:** The research core (`src/environment`, `src/learner_model`, `src/simulator`, `src/rl`, `src/teachers`, `src/evaluation`) has zero Streamlit and zero LLM imports. It runs completely headlessly.
3. **Provider-Agnostic LLM Layer:** All LLM selection is config-driven via environment variables (`LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`, `LLM_BASE_URL`). Without configured API credentials, the application uses the offline template adapter by default; live external LLM calls were not part of this verification.

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

### 4.1 Dependency Architecture
- **Core Platform:** Tabular Q-learning, simulated environment, teacher baselines, and evaluation battery require **no PyTorch, no GPU, and no external databases**. Designed for lightweight CPU execution (8 GB RAM target).
- **Optional Extensions:** 
  - The Deep Q-Network (DQN) extension (`src/rl/dqn_agent.py`, `src/evaluation/run_dqn_benchmark.py`) requires PyTorch (`torch>=2.2.0`).
  - The RAG material ingestion pipeline (`src/llm/rag_engine.py`) requires `faiss-cpu`, `sentence-transformers`, and `pypdf`.
- Core dependencies in `requirements.txt` are version-pinned; optional extensions declare verified minimum versions (tested with `torch==2.9.1`, `faiss-cpu==1.15.1`, `sentence-transformers==6.1.0`, `pypdf==6.18.0`).

### 4.2 Quick Start
```bash
# 1. Clone repository
git clone https://github.com/your-username/AirTutor.git
cd AirTutor

# 2. Create and activate virtual environment (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# (On Linux / macOS: python3 -m venv .venv && source .venv/bin/activate)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run full test suite (86 passed)
python -m pytest

# 5. Run authoritative Stage 3 benchmark
python src/evaluation/compare_teachers.py

# 6. Launch interactive demo interface
streamlit run app/main.py
```

### 4.3 Operational Modes
1. **Offline Mode (Default):** Without configured API credentials, the application uses the offline template adapter by default; live external LLM calls were not part of this verification. Every tutoring interaction, question, policy rationale, and benchmark visualization functions headlessly using deterministic curriculum templates.
2. **Online LLM Mode (Optional):** To configure external LLM providers (live external API calls were not part of this test suite verification):
   - Copy configuration template:
     ```bash
     cp .env.example .env
     ```
   - Set your provider and API key:
     ```env
     LLM_PROVIDER=gemini
     GEMINI_API_KEY=your_gemini_api_key_here
     ```
     *(Or set `LLM_PROVIDER=openai_compatible` with `OPENAI_API_BASE` and `OPENAI_API_KEY`).*

---

## 5. Empirical Results & Findings

### 5.1 Held-Out Teacher Comparison (Stage 3 Benchmark)

Evaluated across $N_{\text{eval}} = 100$ held-out simulated learner configurations sampled from seed range $[10000, 10100)$, strictly disjoint from the training population $[0, 300)$:

| Pedagogical Teacher | Observable Gain $\Delta \hat{M}$ | True Latent Gain $\Delta M$ | Final Mastery | Intervention Efficiency | Hint Rate | Action Entropy |
|---|---|---|---|---|---|---|
| **Random** | +0.019 $\pm$ 0.055 [$\pm$0.011] | +0.084 $\pm$ 0.110 [$\pm$0.022] | 0.119 $\pm$ 0.055 [$\pm$0.011] | 0.0006 [$\pm$0.0004] | 21.8% | 2.230 bits |
| **Fixed (Curriculum)** | -0.057 $\pm$ 0.034 [$\pm$0.007] | +0.069 $\pm$ 0.131 [$\pm$0.026] | 0.043 $\pm$ 0.034 [$\pm$0.007] | -0.0019 [$\pm$0.0002] | 0.0% | 1.000 bits |
| **Rule Baseline** | +0.052 $\pm$ 0.104 [$\pm$0.020] | **+0.263 $\pm$ 0.182 [$\pm$0.036]** | 0.152 $\pm$ 0.104 [$\pm$0.020] | 0.0017 [$\pm$0.0007] | 6.3% | 1.077 bits |
| **RL (Tabular Q-Agent)** | **+0.115 $\pm$ 0.077 [$\pm$0.007]** | +0.228 $\pm$ 0.171 [$\pm$0.015] | **0.215 $\pm$ 0.077 [$\pm$0.007]** | **0.0038 [$\pm$0.0002]** | 13.3% | 1.512 bits |

*Results represent Mean $\pm$ Standard Deviation [95% Confidence Interval half-width] over 100 held-out evaluation learners across 5 independent RL seeds ($42, 101, 2024, 777, 999$). Artifacts stored at `experiments/final_benchmark/` (generated from benchmark source commit `bddc4a2`, verified valid across release commits `81de5f8` and `8dc37df` via zero diff in `src/`, `data/`, `configs/`).*

#### Key Scientific Insights:
1. **Observable Gain Superiority:** The learned RL policy achieves more than double the observable gain of the Rule baseline ($+0.115$ vs. $+0.052$, $p < 10^{-15}$ under Welch's $t$-test), and higher intervention efficiency ($0.0038$ vs. $0.0017$), demonstrating effective targeting of pedagogical actions based on observable interaction history.
2. **The Observable vs. True Latent Tradeoff:** The Rule baseline achieves higher true latent gain ($+0.263$ vs. $+0.228$) than RL under Reward v1. This finding illustrates a core property of partially observable environments: because the RL agent optimizes strictly against the observable mastery estimator $\hat{M}$ without access to latent $\theta$ or true continuous mastery $M(c, t)$, policies that maximize observable progress may differ from policies that maximize latent simulator dynamics.

### 5.2 Reward Formulation Ablation (Stage 4)
- **Reward v1** ($r_t = \Delta \hat{M}$) optimizes purely for immediate observable gain, causing the policy to rely predominantly on `EXPLAIN` (64.5%) and `EXAMPLE` (17.2%), with modest `HINT` (13.0%) and low `PRACTICE` (2.0%).
- **Reward v2 & v3** incorporate retention terms, step costs, and hint penalties, increasing formative assessment (`PRACTICE`) to **8.7%** and reducing hint overuse, producing a more pedagogically balanced intervention strategy.

### 5.3 12-Axis Sensitivity Sweep & Boundary Conditions (Stage 4)
When systematically perturbing 12 simulator parameter axes ($\pm 20\%$ learning/forgetting rates, $\pm 50\%$ response noise, $\pm 20\%$ intervention effects, high hint skew, and curriculum difficulty offsets):
- The RL advantage over the Rule baseline **holds across 11 of 12 parameter axes** ($\Delta \ge +0.052$, well above the $+0.020$ hypothesis threshold).
- **Reported Boundary Condition:** Under the **Harder Curriculum** condition ($+1.0$ difficulty offset), the RL advantage **vanishes** ($\Delta = -0.0075$, status: **Vanishes**). In this regime, elevated failure rates constrain adaptive differentiation, causing the Rule teacher's hand-crafted remedial heuristic (heavy `EXPLAIN` and `HINT` scaffolding) to match or slightly exceed unconstrained tabular exploration.

---

## 6. Project Structure

```
AirTutor/
├── app/
│   └── main.py                     # Streamlit presentation layer (interactive demo)
├── configs/
│   └── learner_population.json     # Externalized learner parameter distributions
├── data/
│   └── curriculum/                 # Curriculum definitions (JSON DAGs)
│       ├── gate_programming_fundamentals.json
│       └── object_oriented_programming.json
├── docs/
│   ├── research_roadmap_v2.md      # Authoritative 43-section research specification
│   ├── methodology.md              # Experimental methodology & evaluation protocols
│   ├── AI_ASSISTANCE_LOG.md        # Running audit trail of AI assistance
│   └── AirTutor_Comprehensive_Project_Report.pdf  # Compiled research report
├── experiments/
│   ├── final_benchmark/            # Current authoritative release benchmark artifacts
│   │   ├── README.md               # Detailed provenance, configuration, and metrics
│   │   ├── config.json             # Benchmark configuration snapshot
│   │   ├── summary.json            # Full statistical output
│   │   └── metrics.csv             # Tabular metrics for downstream analysis
│   ├── dqn_benchmark/              # Optional Deep Q-Network comparative benchmark
│   └── *.png                       # Publication figures (learning curves, ablations)
├── src/
│   ├── environment/                # State encoder, actions, curriculum, rewards
│   ├── learner_model/              # Observable mastery tracker, BKT belief tracker
│   ├── simulator/                  # Parametric learner dynamics, population sampling
│   ├── rl/                         # Tabular Q-learning agent, training loop, DQN agent
│   ├── teachers/                   # Random, Fixed, Rule, and Base teacher baselines
│   ├── evaluation/                 # Metrics, comparisons, ablations, plotting scripts
│   └── llm/                        # Provider-agnostic delivery layer & adapters
├── tests/                          # 86 unit, integration, UI, and POMDP barrier tests
├── requirements.txt                # Pinned core & optional extension dependencies
├── .env.example                    # Template for optional LLM provider configuration
├── .gitignore                      # Git ignore rules for caches, venvs, and logs
└── LICENSE                         # MIT License
```

---

## 7. Limitations & Future Work

Per Roadmap Sec. 40, several research boundaries are explicitly identified:
1. **Simulated Learner Scope:** All evaluations are conducted within a synthetic parametric learner simulator. While sensitivity sweeps demonstrate robustness across parameter shifts, results represent preliminary simulated evidence rather than human-subject educational validation.
2. **Bayesian Knowledge Tracing (Level 2 Extension):** The core benchmark uses an empirical exponential moving average ($\eta=0.20, \delta=0.01$). A BKT belief tracker is implemented in `src/learner_model/bkt_tracker.py` and evaluated in the DQN benchmark, providing calibrated probabilistic mastery estimates.
3. **Deep Q-Network Extension:** Tabular Q-learning operates on discretized 320-state representations. A continuous 14-D cognitive state DQN extension is implemented in `src/rl/dqn_agent.py` and benchmarked in `src/evaluation/run_dqn_benchmark.py`.
4. **Counterfactual Off-Policy Evaluation (Level 5):** Future work will evaluate doubly robust off-policy estimators on logged trajectory data.
5. **Real-User Pilot (Level 6):** Controlled human-subject trials under appropriate institutional review.

---

## 8. Scientific Terminology Standards

Per Roadmap Sec. 39, AirTutor enforces strict terminology standards to avoid academic overclaiming:
- **"Simulated Learner"** is used consistently over "student" or "human learner" to prevent any implication of human-subject empirical validation.
- **"Transparent Policy Inspection"** is used instead of "explainable AI", reflecting that tabular Q-learning decisions are inspectable table lookups rather than opaque inferences.
- **"Preliminary Simulated Evidence"** is used rather than "proof" or "guarantee", reflecting that results are bounded by the parametric simulation environment.

---

## 9. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
