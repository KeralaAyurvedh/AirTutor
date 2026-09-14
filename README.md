# AirTutor — RL-Based Adaptive Pedagogical Decision System

AirTutor is a research artifact investigating reinforcement-learning-driven pedagogical intervention selection for simulated learners.

- **Research Core:** An RL policy selects discrete pedagogical interventions based on an observable learner state.
- **Delivery Layer:** An LLM renders the selected intervention into natural language (provider-agnostic, replaceable, never the decision maker).
- **Evaluation:** Evaluated against Random, Fixed, and Rule-based teacher baselines on held-out simulated learner populations across multiple seeds.

The authoritative project specification is documented in [`docs/research_roadmap_v2.md`](docs/research_roadmap_v2.md).

---

## Environment Setup (Windows-Friendly)

This project targets Python 3.11+, CPU-only, and 8 GB RAM. Follow the steps below using PowerShell or Command Prompt on Windows.

### 1. Prerequisites
Ensure Python 3.11 or higher is installed and added to your `PATH`:
```powershell
python --version
```

### 2. Create Virtual Environment
From the repository root directory (`AirTutor/`):
```powershell
python -m venv .venv
```

### 3. Activate Virtual Environment

**In PowerShell:**
```powershell
.\.venv\Scripts\Activate.ps1
```
*(Note: If you encounter an execution policy restriction, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in your PowerShell session first.)*

**In Command Prompt (cmd.exe):**
```cmd
.\.venv\Scripts\activate.bat
```

**In Git Bash:**
```bash
source .venv/Scripts/activate
```

### 4. Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Stage 0: Mechanical RL Proof

To verify that the tabular Q-learning loop operates cleanly on a hand-rolled environment:

```powershell
python src/rl/toy_loop_demo.py
```

This runs 300 episodes of tabular Q-learning on a toy discrete MDP, logs reward statistics, and exports the learning curve plot to:
```
experiments/toy_learning_curve.png
```

### Run Tests
To run unit and integration tests via pytest:
```powershell
pytest
```

---

## Repository Structure

```
airtutor/
  data/curriculum/       one JSON file per module
  configs/               hyperparameters, learner-population distributions
  src/
    environment/         state encoder, action space, transition/reward orchestration
    learner_model/       observable mastery estimator
    simulator/           parametric population, dynamics, sanity checks
    rl/                  tabular Q-learning agent, training loop
    teachers/            random, fixed, rule, oracle baselines
    evaluation/          metrics, comparison runner, stats, plots
    llm/
      provider.py        abstract interface + get_llm_provider()
      providers/         gemini.py, openai_compatible.py, offline.py
      validation.py      schema validation for generated content
  experiments/           per-run config snapshots, seeds, results, plots
  app/                   Streamlit UI (imports core, no logic)
  tests/                 unit / integration / experiment / UI tests
  docs/                  research_roadmap_v2.md, methodology.md, AI_ASSISTANCE_LOG.md
  README.md
  requirements.txt
  .env.example
  .gitignore
```
