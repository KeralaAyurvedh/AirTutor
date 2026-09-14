# AI Assistance Log

This log documents AI-assisted development across each stage of the AirTutor research project, in compliance with the guidelines in `GEMINI.md` and Section 33 of `docs/research_roadmap_v2.md`.

---

## Stage 0: Repository Scaffold & Mechanical RL Loop
- **Date:** 2026-09-14
- **Prompt:** Prompt 1 of 7
- **AI Contributions:**
  - Initialized git repository and set up folder hierarchy according to Section 30 of the research roadmap.
  - Authored `requirements.txt`, `.gitignore`, `.env.example`, and Windows virtual environment setup in `README.md`.
  - Implemented lightweight `ToyChainEnv` hand-rolled MDP and `TabularQLearningAgent` in `src/rl/toy_loop_demo.py`.
  - Implemented unit test in `tests/test_toy_loop.py` to verify Bellman Q-table update mechanics.
- **Verification & Review:**
  - Verified package environment and Python 3.13 CPU-only runtime.
  - Executed `pytest tests/` to confirm mathematical correctness of Q-table updates.
  - Executed `python src/rl/toy_loop_demo.py` and inspected resulting training curve plot (`experiments/toy_learning_curve.png`).
- **Deviations / Notes:**
  - None. Stage 0 keeps strict separation: zero simulator/curriculum/LLM imports.

---

## Stage 1: Curriculum, Learner Model, & Parametric Simulator
- **Date:** 2026-09-14
- **Prompt:** Prompt 2 of 7
- **AI Contributions:**
  - Created curriculum JSON schema and seeded `data/curriculum/gate_programming_fundamentals.json` with 5 concepts and 30 questions across types (`mcq`, `short`, `code`).
  - Implemented generic dynamic curriculum loader and prerequisite validator in `src/environment/curriculum.py`.
  - Defined discrete pedagogical action space enum in `src/environment/actions.py` (`EXPLAIN`, `EXAMPLE`, `HINT`, `PRACTICE`, `REVIEW_PREREQUISITE`).
  - Created observable learner model in `src/learner_model/mastery.py` updating $\hat{M}(c) \in [0, 1]$ strictly based on observable response events with time-decay.
  - Implemented parametric simulated learner population in `src/simulator/population.py` sampling $\theta = [\alpha, \beta, \gamma, h, \tau, \sigma]$ independently over customizable seed ranges, configured via `configs/learner_population.json`.
  - Implemented smooth simulator dynamics in `src/simulator/dynamics.py` enforcing non-circular action effects without learner-type lookup tables, including logistic response correctness probability $P(\text{correct})$ and prerequisite transfer $\tau$.
  - Implemented 16 unit tests:
    - 3 curriculum loading and graph dependency tests in `tests/test_curriculum.py`.
    - 5 observable estimator tests in `tests/test_learner_model.py`.
    - 5 automated simulator sanity checks in `tests/test_simulator_sanity.py` (monotonic in mastery, difficulty, $\alpha$, $\beta$, and $\sigma$).
    - 3 baseline tests in `tests/test_toy_loop.py`.
  - Authored trajectory demonstration script `src/simulator/demo_trajectories.py` evaluating 20 simulated learners side-by-side on true vs. estimated mastery.
- **Verification & Review:**
  - Ran `python -m pytest` with 16/16 tests passing cleanly in <1s.
  - Ran `python src/simulator/demo_trajectories.py` verifying true vs. estimated trajectories track with high correlation ($r \approx 0.82$) while preserving realistic estimation error ($\text{MAE} \approx 0.11$).
  - Verified architectural barrier: $\theta$ and true latent mastery $M_t$ remain strictly simulator-internal and are never accessed by the learner model.
- **Deviations / Notes:**
  - Parameter distributions externalized to `configs/learner_population.json` per GEMINI.md requirements.
