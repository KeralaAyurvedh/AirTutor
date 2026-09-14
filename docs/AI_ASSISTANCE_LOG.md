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

---

## Stage 2: Observable State Encoder, Baseline Teachers, & Core Metrics
- **Date:** 2026-09-14
- **Prompt:** Prompt 3 of 7
- **AI Contributions:**
  - Implemented modular observable state encoder in `src/environment/state_encoder.py` mapping interaction histories to discrete feature tuples:
    mastery bucket (5), last correct (2), recent accuracy (4), wrong streak (4), hint used (2). Exactly 320 discrete states ($5 \times 2 \times 4 \times 4 \times 2$), with bidirectional index conversion and feature toggling for ablation studies.
  - Implemented `BaseTeacher` and `TeacherObservation` in `src/teachers/base_teacher.py`.
  - Implemented `RandomTeacher` (uniform exploration) in `src/teachers/random_teacher.py`.
  - Implemented `FixedTeacher` (alternating EXPLAIN $\to$ PRACTICE curriculum cycles) in `src/teachers/fixed_teacher.py`.
  - Implemented competitive `RuleTeacher` (scaffolding via EXPLAIN, HINT on struggle, EXAMPLE, PRACTICE, and periodic prerequisite review) in `src/teachers/rule_teacher.py`.
  - Implemented evaluation metrics suite in `src/evaluation/metrics.py` (learning gain, final mastery, steps to mastery, intervention efficiency, hint usage rate, action entropy).
  - Implemented 16 new unit tests (32 tests total) in `tests/test_state_encoder.py`, `tests/test_teachers.py`, and `tests/test_metrics.py`.
  - Authored 50-episode baseline comparison experiment script in `src/evaluation/baseline_comparison.py`.
- **Verification & Review:**
  - `python -m pytest` passed all 32 tests in 1.03s.
  - Executed `python src/evaluation/baseline_comparison.py` running 50 paired episodes per teacher on identical cohorts of simulated learners.
  - Verified that Rule teacher visibly outperforms Random and Fixed teachers on learning gain:
    - Observable Learning Gain: Rule (0.031 $\pm$ 0.086) > Random (0.023 $\pm$ 0.067) > Fixed (0.013 $\pm$ 0.061).
    - True Learning Gain: Rule (0.214 $\pm$ 0.177) > Fixed (0.193 $\pm$ 0.223) > Random (0.158 $\pm$ 0.172).
    - Action Entropy: Random (2.225 bits, near max), Fixed (1.000 bits), Rule (1.384 bits).
- **Deviations / Notes:**
  - In `src/simulator/dynamics.py`, introduced `FORGETTING_SCALE = 0.08` to convert $\beta \in [0.01, 0.15]$ to per-step decay ($0.001 - 0.012$), matching observable estimator $\delta = 0.01$ and preventing unnatural curriculum-wide knowledge erasure over multi-concept episodes.

---

## Stage 3: Tabular Q-Learning Agent & Held-Out Benchmark
- **Date:** 2026-09-14
- **Prompt:** Prompt 4 of 7
- **AI Contributions:**
  - Implemented Reward v1 in `src/environment/rewards.py`: $r_t = \hat{M}_{c, t+1} - \hat{M}_{c, t}$ based strictly on observable estimated mastery.
  - Implemented `TabularQAgent` in `src/rl/q_learning.py` conditioning strictly on the 320 discrete observable states from Stage 2. Hyperparameters declared near top: `learning_rate = 0.15`, `discount_factor = 0.90`, `epsilon = 1.0 -> 0.05` (`decay = 0.992`). Includes serialization and persistence.
  - Implemented multi-seed training pipeline in `src/rl/train.py` training over 5 independent seeds on training population of 300 simulated learners (seed range $[0, 300)$).
  - Implemented publication plotting utilities in `src/evaluation/plotting.py` generating:
    - Training reward curve across RL seeds with $\pm 1 \sigma$ error bands.
    - Held-out learning gain by teacher with standard deviation error bars.
  - Implemented held-out statistical benchmark in `src/evaluation/compare_teachers.py` evaluating Random, Fixed, Rule, and RL on 100 held-out unseen learners (seed range $[10000, 10100)$, strictly disjoint from training).
  - Implemented automated experiment logging saving config snapshot, seeds, git commit hash, raw episode results, summary statistics with 95% CI, and plots under `experiments/<run_id>/`.
  - Added 6 unit tests in `tests/test_rewards.py` and `tests/test_q_learning.py` (total 38 tests passing).
- **Verification & Review:**
  - `python -m pytest` passed all 38 tests in 1.71s.
  - Executed `python src/evaluation/compare_teachers.py` end-to-end.
  - Verified held-out evaluation on disjoint population:
    - Learning Gain (Observed): **RL (0.131 $\pm$ 0.076 [$\pm$0.007])** > Rule (0.028 $\pm$ 0.071 [$\pm$0.014]) > Fixed (0.018 $\pm$ 0.065) > Random (0.012 $\pm$ 0.061).
    - True Learning Gain: **RL (0.299 $\pm$ 0.191 [$\pm$0.017])** > Rule (0.234 $\pm$ 0.157) > Fixed (0.215 $\pm$ 0.210) > Random (0.166 $\pm$ 0.151).
    - Intervention Efficiency: **RL (0.0044)** > Rule (0.0009) > Fixed (0.0006) > Random (0.0004).
    - Hint Usage Rate: RL (0.005) avoided over-hinting under Reward v1.
    - Policy Diversity: RL maintained healthy action entropy ($1.374$ bits) without collapsing into a single action.
- **Deviations / Notes:**
  - None. All evaluation protocol, seed disjointness, and standing rules were strictly enforced.
