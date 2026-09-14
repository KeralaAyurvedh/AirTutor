# AirTutor — Project Identity (Research-Grade Build)

## What this project is
AirTutor is a research artifact: an RL agent selects a pedagogical
intervention for a simulated learner based on an observable state; an
LLM renders that intervention as natural-language content. The RL
policy, the learner simulation, and the evaluation methodology are the
research contribution. The LLM is a replaceable delivery layer, never
the decision-maker.

## Full specification
The complete, authoritative spec is docs/research_roadmap_v2.md (43
sections). This file holds only the constraints that must apply on
every single prompt. Before implementing each stage, read the specific
roadmap section(s) cited in that stage's prompt in full — don't work
from a one-line summary of it.

## Non-negotiable constraints
- Python 3.11+, CPU-only, 8GB RAM target. No PyTorch, no Gymnasium, no
  neural nets anywhere in this build. Tabular Q-learning only.
- The RL policy's state input must NEVER contain hidden learner
  parameters (the theta vector: learning rate, forgetting rate,
  difficulty sensitivity, hint dependence, transfer ability, noise) or
  true simulator-internal mastery. Only the observable feature vector
  (roadmap Sec. 13) is ever passed to the policy. This applies even
  during debugging — don't print theta anywhere the policy's code path
  could see it.
- The LLM must be fully provider-agnostic. RL/environment/simulator
  code never imports a specific LLM provider, never contains
  provider-specific logic, and never hard-codes a model name. All
  provider/model selection is config-driven via environment variables
  (roadmap Sec. 24-26). If you catch yourself writing `import
  google.genai` anywhere outside src/llm/providers/, stop.
- The Research Core (src/environment, src/learner_model,
  src/simulator, src/rl, src/teachers, src/evaluation) must have zero
  Streamlit import and zero LLM import, anywhere, ever. It must be
  fully runnable headlessly from a plain script.
- If a trained RL policy does not beat the Rule baseline, that result
  is kept and reported as-is. Never modify the simulator, reward,
  evaluation protocol, or baseline strength after the fact to
  manufacture a win. A negative result with sound methodology is a
  valid, reportable outcome — this is a standing rule for the whole
  project, not a one-time checkbox.
- Curriculum content lives in JSON under data/curriculum/, never
  hard-coded in Python.
- Never hard-code an API key. Read from environment variables via
  python-dotenv. Every stage before Prompt 6 must run with zero API
  keys configured at all.

## Working style
- Read the cited roadmap section(s) before implementing each stage.
- Plan before coding: a short Task List / Implementation Plan for
  anything touching more than one file.
- Run what you write — execute it, run pytest, don't claim a stage
  done without having actually verified it — and fix what breaks
  yourself.
- Operate autonomously on implementation details. Only pause and ask
  me for a genuine product/research-design ambiguity, not a technical
  bug. State your assumption in the Walkthrough and keep going
  otherwise.
- Maintain a running AI_ASSISTANCE_LOG.md, updated every stage.
- Commit to main after each stage with a clear message. No branches.

## Repo layout
airtutor/
  data/curriculum/       one JSON file per module
  configs/                hyperparameters, learner-population
                           distributions, per-experiment configs —
                           never hard-coded in Python
  src/
    environment/          state encoder, action space, transition/
                           reward orchestration
    learner_model/         observable mastery estimator
    simulator/              parametric population, dynamics, and the
                            sanity-check tests live alongside it
    rl/                     tabular Q-learning agent, training loop
    teachers/               random, fixed, rule, (optional) oracle
    evaluation/             metrics, comparison runner, stats, plots
    llm/
      provider.py            abstract interface + get_llm_provider()
      providers/              gemini.py, openai_compatible.py,
                               offline.py
      validation.py           schema validation for generated content
  experiments/            per-run: config snapshot, seeds, results,
                           plots
  app/                    Streamlit UI — imports core, no logic
  tests/                  unit / integration / experiment / UI
  docs/
    research_roadmap_v2.md    the full spec (already placed)
    methodology.md              cross-references the roadmap
    AI_ASSISTANCE_LOG.md
  README.md
  requirements.txt
  .env.example
  .gitignore

## Definition of done, this build
Defined stage by stage in the prompts that follow. Don't build ahead
of the current stage.
