---
title: "AirTutor: An RL-Based Adaptive Pedagogical Decision System"
subtitle: "Research & Development Roadmap, v2.0 — supersedes the v1 MVP roadmap (29 Aug 2026)"
date: "September 2026"
toc: true
toc-depth: 2
---

**Document status:** Everything in this document is *proposed design*. Nothing described here has been implemented or experimentally validated yet — that work begins at Stage 0. Wherever a claim could be misread as already-demonstrated, it is explicitly tagged `[proposed]`, `[implemented]`, `[experimentally validated]`, or `[future work]`. As of this version, essentially everything is `[proposed]`.

# Executive Summary

AirTutor is an adaptive tutoring system whose research contribution is a **reinforcement-learning (RL) policy that selects pedagogical interventions** for a learner based on their current state. A large language model (LLM) renders whatever intervention the policy selects into natural-language content. The LLM is the delivery mechanism; it is not, and must never become, the decision-maker.

This document replaces the original single-PDF roadmap (29 Aug 2026). That version optimized for a fast, disposable MVP. This version optimizes for **scientific validity, reproducibility, lightweight implementation, extensibility, and a clear research contribution** — buildable incrementally over several months on modest hardware (Windows 11, 8 GB RAM, CPU-only), without premature software engineering.

The guiding principle, unchanged from the original: **build evidence, not features.** Every component in this document is justified against one of: *what research question does this test*, *what validity problem does this solve*, *what does this make measurable, reproducible, or extensible*. Anything that fails all four is marked future work.

# Part I — Research Foundation

## 1. What Changed From the Original Roadmap

| Area | v1 (original PDF) | v2 (this document) |
|---|---|---|
| Timeline framing | 1–2 day MVP, Oct/Nov calendar dates | Staged by dependency, not calendar-locked |
| Simulated learners | 3 fixed named profiles | Parametric population sampled from justified distributions ([Sec. 9](#parametric-learner-population)) |
| Learner state | Implicitly whatever the profile "was" | Formal hidden/observable split — the agent never sees the true learner parameters ([Sec. 10](#hidden-vs-observable-learner-state)) |
| RL formalism | Described informally | Formal state/action/reward/policy definition, explicit about the POMDP-like structure ([Sec. 7](#mathematical-rl-formulation)) |
| Simulator | Assumed valid by construction | Dedicated validity/circularity analysis with empirical sanity checks ([Sec. 12](#simulator-validity-assumptions-and-threats-to-validity)) |
| Baselines | Fixed + Rule | Random, Fixed, Rule (strengthened), RL, optional Oracle ([Sec. 16](#baseline-teachers)) |
| Train/eval | Implicit | Explicit held-out learner configurations, multiple seeds, disjoint seed ranges ([Sec. 17](#training--evaluation-protocol)) |
| Robustness | Not addressed | Dedicated sensitivity-analysis and ablation sections ([Secs. 21–22](#robustness-and-sensitivity-analysis)) |
| LLM coupling | "Keep API usage optional" | Formal provider-agnostic abstraction; RL core has zero knowledge of which LLM is used ([Secs. 24–26](#llm-abstraction-architecture)) |
| Self-critique | Drawbacks section | Dedicated "Remaining Risks and Design Trade-offs" chapter, required before any claim is made ([Sec. 37](#remaining-risks-and-design-trade-offs)) |

What is **retained** from v1, deliberately: RL chooses the pedagogical action and the LLM renders it; a compact starting curriculum; a learner model; a simulated learner; tabular Q-learning as the first algorithm; reward based initially on mastery improvement; fixed/rule baselines; staged development; honest limitations; a future path toward knowledge tracing and unseen-learner evaluation. v1 was not wrong — it was under-specified for the level of rigor this version targets.

## 2. Research Problem

Most adaptive tutoring systems focus on *what content to show* or *predicting learner performance*. This project investigates a different, narrower decision layer: **given a representation of a learner's current state, which pedagogical intervention (explain, exemplify, hint, drill, or review a prerequisite) should be selected, and can that selection be learned rather than hand-authored?**

## 3. Research Questions

- **RQ1 (primary):** Can an RL-based tutor learn a pedagogical policy that improves a simulated learner's mastery more efficiently than a fixed or a strong rule-based teaching policy?
- **RQ2:** Does conditioning the policy on observable learner-behavior features (vs. a non-adaptive baseline) yield a measurable advantage, and is that advantage robust to reasonable changes in simulator assumptions?
- **RQ3:** Does optimizing for mastery gain alone (reward v1) produce degenerate ("reward-hacked") policies, and do richer reward formulations (v2/v3) prevent this?
- **RQ4:** Which state features are actually load-bearing for policy quality (via ablation), and which pedagogical actions are consistently preferred for which learner states?
- **RQ5:** Does a policy trained on one population of simulated learners transfer to a disjoint, held-out population sampled from the same parametric distributions?

## 4. Hypotheses

- **H1:** A learner-state-aware RL policy will achieve higher mean learning gain and lower mean time-to-mastery than the Fixed and Rule baselines, evaluated on held-out simulated learners, across multiple training seeds.
- **H2:** Reward v1 (mastery gain only) is more susceptible to degenerate behavior (e.g., over-selecting the action that maximizes short-term correctness without durable gain) than reward v2/v3.
- **H3:** The RL advantage over the Rule baseline, if it exists, is not an artifact of one narrow simulator configuration — it should survive a moderate sensitivity sweep over learner-population parameters.

These are hypotheses to test, not conclusions. **If H1 fails — if RL does not beat the Rule baseline — that result is retained and reported, not engineered away.** Modifying the simulator, reward, evaluation protocol, or baseline strength after the fact specifically to manufacture an RL win is not permitted under any circumstance; see the standing rule in [Sec. 15](#reward-design).

## 5. Research Contribution

The contribution is the **learner-state representation, the pedagogical decision policy, and the reward/evaluation framework** — not the LLM. Concretely, the strongest possible demonstration of this project is not *"look, a chatbot that teaches Python."* It is: *"the learner's behavioral state changed, therefore the policy's action changed, and only then did an LLM render that decision into words."* If a visitor cannot tell the difference between AirTutor and a well-prompted chatbot without being shown the policy's internal state, the research identity has been lost, and the LLM has quietly become the product. Guarding against this drift is a standing design constraint, not a one-time check.

# Part II — System Design

## 6. System Architecture

```
Learner (simulated)
   |
   v
Learner Model  ---->  Observable State O_t
   |                        |
   | (hidden: theta, M_t)   v
   |                   RL Policy pi(a | O_t)
   |                        |
   |                        v
   |                Pedagogical Action a_t
   |                        |
   |         +--------------+--------------+
   |         |                             |
   |         v                             v
   |   LLM Abstraction Layer      (offline template
   |         |                     fallback path)
   |         v
   |   Rendered tutoring content (text shown to learner)
   |         |
   +---------+  learner responds
             |
             v
   State Update (learner model + simulator dynamics)
             |
             v
   Reward r_t  --------->  Policy Update
```

Three layers, deliberately separated so each can be built, tested, and reasoned about independently ([Sec. 29](#research-core-vs-delivery-vs-demo-layer)):

- **Research Core** — learner simulator, learner model, state encoder, environment, RL policy, baselines, reward, evaluation. Runs headlessly, with zero UI or LLM dependency.
- **Delivery Layer** — the LLM provider abstraction. Converts a pedagogical action into natural-language content. Swappable without touching the Research Core.
- **Demo Layer** — the Streamlit UI. Contains no decision logic; it only calls into the two layers above.

## 7. Mathematical RL Formulation

This problem is, strictly, a **Partially Observable MDP (POMDP)**: the true learner parameters and true mastery are hidden, and the agent acts on an engineered observation. We do not implement full POMDP belief-state tracking (that would be over-engineering for a tabular first system); instead we treat the observable feature vector as a Markov state for the purposes of tabular Q-learning — a standard, deliberate simplification. This simplification is itself listed as a threat to validity in [Sec. 12](#simulator-validity-assumptions-and-threats-to-validity) and as a limitation in [Sec. 37](#remaining-risks-and-design-trade-offs).

**Hidden (simulator-only) state:**

- $\theta \in \Theta$ — the latent learner parameter vector ([Sec. 9](#parametric-learner-population)). Never exposed to the policy.
- $M_t \in [0,1]^K$ — true latent mastery over $K$ curriculum concepts at time $t$.

**Observable state (what the policy conditions on):**

- $O_t = \phi(h_t)$ — a fixed-length feature vector computed from interaction history $h_t$ (estimated mastery, recent correctness, streaks, hint usage, etc.). Full definition in [Sec. 13](#observable-state-representation).

**Action space:** $A = \{\text{EXPLAIN}, \text{EXAMPLE}, \text{HINT}, \text{PRACTICE}, \text{REVIEW\_PREREQUISITE}\}$ ([Sec. 14](#pedagogical-action-space)).

**Transition:** the simulator maps $(\theta, M_t, a_t, \text{question difficulty}) \to \text{correct}_t \in \{0,1\}$ (stochastic), updates $M_{t+1}$ via the dynamics in [Sec. 11](#simulator-dynamics), and the learner model separately produces its own estimate $\hat{M}_{t+1}$ feeding $O_{t+1} = \phi(h_{t+1})$.

**Reward:** $r_t = r(\cdot)$, versioned in [Sec. 15](#reward-design). Note a deliberate, documented asymmetry: because training happens in simulation, the reward function is permitted to use privileged ground-truth information ($M_t$, computed by the simulator) even though the **policy's input must never contain hidden information** — only $O_t$. This is standard practice in simulation-based RL (the critic/reward can see more than the actor) and is not a violation of the hidden-parameter rule, which governs the policy's observation, not the reward computation. We default to reward computed from *estimated* mastery $\hat{M}_t$ rather than ground truth $M_t$, because it keeps the whole pipeline consistent with what would be available in a real deployment; using ground-truth-oracle reward is documented as an explicit, optional experimental variant (useful for isolating how much estimation error in $\hat{M}_t$ hurts the learned policy), not the default.

**Policy:** $\pi(a_t \mid O_t)$, updated via tabular Q-learning ([Sec. 8 of the implementation plan — algorithm details in Stage 3](#phased-implementation-roadmap)).

**Canonical loop:** $s_t \to a_t \to \text{learner response} \to s_{t+1} \to r_t \to \text{policy update}$, where $s_t \equiv O_t$ throughout this document.

# Part III — Learner Simulation

This is the highest-risk part of the whole project. A weak simulator invalidates every downstream result, no matter how sound the RL algorithm is. This part is written to be read critically, including by the person building it.

## 8. Learner Model

`[proposed]` The learner model is the tutor's own **estimate** of mastery — deliberately distinct from the simulator's ground-truth $M_t$. It maintains $\hat{M}(c) \in [0,1]$ per concept $c$, updated after every observed response:

$$\hat{M}(c) \leftarrow \hat{M}(c) + \eta \cdot (\text{correct} - \hat{M}(c)) - \delta \cdot \Delta t_{\text{since practiced}}(c)$$

where $\eta$ (a fixed, small learning-rate constant of the *estimator*, not the learner) controls how quickly the estimate moves toward observed evidence, and $\delta$ is a small time-decay applied per step the concept goes unpracticed. This is intentionally simple — it is not the research contribution, and a more sophisticated estimator (Bayesian Knowledge Tracing or similar) is explicitly future work ([Sec. 40](#future-research-extensions)), not a Stage-0 requirement.

**Inputs:** concept id, correctness (bool). **Outputs:** updated $\hat{M}(c)$. **Constraints:** $\hat{M}(c) \in [0,1]$ always (clip). **Tests:** correct answers move $\hat{M}$ up; incorrect move it down; unpracticed concepts decay over time; the estimate never leaves $[0,1]$.

## 9. Parametric Learner Population

Replacing three fixed named profiles, each simulated learner is an independent draw of a latent parameter vector:

$$\theta = [\alpha, \beta, \gamma, h, \tau, \sigma]$$

| Parameter | Meaning | Proposed range | Justification |
|---|---|---|---|
| $\alpha$ learning rate | how much mastery increases per well-matched intervention | Uniform(0.05, 0.35) | Wide enough to produce meaningfully "fast" vs "slow" learners without either extreme dominating the population |
| $\beta$ forgetting rate | per-step mastery decay when a concept isn't practiced | Uniform(0.01, 0.15) | Modest per-step values so decay compounds realistically over many steps rather than erasing mastery in one step |
| $\gamma$ difficulty sensitivity | how strongly a mismatch between question difficulty and current mastery hurts correctness probability | Uniform(0.5, 2.0) | 1.0 as a neutral midpoint; range allows learners who are "brittle" under difficulty mismatch ($\gamma>1$) vs. more forgiving ($\gamma<1$) |
| $h$ hint dependence | how much more a HINT boosts *immediate* correctness relative to durable mastery gain | Uniform(0.0, 1.0) | 0 = hints barely help this learner; 1 = hints strongly boost the current answer but contribute little to retained mastery — this is the mechanism that makes reward-hacking via over-hinting *detectable* rather than assumed |
| $\tau$ transfer ability | fraction of a prerequisite's mastery gain that propagates to dependent concepts | Uniform(0.2, 1.0) | Always some transfer (never 0, which would make REVIEW_PREREQUISITE pointless by construction) but bounded below 1 (transfer is never perfect) |
| $\sigma$ response noise | stochastic noise added to correctness probability | Uniform(0.02, 0.10) | Small but non-trivial — enough to require multiple seeds/episodes per learner for stable estimates, without swamping the signal |

**Sampling protocol:** each simulated learner instance draws $\theta \sim \prod_i \text{Uniform}(\text{range}_i)$ independently at instantiation. **Training population:** $N_{\text{train}}$ learners sampled using seed range $[0, N_{\text{train}})$. **Evaluation population:** $N_{\text{eval}}$ learners sampled using a **disjoint** seed range (e.g. $[10000, 10000+N_{\text{eval}})$), never used during training — see [Sec. 17](#training--evaluation-protocol).

These ranges are starting points, not validated truths. They must be stated explicitly in every experiment's config file (never hard-coded), and any change to them is itself a robustness-relevant event to log ([Sec. 21](#robustness-and-sensitivity-analysis)).

## 10. Hidden vs. Observable Learner State

**This separation is non-negotiable and applies to the policy's input, not to the reward function** (see the asymmetry noted in [Sec. 7](#mathematical-rl-formulation)).

| Hidden (simulator only — the policy never sees this) | Observable (the policy's actual input) |
|---|---|
| $\theta = [\alpha,\beta,\gamma,h,\tau,\sigma]$ | Estimated mastery bucket $\hat{M}(c)$ for the current concept |
| True continuous mastery $M_t$ | Last response correct (bool) |
| | Accuracy over the last 3 responses |
| | Consecutive wrong-answer streak |
| | Whether a hint was used on the last step |
| | Recently taken action |
| | Estimated mastery of the weakest prerequisite |
| | Question difficulty |

Note the double approximation this creates, intentionally: the simulator's *true* generative process ($\theta$, $M_t$) is one layer removed from reality by construction (it's a simulation); the learner model's $\hat{M}(c)$ is a second, independent approximation of even the simulator's own ground truth. This mirrors what would actually be available with real students (you never observe true mastery directly) and is treated as a feature of the design, not swept under the rug.

## 11. Simulator Dynamics

`[proposed]` **The single most important design rule in this section: interventions must affect learning through a smooth, parametric mechanism — never through a lookup table keyed on a learner "type."** A rule of the shape *"if learner is type X and action is EXAMPLE, add +0.20 mastery"* is circular: it hard-codes the conclusion the RL agent is supposed to discover. The mechanism below is written so that *why* an action helps a given learner emerges from the interaction of continuous $\theta$ values with each action's pedagogical profile, not from a per-type special case.

Illustrative (tunable, must be documented and justified wherever the actual values are chosen) mastery update:

$$M_{c,t+1} = \text{clip}\Big(M_{c,t} + \alpha \cdot A_{\text{effect}}(a_t, M_{c,t}, \theta) \cdot \text{correct}_t \;-\; \beta \cdot \Delta t_{\text{unpracticed}}(c),\; 0,\; 1\Big)$$

where $A_{\text{effect}}$ is a smooth, action-dependent multiplier — for example:

- **EXPLAIN:** roughly constant conceptual-clarity contribution, largely independent of $\theta$.
- **EXAMPLE:** contribution scaled by $(1-\alpha)$ — slower learners benefit *relatively* more from concrete illustration, an assumption stated here explicitly rather than hidden in a hard-coded bonus.
- **HINT:** strongly boosts $\text{correct}_t$ itself (scaled by $h$), but contributes comparatively little to the durable $M_{c,t+1}$ term — this is the parametric mechanism that makes reward hacking via over-hinting an *emergent, measurable* phenomenon rather than an assumption.
- **PRACTICE:** contribution peaks at *moderate* current mastery and is small at very low or very high mastery — reflecting the "desirable difficulty" idea that drilling a concept you can't yet attempt, or one you've already mastered, teaches little.
- **REVIEW_PREREQUISITE:** raises the mastery of a prerequisite concept $p$; a fraction $\tau$ of that gain propagates to dependent concept $c$.

Question-level correctness probability should further depend on $\gamma$ (difficulty sensitivity) applied to the gap between question difficulty and $M_{c,t}$, and on $\sigma$ (noise). All named constants in $A_{\text{effect}}$ must be defined near the top of the implementation, commented, and treated as tunable — not verified — until the sanity checks below and the sensitivity analysis in [Sec. 21](#robustness-and-sensitivity-analysis) have been run.

## 12. Simulator Validity, Assumptions, and Threats to Validity

**What is modeled:** a single learner's mastery trajectory across concepts, as a function of latent traits and the pedagogical actions taken, with noise.

**What is simplified:** no modeling of attention, fatigue, motivation, real-world interruptions, or multi-session forgetting curves beyond a single decay constant; no modeling of question-wording effects; transfer is modeled as a single scalar rather than concept-pair-specific.

**Key assumptions:** learner behavior is well-approximated by six scalar traits; the effect of an action on mastery is separable into the smooth mechanism above (rather than arbitrarily complex ability-by-action interactions); noise is i.i.d. per response.

**Where circularity could occur:** if the *rule-based baseline's* heuristic happens to align suspiciously exactly with the mechanism above (e.g., "give HINT when a streak of wrong answers is detected" — which is also mechanically favored by the simulator), any RL "advantage" could be an artifact of exploiting the simulator's specific mechanism rather than genuine adaptive pedagogy. **Detection:** if the RL policy's learned action distribution matches the rule-baseline's decision boundaries almost exactly, treat this as a signal to investigate circularity, not as confirmation of a well-designed rule baseline. **Reduction:** keep the simulator mechanism's constants independent of and unknown to whoever designs the rule-baseline's thresholds (document the rule baseline design *before* finalizing simulator constants, or vice versa, and note which order was used).

**Sensitivity/robustness analysis:** required before any RL-beats-baseline claim is treated as more than preliminary — see [Sec. 21](#robustness-and-sensitivity-analysis).

**Why simulated results are not proof of human educational effectiveness:** a simulated learner is a stated set of assumptions, not a person. Results support claims of the form *"a learned policy improves outcomes for simulated learners with these dynamics"* — not *"this tutor works for real students."* Bridging that gap is explicitly out of scope until a real-user pilot (Level 6, [Sec. 40](#future-research-extensions)), which requires informed consent and appropriate review ([Sec. 38](#research-ethics--limitations)).

**Required sanity checks** (implemented as automated tests, not eyeballed once):

| # | Property | Check |
|---|---|---|
| 1 | Monotonic in mastery | Higher $M_{c,t}$ → higher $P(\text{correct})$, holding difficulty fixed |
| 2 | Monotonic in difficulty | Higher question difficulty → lower $P(\text{correct})$, holding mastery fixed |
| 3 | Monotonic in $\alpha$ | Higher learning rate → faster mastery growth under a fixed action sequence |
| 4 | Monotonic in $\beta$ | Higher forgetting rate → faster mastery decay when unpracticed |
| 5 | Monotonic in $\sigma$ | Higher noise → higher variance in observed correctness at fixed mastery |

These five checks are unit tests against the simulator itself (see [Sec. 31](#testing-strategy)) and must pass before any RL experiment is trusted.

# Part IV — RL Problem Design

## 13. Observable State Representation

`[proposed]` State-space explosion is a real risk: naively tracking a mastery bucket for every one of $K$ curriculum concepts jointly would produce a state space of roughly $5^K \times (\text{other factors})$, which becomes intractable for tabular Q-learning once $K$ exceeds 3–4. The design below avoids this by conditioning on the *current* concept's own bucket plus a small number of cross-cutting behavioral features, rather than the full per-concept mastery vector.

**Proposed state (v1), with bucket counts:**

| Feature | Buckets | Rationale |
|---|---|---|
| Current concept's mastery bucket | 5 (0.0–0.2 … 0.8–1.0) | Core driver of difficulty-appropriate action choice |
| Last response correct | 2 | Immediate feedback signal |
| Accuracy over last 3 responses | 4 (0/1/2/3 correct) | Short-term trend, smooths single-response noise |
| Consecutive wrong-answer streak | 4 (0, 1, 2, 3+) | Detects sustained struggle |
| Hint used on last step | 2 | Detects hint-dependence in the moment |

**Estimated state count:** $5 \times 2 \times 4 \times 4 \times 2 = 320$ states $\times$ 5 actions $=$ 1,600 Q-values — trains in seconds on CPU, comfortably within the 8 GB RAM constraint.

**Optional v1.1 extension** (add only if v1 trains cleanly and there's a concrete reason): weakest-prerequisite mastery bucket (5 buckets) — relevant specifically to when REVIEW_PREREQUISITE is valuable. This multiplies the state count by 5 (→ 1,600 states, 8,000 Q-values) — still tractable, but the trade-off should be a deliberate choice, not a default.

The state encoder must be **modular** — implemented as a single function mapping raw history to the feature tuple — so that ablated variants (v1 minus hint-usage, v1 minus recent-accuracy, v1 minus prerequisite feature, etc.) can be produced by toggling which features are included, without touching the RL algorithm. This modularity is what makes [Sec. 22](#ablation-studies) possible.

## 14. Pedagogical Action Space

$A = \{\text{EXPLAIN}, \text{EXAMPLE}, \text{HINT}, \text{PRACTICE}, \text{REVIEW\_PREREQUISITE}\}$ — five actions, discrete, interpretable. `[future work, explicitly deferred]`: EASIER / HARDER difficulty-control actions, a Socratic/guided-question action, a TEST action, a transfer/application-task action. The action interface (a single enum plus a dispatch table mapping action → simulator effect → LLM prompt template) is designed so a new action can be added in those three places without modifying the RL algorithm itself.

## 15. Reward Design

Reward is treated as a **research variable**, not a solved problem.

- **v1 (default, Stage 3):** $r_t = \hat{M}_{c,t+1} - \hat{M}_{c,t}$ — mastery gain on the concept just taught, using the *estimated* mastery (see the reward-source note in [Sec. 7](#mathematical-rl-formulation)).
- **v2 (Stage 4):** $r_t = (\hat{M}_{c,t+1} - \hat{M}_{c,t}) + \lambda_{\text{ret}} \cdot \text{retention\_signal} - \lambda_{\text{cost}} \cdot \text{intervention\_cost}$ — a small penalty discourages stalling on excessive hints or trivially easy repetition.
- **v3 (Stage 4, if v2 is stable):** adds a transfer term (mastery gain on *other*, dependent concepts attributable to $\tau$) and further penalizes wasted steps (e.g. re-teaching an already-mastered concept).

**Known limitations of v1, stated up front rather than discovered later:** short-term optimization; possible reward hacking (see [Sec. 35](#failure-modes)); no direct retention or transfer measure; may favor interventions that are easy to show gains on rather than genuinely durable ones.

**Standing rule — negative results must be accepted.** If the RL policy does not outperform the Rule baseline under the protocol in [Sec. 17](#training--evaluation-protocol), that result is retained and reported as-is. The simulator, reward function, evaluation protocol, or baseline strength must never be modified after the fact specifically to manufacture an RL win. A negative result produced with a sound methodology is a legitimate, reportable research finding — an inflated positive result is not. This applies for the life of the project, not just at the first checkpoint.

**Explicit failure case to watch for:** if the agent achieves high measured reward (or near-perfect immediate accuracy) while independent, held-out post-test-style performance does not improve, the system is not succeeding, regardless of the reward-curve plot looking good.

## 16. Baseline Teachers

| Baseline | Decision rule | Purpose |
|---|---|---|
| Random | Uniform random choice among the 5 actions | Sanity-check lower bound; anything should beat this |
| Fixed | Curriculum order, always EXPLAIN → PRACTICE → next concept, ignores performance | Represents "no personalization at all" |
| Rule | If mastery < 0.4: EXPLAIN; elif last wrong: HINT; elif mastery < 0.7: EXAMPLE; else: PRACTICE; periodically REVIEW the weakest prerequisite | A **genuinely competitive**, transparent hand-designed heuristic — not a strawman. Designing this to actually be strong is important: the real research question is whether a *learned* policy beats a *good* heuristic, not merely "beats something." |
| RL | Tabular Q-learning policy $\pi(a\mid O_t)$ | The system under study |
| Oracle *(optional, stretch)* | Privileged access to $\theta$ and true $M_t$ directly | A theoretical upper bound showing how much headroom exists between learning-from-observables and an all-knowing teacher — not a fair comparison, but informative context |

# Part V — Experimental Methodology

## 17. Training / Evaluation Protocol

- **Training population:** $N_{\text{train}}$ (proposed: 200–500) simulated learners drawn using seed range $[0, N_{\text{train}})$.
- **Evaluation population:** $N_{\text{eval}}$ (proposed: 100) simulated learners drawn using a **disjoint** seed range (e.g. $[10000,\,10000+N_{\text{eval}})$), never touched during training.
- **RL training seeds:** the Q-learning training process itself is repeated across 5–10 independent random seeds (affecting exploration and initialization), so that final performance is reported as a distribution, not a single number.
- All seeds, the exact ranges used, the simulator-version identifier, and the config file are recorded per experiment ([Sec. 23](#experiment-logging--structure)).

## 18. Unseen-Learner Evaluation

"Unseen learner" in this project means specifically: **a learner instance whose $\theta$ was sampled from the same parametric distributions used in training, but using a disjoint random seed never presented during training.** This is a precise, defensible claim. It is explicitly *not* the same as, and must never be described as, generalization to a *different* population (e.g., real students, or a shifted distribution) — that is a strictly harder and separate claim, listed as a future robustness extension (train on one distribution, evaluate on a deliberately shifted one).

## 19. Metrics

**Primary:** learning gain (mean $\hat{M}$ increase over an episode), final mastery, time/steps-to-mastery.

**Secondary:** intervention efficiency (gain per step), hint-usage rate, action-distribution entropy (a proxy for policy diversity vs. degenerate collapse onto one action), variability across seeds.

**Contingent on simulator extension (future work):** retention (mastery measured after a simulated delay with no practice), independent post-test performance (evaluated on questions never used during the interaction), transfer performance on a held-out dependent concept.

## 20. Statistical Evaluation

Every headline comparison is reported as **mean ± standard deviation across training seeds**, with a 95% confidence interval where the seed count reasonably supports one. Given compute constraints, expect 5–10 seeds — be explicit that this yields wide confidence intervals, and say so plainly rather than implying more precision than the sample size supports. A single successful run is a demonstration, not evidence; it is never reported as if it were the latter.

## 21. Robustness and Sensitivity Analysis

**Purpose:** determine whether an observed RL-vs-baseline advantage is a genuine, stable effect or an artifact of one specific simulator configuration.

**Protocol:** re-run the core comparison (Stage 3 experiment) while perturbing, one at a time: the $\alpha$ and $\beta$ distribution bounds ($\pm 20\%$), response noise $\sigma$, the constants inside $A_{\text{effect}}$ ([Sec. 11](#simulator-dynamics)), the difficulty distribution of curriculum questions, and the population composition (e.g., skewing the population toward high-$h$ learners). For each perturbation, report whether the RL policy's relative advantage over the Rule baseline holds, shrinks, or disappears. A result that only appears under one narrow configuration is reported as fragile, not hidden.

## 22. Ablation Studies

Each ablation reuses the full Stage-3 protocol (same seeds, same metrics, same held-out evaluation) with exactly one design element changed:

- **State ablation:** full state (Sec. 13) vs. state minus hint-usage vs. minus recent-accuracy vs. minus prerequisite feature (if the v1.1 extension is in use).
- **Reward ablation:** v1 vs. v2 vs. v3 ([Sec. 15](#reward-design)) — compare not just final metrics but the *qualitative* change in which actions the policy prefers.
- **Learner-population ablation:** a narrow-$\theta$-range population vs. the full, diverse range in [Sec. 9](#parametric-learner-population).
- **Action-space ablation** *(optional, stretch)*: the full 5-action space vs. a reduced 3-action set (EXPLAIN, HINT, PRACTICE).

The point of every ablation is to answer *"which design choices actually mattered?"* — not to pad the results section. An ablation with no interesting delta is still reported; a null result here is informative.

## 23. Experiment Logging & Structure

Every experiment run must record, at minimum: experiment ID, all random seeds (learner sampling + RL training), simulator version identifier, learner-parameter distributions used (verbatim from the config), RL hyperparameters (learning rate, discount factor, exploration schedule), state-representation version, reward version, baseline(s) compared, training and evaluation configuration, computed metrics (raw + summary statistics), timestamp, and the git commit hash of the code that produced the run. Lightweight JSON/CSV files under `experiments/<run_id>/` are sufficient — no experiment-tracking platform is justified at this stage ([Sec. 3 of the original constraints, retained](#important--do-not-over-engineer), i.e. avoid tooling that exists to look professional rather than to answer a question).

# Part VI — LLM Delivery Layer (Strictly Decoupled)

The LLM layer is built **only after** the RL core is validated on the simulator (Stage 5, not before). Its job is narrow: *render a pedagogical action into natural-language content.* It must never decide *which* action to take.

Correct: RL selects EXAMPLE → LLM generates an example.
Incorrect: LLM decides whether to explain, exemplify, or hint while RL "assists." This would quietly convert the research contribution from an RL decision system into an LLM agent with an RL suggestion box, which is a different (and much weaker, for this project's purposes) research claim.

## 24. LLM Abstraction Architecture

```
RL Policy -> Pedagogical Action --+
                                   v
                     LLMProvider (abstract interface)
                     generate_tutoring_response(
                         action, concept, learner_state, curriculum_context
                     ) -> TutorResponse(text, success, meta)
                                  |
              +-------------------+-------------------+
              |                   |                    |
              v                   v                    v
       GeminiAdapter    OpenAICompatibleAdapter   OfflineTemplateAdapter
       (native Gemini)  (OpenAI, Grok/xAI,          (curriculum-stored
                          Ollama, vLLM, any           templates, zero
                          OpenAI-compatible           network dependency)
                          /chat/completions API)
```

The Research Core imports only the abstract `LLMProvider` interface — never a specific adapter. Provider-specific imports are confined to `src/llm/providers/`. This is what allows the underlying LLM to change without touching the environment, learner model, state representation, reward function, or policy.

**Do not** build a separate hand-rolled adapter for every provider on day one. One native adapter (Gemini, since it pairs with the free-tier constraint) plus one OpenAI-compatible adapter (covers OpenAI, Grok/xAI, Ollama, vLLM, and any future service exposing a compatible `/chat/completions`-style endpoint) plus the offline fallback is sufficient; more native adapters are added only if a real, specific need arises.

**Config-driven selection**, never hard-coded:

```
LLM_PROVIDER=gemini            # or: openai_compatible | offline
LLM_MODEL=<current flash-tier model — verify against the provider's docs at build time, not against this document>
LLM_API_KEY=<from environment, never committed>
LLM_BASE_URL=<only for openai_compatible, e.g. a local Ollama endpoint>
```

A single factory function, `get_llm_provider()`, reads this configuration and returns the appropriate adapter instance.

## 25. Provider Switching Design

Switching providers must require **only a config change** (the `LLM_PROVIDER`/`LLM_MODEL` environment variables), never a code change to anything outside `src/llm/`. This is tested directly ([Sec. 31](#testing-strategy)): the same test suite for prompt construction and response handling must pass against a mocked Gemini adapter and a mocked OpenAI-compatible adapter without modification.

## 26. Offline Fallback & Failure Handling

External LLM failure must never crash the Research Core or the demo. Every provider call is wrapped with a timeout and a try/except; on any failure — authentication error, timeout, malformed response, rate limit, or empty output — the system falls back automatically to the `OfflineTemplateAdapter` for that single call, and logs the fallback event (provider, error type, timestamp) rather than surfacing a stack trace. The `OfflineTemplateAdapter` draws from pre-written per-action, per-concept template text stored alongside the curriculum content, so the Research Core and the demo remain fully runnable with **no API key configured at all** — this is a hard requirement, not an aspiration.

## 27. LLM Output Validation

Where the LLM is asked to produce *structured* content — specifically, new curriculum modules generated from freeform notes (the "add material" extensibility feature) — its JSON output is validated before being accepted, never trusted blindly:

- Required schema fields present ([Sec. 28](#curriculum-architecture)).
- `concept_id`s unique within the module.
- Every `prerequisite` reference either empty or points to an existing `concept_id`.
- `question.type` is one of the allowed enum values.
- No empty `prompt`/`answer` fields.

On validation failure: retry once, sending the specific validation errors back to the LLM with a request to correct them. On a second failure, reject the output and surface the concrete validation errors — never silently accept malformed content, and never crash.

# Part VII — Curriculum & Software Architecture

## 28. Curriculum Architecture

Content is fully decoupled from the RL engine via a generic JSON schema — adding a subject never means touching RL code:

```
{
  "module_id": str, "title": str,
  "concepts": [
    { "concept_id": str, "name": str,
      "prerequisites": [concept_id, ...],
      "difficulty": 1-5,
      "questions": [
        { "question_id": str, "prompt": str, "answer": str,
          "type": "mcq" | "short" | "code" }
      ]
    }
  ]
}
```

The initial seed module targets GATE CS "Programming & Data Structures" fundamentals (Variables, Conditionals, Loops, Functions, Lists) — deliberately compact (5 concepts, 5–8 questions each). `[future work]`: further GATE-relevant modules (Algorithms, Discrete Mathematics, Probability & Statistics, Linear Algebra, Computer Organization, Operating Systems, DBMS) can be added as additional module files without any RL-engine changes — this is the extensibility the schema exists to guarantee, not an incidental benefit.

## 29. Research Core vs. Delivery vs. Demo Layer

- **Research Core** (`src/environment`, `src/learner_model`, `src/simulator`, `src/rl`, `src/teachers`, `src/evaluation`): pure Python, no Streamlit import anywhere, no LLM import anywhere. Every experiment in this document can be run headlessly from a script or notebook.
- **Delivery Layer** (`src/llm`): the provider abstraction. Depends on the Research Core's action/concept/state types, but the Research Core never depends on it.
- **Demo Layer** (`app/`): Streamlit only. Imports from both layers above; contains zero decision logic of its own.

This separation is what makes headless, reproducible experimentation possible and keeps the "is this real research or a chatbot with a plot attached" line unambiguous.

## 30. Repository Structure

```
airtutor/
  data/curriculum/         one JSON file per module
  configs/                 hyperparameters, learner-population distributions,
                            per-experiment configs (YAML/JSON, never hard-coded)
  src/
    environment/           state encoder, action space, transition/reward orchestration
    learner_model/          observable mastery estimator
    simulator/              parametric population, dynamics, sanity-check tests live here
    rl/                     tabular Q-learning agent, training loop
    teachers/               random, fixed, rule, (optional) oracle baselines
    evaluation/             metrics, comparison runner, statistics, plotting
    llm/
      provider.py            abstract interface + get_llm_provider() factory
      providers/             gemini.py, openai_compatible.py, offline.py
      validation.py          schema validation for LLM-generated curriculum content
  experiments/              per-run: config snapshot, seeds, results.json, plots/
  app/                      Streamlit demo — imports core, contains no logic
  tests/                    unit / integration / experiment / UI, mirroring src/
  docs/                     research_problem.md, methodology.md, this roadmap,
                            AI_ASSISTANCE_LOG.md
  README.md
  requirements.txt
  .env.example
  .gitignore
```

This is a proposal grounded in the responsibilities above, not a template applied for its own sake — e.g. there is no `configs/` bloat beyond what the reproducibility requirements in [Sec. 32](#reproducibility) actually need, and no `docs/` subfolder structure beyond what a professor would actually need to navigate.

## 31. Testing Strategy

`[proposed]` Four levels, each with a concrete target — and note the honest framing: the goal is **graceful failure**, not a literal claim of being crash-proof.

- **Unit:** state encoder (history → expected bucket tuple); learner model (mastery moves the right direction); the five simulator sanity checks ([Sec. 12](#simulator-validity-assumptions-and-threats-to-validity)); reward calculation (matches the formula for known inputs); Q-table update (matches the Bellman update for a known transition); baseline policies (deterministic given fixed inputs); LLM adapter interface (a mocked provider returns the expected shape; a simulated failure triggers the offline fallback).
- **Integration:** the full environment–simulator–RL loop runs $N$ steps without error; the selected action correctly triggers the corresponding LLM adapter call; the offline fallback path produces valid output with no API key configured at all.
- **Experiment:** a fixed-seed run reproduces identical metrics across two separate executions; the results JSON matches the logging schema in [Sec. 23](#experiment-logging--structure); the training/evaluation seed ranges are asserted disjoint (no accidental leakage).
- **UI:** the Streamlit app loads without error; one full interaction completes end-to-end; the UI degrades to a visible, non-crashing message if the LLM call fails.

## 32. Reproducibility

A professor (or anyone else) should be able to clone the repository and understand exactly how a reported result was produced. This requires: pinned dependency versions in `requirements.txt`; every hyperparameter and distribution living in `configs/`, never hard-coded; every experiment saving its full config snapshot, all seeds used, and the git commit hash alongside its results and plots under `experiments/<run_id>/`; a `docs/methodology.md` explaining the experimental design in prose, cross-referencing this document; an explicit "Limitations & Future Work" statement wherever results are presented; and a running `AI_ASSISTANCE_LOG.md` disclosing what was AI-generated versus hand-written and verified.

# Part VIII — Build Plan

## 33. AI-Assisted Development Workflow

Using AI to help build this is fully compatible with the project — but the person building it must remain able to explain every line implementing state, reward, transition, exploration, or policy selection, without opening an AI assistant to check. Workflow: write or request a first version → critique/debug it → run the tests in [Sec. 31](#testing-strategy) → inspect the diff → explain the code back to yourself → commit. `AI_ASSISTANCE_LOG.md` records, per stage, what was AI-generated and what was hand-verified. Boilerplate (UI wiring, plotting, refactoring, documentation drafts) can be delegated freely; the RL algorithm, reward function, and simulator dynamics should never be pasted in without being understood.

## 34. Phased Implementation Roadmap

No hard calendar dates are assumed here (the original 1–2 day / Oct / Nov timeline no longer applies); each stage is defined by a verifiable output and can be mapped onto whatever calendar is relevant once one is fixed. Rough relative effort is noted for planning only.

| Stage | Deliverable | Resolves |
|---|---|---|
| 0 | Repo scaffold; a trivial hand-rolled (non-Gymnasium) toy environment + tabular Q-learning proving the mechanical loop end-to-end | "Does the RL loop even work, mechanically?" |
| 1 *(largest stage)* | Learner model; parametric simulator with the six $\theta$ traits; all five sanity checks passing as automated tests | "Is the simulator behaving sensibly before anything is built on top of it?" |
| 2 | Observable state encoder; Random/Fixed/Rule baselines; core metrics | "What does a strong non-learned baseline even look like?" |
| 3 | Tabular Q-learning; held-out train/eval protocol; core RL-vs-baseline comparison with statistical reporting across seeds | RQ1, H1 |
| 4 | Reward v2/v3; robustness sensitivity sweep; state/reward/population ablations | RQ3, RQ4, H2, H3 |
| 5 | LLM abstraction (Gemini + OpenAI-compatible + offline adapters); output validation; curriculum "add material" extensibility | Decouples delivery from decision-making; enables extensibility |
| 6 | Streamlit demo (research core untouched); README; `docs/`; professor-facing packaging | Communicability |

Each stage's exit criterion is a **verifiable artifact** (a passing test suite, a reproducible plot, a working script) — not a date on a calendar.

## 35. Expected Outputs and Plots

Every plot exists to answer a specific question — none are generated merely to increase figure count.

| Plot | Answers |
|---|---|
| Training reward curve (RL) | Is the agent's policy actually improving during training? (basic sanity check) |
| Learning gain by teacher, with seed error bars | The core RQ1 comparison |
| Final mastery / time-to-mastery by teacher | Secondary confirmation of the core comparison |
| Performance distribution across learner configurations (box/violin per teacher) | Does RL help uniformly, or only for some learner types? |
| Held-out vs. training-set evaluation, side by side | Detects overfitting to the specific training population |
| Reward ablation (v1 vs v2 vs v3) | RQ3 / H2 — does richer reward prevent degenerate policies? |
| State ablation (metric deltas per removed feature) | RQ4 — which state features are load-bearing? |
| Robustness/sensitivity sweep (metric vs. perturbation magnitude) | Is the RL advantage, if any, fragile or stable? |

# Part IX — Honest Accounting

## 36. Failure Modes

| Mode | Symptom | Detection | Mitigation |
|---|---|---|---|
| Reward hacking | High reward / near-perfect immediate accuracy, flat independent-mastery signal | Held-out, post-test-style evaluation questions | Reward v2/v3; monitor action-distribution entropy |
| Simulator circularity | RL policy exactly reproduces a hard-coded rule | Code review; compare learned policy to rule-baseline decision boundaries | The smooth, parametric mechanism in [Sec. 11](#simulator-dynamics); independent design of rule-baseline thresholds |
| State-space explosion | Q-table fails to converge; most states visited once | State-visitation histogram | More aggressive discretization; drop low-value features |
| LLM hallucination in generated curriculum | Invalid schema or factually wrong content | The validation layer ([Sec. 27](#llm-output-validation)) | Retry-with-correction; human review before merging new modules |
| Provider variability | Different LLMs render the same action differently in tone/quality | Small manual spot-check across providers | Accepted, documented delivery-layer limitation — does **not** affect the RL research validity, since the policy's decision is unaffected by which provider renders it |
| Overfitting to simulator assumptions | RL advantage vanishes under any perturbation | [Sec. 21](#robustness-and-sensitivity-analysis) | Report honestly; do not claim robustness that wasn't tested |
| Over-engineering | Time spent on infrastructure with no corresponding research question answered | The "build evidence, not features" test on every proposed addition | Decline features that don't pass the test in the Executive Summary |
| Hardware limits | Training too slow or memory-heavy for the 8 GB machine | Wall-clock/memory profiling during Stage 0–1 | Tabular-only, CPU-only, no-neural-net constraint (retained from v1) |

## 37. Remaining Risks and Design Trade-offs

This section exists so the document does not overclaim its own soundness.

- **Simulator realism.** Six scalar traits are a simplification of real learning; the specific functional forms in [Sec. 11](#simulator-dynamics) are chosen for smoothness and interpretability, not validated against real interaction data. Calibration against real data, if any becomes available, is future work.
- **Circularity.** Mitigated by design ([Sec. 12](#simulator-validity-assumptions-and-threats-to-validity)), but not provably eliminated — a persistent risk requiring ongoing vigilance, not a box to check once.
- **Reward hacking.** A known, expected risk of reward v1 specifically; v2/v3 are attempts, not guarantees, and are themselves subject to ablation rather than assumed superior.
- **State-space explosion.** The v1 state (320 states) is deliberately conservative; any feature addition should be justified against the explosion risk, not added by default.
- **Tabular Q-learning limitations.** Cannot generalize across unseen discrete states (no function approximation); scaling to a richer state or larger curriculum without moving to function approximation (DQN or similar) will eventually hit a wall — explicitly future work, not attempted now for hardware and rigor-of-understanding reasons (Sec. 8 of the original PDF's "AI-assisted development protocol," retained: never rely on a library implementation of something you can't first build small yourself).
- **Baseline strength.** The Rule baseline's quality is itself a design choice that can be second-guessed; a weak Rule baseline would make any RL win meaningless, which is precisely why [Sec. 16](#baseline-teachers) insists it be genuinely competitive.
- **Limited curriculum.** One module (5 concepts) at Stage 0–4; results should not be assumed to generalize across the full GATE CS syllabus without further modules being added and re-tested.
- **Simulated vs. human learning.** Addressed directly in [Sec. 12](#simulator-validity-assumptions-and-threats-to-validity); repeated here because it is the single most important limitation of the entire project as scoped.
- **LLM hallucination and provider variability.** Bounded by the validation layer and the deliberate decoupling from decision-making, but not eliminated for the *delivery* layer's content quality.
- **API availability.** Free-tier rate limits on any given provider are real and can interrupt work; the offline fallback exists specifically so this never blocks the Research Core.
- **Reproducibility gap between layers.** The Research Core (tabular RL) is fully deterministic given a seed; the LLM delivery layer is inherently less deterministic across calls — this asymmetry is documented, not hidden.
- **Overfitting to simulator assumptions.** Addressed by [Sec. 21](#robustness-and-sensitivity-analysis); genuinely unresolved until that analysis is actually run, not merely planned.
- **Over-engineering tension.** This document itself is far more elaborate than a lightweight implementation — that tension is real and is resolved by keeping the *implementation* lightweight (tabular methods, plain Python, JSON/CSV logging) even where the *methodology* is rigorous. Rigor and complexity are not the same axis, and this document deliberately optimizes the first without the second.
- **Hardware limitations.** CPU-only, 8 GB RAM bounds every algorithmic choice in this document; this is treated as a design constraint to work within, not a temporary inconvenience to route around with cloud compute.

## 38. Research Ethics / Limitations

No real human subjects are involved through Stage 6 — the system operates entirely on simulated learners. If a real-user pilot is ever pursued (Level 6, [Sec. 40](#future-research-extensions)), it requires informed consent and appropriate institutional review before any data collection; that requirement is not waived by this being a student project. The system must never be presented, at any stage described in this document, as validated for real educational deployment. A further equity consideration worth naming even though it isn't solvable at this stage: a policy trained on a narrow simulated population may not transfer fairly across real learners with different backgrounds, prior knowledge, or access to resources — this is flagged as a future-work concern, not resolved here.

## 39. Terminology Standards

To stay academically precise and avoid overclaiming, this document (and any derived research writing) consistently prefers:

| Prefer | Over | Because |
|---|---|---|
| "simulated learner" | "learner" / "student" (unqualified) | Avoids implying human-subject evidence |
| "policy decision rationale" / "transparent policy inspection" | "explainable AI" | The Q-table lookup is inspectable, not a claim of formal XAI |
| "held-out simulated learner configurations" | "generalization" (unqualified) | Precise about what was actually tested |
| "preliminary simulated evidence" | "proof" | No claim survives beyond what the evaluated conditions support |

Every claim in project documentation should additionally be tagged `[implemented]`, `[experimentally validated]`, `[proposed]`, or `[future work]` — as this document itself attempts to do throughout.

# Part X — Forward-Looking & Closing

## 40. Future Research Extensions

Explicitly **not** attempted before Stage 6, and not attempted out of order — each depends on the previous levels producing trustworthy results:

- **Level 2 — Knowledge tracing:** replace the hand-built learner-model update ([Sec. 8](#learner-model)) with a standard knowledge-tracing model (e.g. Bayesian Knowledge Tracing).
- **Level 3 — Personalized policy:** learn which interventions work for *individual* simulated learners rather than the population-level policy.
- **Level 4 — Unseen-learner adaptation:** rapid adaptation to a previously unseen simulated profile within a session, rather than only held-out *evaluation*.
- **Level 5 — Counterfactual evaluation:** estimate outcomes of alternative teaching actions where feasible (off-policy evaluation) — an advanced extension, not a requirement for any stage in this document.
- **Level 6 — Real-user pilot:** a small, consented, appropriately reviewed study — only once simulation-based evaluation is stable and trusted ([Sec. 38](#research-ethics--limitations)).

Pick **one** advanced extension at a time once Stage 6 is stable, per the same "build evidence, not features" discipline that governs everything else in this document.

## 41. Professor-Facing 3-Minute Demonstration

- **0:00–0:30** — the research question: RL-based pedagogical decision-making, not "an AI tutor."
- **0:30–1:15** — architecture: hidden learner traits → observable state → RL policy → pedagogical action → LLM renders it. Emphasize the decoupling explicitly.
- **1:15–2:00** — a live moment: the learner's observable state changes (e.g. a wrong-answer streak appears), the policy visibly switches its selected action, and the LLM-rendered content changes accordingly.
- **2:00–2:40** — the held-out comparison plot: RL vs. Rule vs. Fixed, reported honestly — including if RL does not yet clearly win.
- **2:40–3:00** — one specific, named remaining risk (from [Sec. 37](#remaining-risks-and-design-trade-offs)) and the specific next experiment planned to address it.

The strongest moment in the demo is not visual polish — it is showing that a decision changed *because the learner's state changed*, not because of a scripted branch.

## 42. Definition of Done (per stage)

A stage is done when: its test suite passes ([Sec. 31](#testing-strategy)); its deliverable ([Sec. 34](#phased-implementation-roadmap)) is reproducible from a clean clone with a fixed seed; and the person building it can explain, unaided, every design choice the stage introduced.

## 43. Final Research Checklist

- [ ] State, action, and reward can be defined and justified without notes
- [ ] The five simulator sanity checks pass as automated tests
- [ ] Random, Fixed, and Rule baselines are implemented, and the Rule baseline is genuinely competitive
- [ ] The RL agent trains and produces a learning curve across multiple seeds
- [ ] Held-out (disjoint-seed) evaluation is implemented and distinct from training evaluation
- [ ] Reward v1 vs. v2/v3 comparison has been run, not just proposed
- [ ] At least one state ablation and one reward ablation have been run
- [ ] A robustness/sensitivity sweep has been run against at least two simulator assumptions
- [ ] If RL does not beat the Rule baseline, that result is reported plainly, unmodified
- [ ] The LLM layer is provider-agnostic, config-switchable, and never crashes the Research Core (verified by disconnecting the API key entirely)
- [ ] Generated curriculum content is schema-validated before acceptance
- [ ] The Research Core runs headlessly, with no Streamlit or LLM import
- [ ] Every experiment's seeds, config, and git commit hash are logged alongside its results
- [ ] `AI_ASSISTANCE_LOG.md` and `docs/methodology.md` exist and are current
- [ ] The "Remaining Risks and Design Trade-offs" section reflects the *actual* state of the project, not the aspirational one
- [ ] Terminology throughout project documentation follows [Sec. 39](#terminology-standards)

---

*This document supersedes the 29 August 2026 MVP roadmap in full. It is itself `[proposed]` until Stage 0 begins, and should be revised as stages complete — treat it as a living specification, not a fixed contract.*
