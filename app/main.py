"""AirTutor: Streamlit Demo & Research Artifact Inspection App (Roadmap Sec. 29, 39, 41).

This application serves exclusively as the presentation layer (Demo Layer) for AirTutor:
1. Interactive Tutoring Session: selects concepts, answers questions, observes mastery updates,
   and views LLM-rendered pedagogical interventions.
2. Transparent Policy Inspection: examines observable state features, flat state index (0-319),
   and the learned Q-value distribution Q(s, a) without overclaiming as "explainable AI".
3. Compare Teachers & Research Benchmarks: displays Stage 3 & Stage 4 empirical results,
   including core baseline comparisons, reward ablations, state ablations, and robustness sweeps.
4. Add Curriculum Material: provides a text-area ingestion flow transforming unstructured notes
   into schema-validated curriculum modules via src/llm/add_material.py.

ARCHITECTURAL INVARIANT (GEMINI.md & Roadmap Sec. 29):
This presentation layer imports exclusively from src/ and contains ZERO decision logic:
- No Q-learning Bellman update equations.
- No reward computation functions.
- No simulator dynamics equations.
- Fully decoupled from provider-specific LLM implementations.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

# Ensure repository root is on sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.environment.actions import ACTIONS, NUM_ACTIONS, PedagogicalAction
ACTION_NAMES = [a.name for a in ACTIONS]
from src.environment.curriculum import Concept, CurriculumRepository, Question, load_curriculum
from src.environment.state_encoder import (
    ObservableHistory,
    StateEncoderConfig,
    encode_state,
    get_total_state_count,
    state_tuple_to_index,
)
from src.learner_model.mastery import LearnerModel
from src.llm.add_material import add_module_from_notes
from src.llm.provider import LLMProvider, TutorResponse, get_llm_provider
from src.rl.q_learning import TabularQAgent
from src.rl.train import train_agent_single_seed
from src.teachers.base_teacher import TeacherObservation


# ============================================================================
# --- Page Configuration & Styling ---
# ============================================================================

st.set_page_config(
    page_title="AirTutor — Adaptive Pedagogical RL",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .stCard {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 1.2rem;
        border: 1px solid #E2E8F0;
        margin-bottom: 1rem;
    }
    .action-badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        font-weight: 700;
        font-size: 0.95rem;
        border-radius: 6px;
        background-color: #EEF2FF;
        color: #4F46E5;
        border: 1px solid #C7D2FE;
    }
    .state-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        font-size: 0.85rem;
        border-radius: 4px;
        background-color: #F1F5F9;
        color: #334155;
        border: 1px solid #CBD5E1;
        margin-right: 0.4rem;
    }
    .disclaimer-box {
        font-size: 0.82rem;
        color: #64748B;
        background-color: #F8FAFC;
        border-left: 3px solid #94A3B8;
        padding: 0.6rem 0.8rem;
        margin-top: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# --- Cached Resource Loaders ---
# ============================================================================

@st.cache_resource
def get_cached_curriculum() -> CurriculumRepository:
    """Load all curriculum modules found in data/curriculum/."""
    return load_curriculum()


@st.cache_resource
def get_cached_policy() -> TabularQAgent:
    """Load or train a TabularQAgent policy.
    
    Guarantees that the app runs instantly from a clean clone even before
    any offline experiments have been saved to disk.
    """
    saved_path = Path("experiments/trained_q_agent.npy")
    if saved_path.exists():
        agent = TabularQAgent(seed=42)
        agent.load(saved_path)
        return agent

    # Fast on-the-fly training (~0.6s) across 300 simulated training learners
    curriculum = load_curriculum()
    agent, _ = train_agent_single_seed(
        rl_seed=42,
        learner_seeds=list(range(300)),
        curriculum=curriculum,
        num_episodes=300,
        reward_version="v1",
    )
    saved_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(saved_path)
    return agent


@st.cache_resource
def get_cached_provider() -> LLMProvider:
    """Retrieve the provider-agnostic LLM interface configured by environment."""
    return get_llm_provider()


# ============================================================================
# --- Session State Initialization ---
# ============================================================================

def init_session_state(curriculum: CurriculumRepository) -> None:
    """Initialize interactive tutoring session state if not already present."""
    modules = curriculum.list_modules()
    default_module_id = modules[0] if modules else "gate_programming_fundamentals"

    if "selected_module_id" not in st.session_state:
        st.session_state.selected_module_id = default_module_id

    active_module = curriculum.get_module(st.session_state.selected_module_id)
    concept_ids = [c.concept_id for c in active_module.concepts] if active_module else []

    if "learner_model" not in st.session_state or st.session_state.get("_current_module_id") != st.session_state.selected_module_id:
        st.session_state.learner_model = LearnerModel(concepts=concept_ids)
        st.session_state._current_module_id = st.session_state.selected_module_id
        st.session_state.current_concept_idx = 0
        st.session_state.history_responses = []  # list of bool
        st.session_state.wrong_streak = 0
        st.session_state.hint_used_last = False
        st.session_state.last_response_correct = None
        st.session_state.step_count = 0
        st.session_state.last_intervention = None
        st.session_state.last_action = None
        st.session_state.last_state_idx = None
        st.session_state.last_q_values = None
        st.session_state.interaction_log = []


def reset_session() -> None:
    """Reset the current learner session state."""
    curriculum = get_cached_curriculum()
    active_module = curriculum.get_module(st.session_state.selected_module_id)
    concept_ids = [c.concept_id for c in active_module.concepts] if active_module else []
    st.session_state.learner_model = LearnerModel(concepts=concept_ids)
    st.session_state.current_concept_idx = 0
    st.session_state.history_responses = []
    st.session_state.wrong_streak = 0
    st.session_state.hint_used_last = False
    st.session_state.last_response_correct = None
    st.session_state.step_count = 0
    st.session_state.last_intervention = None
    st.session_state.last_action = None
    st.session_state.last_state_idx = None
    st.session_state.last_q_values = None
    st.session_state.interaction_log = []


# ============================================================================
# --- Main Application Layout ---
# ============================================================================

def main() -> None:
    curriculum = get_cached_curriculum()
    policy = get_cached_policy()
    provider = get_cached_provider()

    init_session_state(curriculum)

    # Top Header
    st.markdown('<div class="main-header">AirTutor: Adaptive Pedagogical Decisions</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        'An RL policy selects pedagogical interventions conditioned purely on observable learner features; '
        'a decoupled LLM layer renders the chosen action into natural language.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Sidebar: Module Selection & Diagnostics
    with st.sidebar:
        st.header("Session Settings")
        module_options = curriculum.list_modules()
        if not module_options:
            st.warning("No curriculum modules loaded.")
            return

        selected_module_id = st.selectbox(
            "Active Curriculum Module",
            options=module_options,
            index=module_options.index(st.session_state.selected_module_id)
            if st.session_state.selected_module_id in module_options
            else 0,
            key="module_selector_sb",
        )
        if selected_module_id != st.session_state.selected_module_id:
            st.session_state.selected_module_id = selected_module_id
            reset_session()
            st.rerun()

        active_module = curriculum.get_module(st.session_state.selected_module_id)
        st.caption(f"**Module:** {active_module.title} ({len(active_module.concepts)} concepts)")

        if st.button("Reset Learner Session", use_container_width=True):
            reset_session()
            st.success("Session reset.")
            st.rerun()

        st.divider()
        st.subheader("System Architecture")
        provider_name = os.getenv("LLM_PROVIDER", "gemini" if os.getenv("GEMINI_API_KEY") else "offline_template")
        st.markdown(f"**Active LLM Adapter:** `{provider_name}`")
        st.markdown(f"**RL Policy:** Tabular Q-Learning (320 States)")
        st.markdown(f"**Observable Feature Dimension:** 5 Features")

        st.divider()
        st.subheader("Estimated Masteries $\\hat{M}(c)$")
        learner_model: LearnerModel = st.session_state.learner_model
        for c in active_module.concepts:
            m_val = learner_model.get_mastery(c.concept_id)
            st.write(f"**{c.name}:** {m_val:.2f}")
            st.progress(min(max(float(m_val), 0.0), 1.0))

        st.markdown(
            '<div class="disclaimer-box">'
            '<strong>Research Note:</strong> Observable mastery $\\hat{M}$ is an empirical estimate '
            'derived from response history. True learner traits ($\\theta$) are strictly hidden from policy.'
            '</div>',
            unsafe_allow_html=True,
        )

    # Main Navigation Tabs
    tab_session, tab_inspection, tab_benchmark, tab_add_material = st.tabs([
        "Interactive Tutoring Session",
        "Transparent Policy Inspection",
        "Compare Teachers & Benchmarks",
        "Add Curriculum Material",
    ])

    # ------------------------------------------------------------------------
    # TAB 1: Interactive Tutoring Session
    # ------------------------------------------------------------------------
    with tab_session:
        render_tutoring_session(active_module, policy, provider)

    # ------------------------------------------------------------------------
    # TAB 2: Transparent Policy Inspection
    # ------------------------------------------------------------------------
    with tab_inspection:
        render_policy_inspection(policy, active_module)

    # ------------------------------------------------------------------------
    # TAB 3: Compare Teachers & Research Benchmarks
    # ------------------------------------------------------------------------
    with tab_benchmark:
        render_benchmark_tab()

    # ------------------------------------------------------------------------
    # TAB 4: Add Curriculum Material
    # ------------------------------------------------------------------------
    with tab_add_material:
        render_add_material_tab(provider)


# ============================================================================
# --- Tab 1: Interactive Tutoring Session ---
# ============================================================================

def render_tutoring_session(active_module: Any, policy: TabularQAgent, provider: LLMProvider) -> None:
    """Render the active tutoring loop: Question -> Answer -> Mastery -> Policy -> LLM."""
    if not active_module.concepts:
        st.info("No concepts in this module.")
        return

    concepts = active_module.concepts
    curr_idx = min(st.session_state.current_concept_idx, len(concepts) - 1)
    current_concept: Concept = concepts[curr_idx]
    learner_model: LearnerModel = st.session_state.learner_model

    col_concept, col_nav = st.columns([3, 1])
    with col_concept:
        st.subheader(f"Current Concept: {current_concept.name}")
        prereqs = current_concept.prerequisites or ["None"]
        st.caption(f"**Prerequisites:** {', '.join(prereqs)} | **Difficulty:** {current_concept.difficulty}/5")
    with col_nav:
        if curr_idx < len(concepts) - 1:
            if st.button("Advance to Next Concept →", use_container_width=True):
                st.session_state.current_concept_idx += 1
                st.rerun()

    # Current Mastery Indicator
    current_m = learner_model.get_mastery(current_concept.concept_id)
    st.write(f"Estimated Mastery $\\hat{{M}}({current_concept.concept_id})$: **{current_m:.3f}**")
    st.progress(min(max(float(current_m), 0.0), 1.0))

    # Pick a question for this concept
    questions: list[Question] = current_concept.questions
    q_idx = st.session_state.step_count % len(questions) if questions else 0
    current_q = questions[q_idx] if questions else None

    if not current_q:
        st.warning("No questions available for this concept.")
        return

    st.markdown("---")
    st.markdown("### Question Challenge")
    st.markdown(f"**Prompt ({current_q.type.upper()}):**")
    st.info(current_q.prompt)

    # Answer Input Form
    with st.form(key=f"answer_form_{st.session_state.step_count}"):
        col_ans, col_hint = st.columns([3, 1])
        with col_ans:
            user_answer = st.text_input("Your Answer:", placeholder=f"Expected format: {current_q.answer}")
        with col_hint:
            used_hint = st.checkbox("Used a hint for this step", value=False)

        col_sub1, col_sub2, col_sub3 = st.columns([1, 1, 2])
        with col_sub1:
            submit_custom = st.form_submit_button("Submit Answer", type="primary", use_container_width=True)
        with col_sub2:
            simulate_correct = st.form_submit_button("Simulate Correct", use_container_width=True)
        with col_sub3:
            simulate_wrong = st.form_submit_button("Simulate Incorrect", use_container_width=True)

    if submit_custom or simulate_correct or simulate_wrong:
        if simulate_correct:
            is_correct = True
            ans_given = current_q.answer
        elif simulate_wrong:
            is_correct = False
            ans_given = "Incorrect response (simulated)"
        else:
            ans_given = user_answer.strip()
            is_correct = (ans_given.lower() == current_q.answer.strip().lower())

        # 1. Update Observable Learner Model (No latent theta, pure observable history)
        prev_m = learner_model.get_mastery(current_concept.concept_id)
        new_m = learner_model.update(current_concept.concept_id, is_correct)
        delta_m = new_m - prev_m

        # 2. Update interaction signals
        st.session_state.last_response_correct = is_correct
        st.session_state.history_responses.append(is_correct)
        if is_correct:
            st.session_state.wrong_streak = 0
        else:
            st.session_state.wrong_streak += 1
        st.session_state.hint_used_last = used_hint
        st.session_state.step_count += 1

        # 3. Assemble TeacherObservation
        recent_3 = st.session_state.history_responses[-3:]
        obs = TeacherObservation(
            current_concept_id=current_concept.concept_id,
            estimated_mastery=new_m,
            all_estimated_masteries=learner_model.get_all_masteries(),
            prerequisite_ids=current_concept.prerequisites,
            weakest_prereq_id=None,
            weakest_prereq_mastery=None,
            last_response_correct=is_correct,
            recent_responses=recent_3,
            wrong_answer_streak=st.session_state.wrong_streak,
            hint_used_last_step=used_hint,
            step_count=st.session_state.step_count,
        )

        # 4. Policy Action Selection (Greedy lookup into learned Q-table)
        selected_action = policy.select_action(obs, greedy=True)
        state_idx = policy.get_state_index(obs)
        q_row = policy.q_table[state_idx].copy()

        # 5. LLM Content Rendering (Delivery Layer)
        curriculum_context = {
            "module_id": active_module.module_id,
            "module_title": active_module.title,
            "concept_name": current_concept.name,
        }
        with st.spinner("Decoupled LLM Layer rendering pedagogical intervention..."):
            response = provider.generate_tutoring_response(
                action=selected_action,
                concept=current_concept,
                learner_state=obs,
                curriculum_context=curriculum_context,
            )

        # Store in session state
        st.session_state.last_intervention = response
        st.session_state.last_action = selected_action
        st.session_state.last_state_idx = state_idx
        st.session_state.last_q_values = q_row
        st.session_state.interaction_log.append({
            "step": st.session_state.step_count,
            "concept": current_concept.name,
            "correct": is_correct,
            "action": selected_action.name,
            "delta_m": delta_m,
        })
        st.rerun()

    # Display Last Intervention Result
    if st.session_state.last_intervention:
        resp: TutorResponse = st.session_state.last_intervention
        action: PedagogicalAction = st.session_state.last_action
        st.markdown("---")
        st.subheader("Selected Pedagogical Intervention & LLM Delivery")

        c_badge, c_status = st.columns([1, 3])
        with c_badge:
            st.markdown(f'<div class="action-badge">Policy Action: {action.name}</div>', unsafe_allow_html=True)
        with c_status:
            meta = resp.meta
            prov = meta.get("provider", "offline_template")
            cached = meta.get("cached", False)
            fallback = meta.get("fallback_used", False)
            st.caption(f"**Provider:** `{prov}` | **Cached:** `{cached}` | **Fallback:** `{fallback}`")

        # Intervention Content Card
        st.markdown(resp.text)

    # Interaction History Table
    if st.session_state.interaction_log:
        with st.expander("Session Interaction History", expanded=False):
            df_hist = pd.DataFrame(st.session_state.interaction_log)
            st.dataframe(df_hist, use_container_width=True)


# ============================================================================
# --- Tab 2: Transparent Policy Inspection ---
# ============================================================================

def render_policy_inspection(policy: TabularQAgent, active_module: Any) -> None:
    """Render transparent Q-table lookup and observable state decomposition (Roadmap Sec. 39)."""
    st.subheader("Transparent Policy Inspection")
    st.markdown(
        "Per **Roadmap Sec. 39**, this panel provides an exact, inspectable readout of the "
        "policy's decision mechanics. Tabular Q-learning is an explicit lookup into a learned "
        "state-action value table $Q(s, a)$, not a black box."
    )

    learner_model: LearnerModel = st.session_state.learner_model
    concepts = active_module.concepts
    curr_idx = min(st.session_state.current_concept_idx, len(concepts) - 1)
    current_concept = concepts[curr_idx]

    # Current Live State Inspection
    st.markdown("### 1. Current Live State Breakdown")
    m_val = learner_model.get_mastery(current_concept.concept_id)
    history = ObservableHistory(
        concept_id=current_concept.concept_id,
        estimated_mastery=m_val,
        last_response_correct=st.session_state.last_response_correct,
        recent_responses=st.session_state.history_responses[-3:],
        wrong_answer_streak=st.session_state.wrong_streak,
        hint_used_last_step=st.session_state.hint_used_last,
    )
    state_tuple = encode_state(history, policy.state_config)
    state_idx = state_tuple_to_index(state_tuple, policy.state_config)
    q_row = policy.q_table[state_idx]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Mastery Bucket (0-4)", f"b_m = {state_tuple[0]}")
    c2.metric("Last Correct (0-1)", f"c_last = {state_tuple[1]}")
    c3.metric("3-Step Accuracy (0-3)", f"acc_3 = {state_tuple[2]}")
    c4.metric("Wrong Streak (0-3)", f"s_err = {state_tuple[3]}")
    c5.metric("Hint Used (0-1)", f"h = {state_tuple[4]}")

    st.write(f"**Mapped Flat State Index:** `{state_idx}` (out of 320 observable states)")

    # Q-values horizontal bar chart
    st.markdown("### 2. Learned Action-Values $Q(s, \\cdot)$ for State $s$")
    df_q = pd.DataFrame({
        "Pedagogical Action": ACTION_NAMES,
        "Q-Value": q_row,
    })
    st.bar_chart(df_q.set_index("Pedagogical Action"), color="#4F46E5")

    # Policy Rationale Statement
    max_action_idx = int(np.argmax(q_row))
    best_action = ACTIONS[max_action_idx]
    st.markdown(
        f"**Policy Decision Rationale:** For state `{state_idx}` "
        f"$(\\hat{{M}} \\in \\text{{bucket }} {state_tuple[0]}, "
        f"c_{{last}}={state_tuple[1]}, s_{{err}}={state_tuple[3]})$, "
        f"the policy selects **`{best_action.name}`** because it yields the maximal expected cumulative "
        f"discounted mastery gain ($Q = {q_row[max_action_idx]:.4f}$) across the training distribution."
    )

    st.divider()

    # Interactive "What-If" Explorer (Roadmap Sec. 41: Live Demonstration)
    st.markdown("### 3. Interactive 'What-If' State Explorer")
    st.caption("Manipulate the 5 observable features below to inspect how the policy dynamically reallocates interventions:")

    wi_col1, wi_col2, wi_col3, wi_col4, wi_col5 = st.columns(5)
    with wi_col1:
        wi_bm = st.selectbox("Mastery Bucket", options=[0, 1, 2, 3, 4], index=state_tuple[0])
    with wi_col2:
        wi_last = st.selectbox("Last Correct", options=[0, 1], index=state_tuple[1])
    with wi_col3:
        wi_acc = st.selectbox("Recent Acc (0-3)", options=[0, 1, 2, 3], index=state_tuple[2])
    with wi_col4:
        wi_streak = st.selectbox("Wrong Streak", options=[0, 1, 2, 3], index=state_tuple[3])
    with wi_col5:
        wi_hint = st.selectbox("Hint Used", options=[0, 1], index=state_tuple[4])

    wi_tuple = (wi_bm, wi_last, wi_acc, wi_streak, wi_hint)
    wi_idx = state_tuple_to_index(wi_tuple, policy.state_config)
    wi_q = policy.q_table[wi_idx]
    wi_best_idx = int(np.argmax(wi_q))
    wi_best_action = ACTIONS[wi_best_idx]

    st.markdown(
        f"**Hypothetical State Index:** `{wi_idx}` → Selected Action: "
        f'<span class="action-badge">{wi_best_action.name}</span> (Q = {wi_q[wi_best_idx]:.4f})',
        unsafe_allow_html=True,
    )

    df_wi = pd.DataFrame({"Action": ACTION_NAMES, "Q-Value": wi_q})
    st.dataframe(df_wi, use_container_width=True)


# ============================================================================
# --- Tab 3: Compare Teachers & Benchmarks ---
# ============================================================================

def render_benchmark_tab() -> None:
    """Render Stage 3/4 empirical plots and baseline comparison table."""
    st.subheader("Empirical Research Benchmark & Teacher Comparison")
    st.markdown(
        "Evaluation across $N=100$ held-out simulated learner configurations "
        "(seed range $[10000, 10100)$) strictly disjoint from the training population. "
        "Baselines implemented per **Roadmap Sec. 16 & 20**."
    )

    # Summary Metrics Table with 95% Confidence Intervals
    st.markdown("### 1. Benchmark Comparison Table (Stage 3)")
    benchmark_data = [
        {"Teacher": "Random", "Learning Gain ΔM_hat": "0.0115 ± 0.0119", "True Gain ΔM": "0.1657 ± 0.0296", "Efficiency": "0.0004", "Hint Rate": "21.8%", "Action Entropy": "2.23"},
        {"Teacher": "Fixed (Curriculum)", "Learning Gain ΔM_hat": "0.0179 ± 0.0126", "True Gain ΔM": "0.2152 ± 0.0411", "Efficiency": "0.0006", "Hint Rate": "0.0%", "Action Entropy": "1.00"},
        {"Teacher": "Rule Baseline", "Learning Gain ΔM_hat": "0.0280 ± 0.0139", "True Gain ΔM": "0.2339 ± 0.0308", "Efficiency": "0.0009", "Hint Rate": "8.0%", "Action Entropy": "1.40"},
        {"Teacher": "RL (Tabular Q)", "Learning Gain ΔM_hat": "0.1309 ± 0.0067", "True Gain ΔM": "0.2989 ± 0.0167", "Efficiency": "0.0044", "Hint Rate": "0.5%", "Action Entropy": "1.37"},
    ]
    st.dataframe(pd.DataFrame(benchmark_data), use_container_width=True)
    st.caption("Values report mean ± 95% bootstrap confidence interval over 100 held-out evaluation episodes.")

    st.markdown("---")
    st.markdown("### 2. Empirical Benchmark Figures (Stages 3 & 4)")

    # Plot 1 & 2: Core Learning Gain & Training Convergence
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        p1 = Path("experiments/learning_gain_comparison.png")
        if p1.exists():
            st.image(str(p1), caption="Figure 1: Observable vs. True Learning Gain across Teachers (p < 1e-15 vs. Rule).")
        else:
            st.info("learning_gain_comparison.png not found.")
    with col_p2:
        p2 = Path("experiments/training_reward_curve.png")
        if p2.exists():
            st.image(str(p2), caption="Figure 2: Multi-Seed RL Training Reward Convergence (5 independent seeds).")
        else:
            st.info("training_reward_curve.png not found.")

    # Plot 3 & 4: Reward & State Ablations
    col_p3, col_p4 = st.columns(2)
    with col_p3:
        p3 = Path("experiments/reward_ablation_comparison.png")
        if p3.exists():
            st.image(str(p3), caption="Figure 3: Reward Formulation Ablation (v1 mastery gain vs. v2 retention vs. v3 transfer).")
        else:
            st.info("reward_ablation_comparison.png not found.")
    with col_p4:
        p4 = Path("experiments/state_ablation_deltas.png")
        if p4.exists():
            st.image(str(p4), caption="Figure 4: Observable Feature Importance Deltas via Systematic State Ablation.")
        else:
            st.info("state_ablation_deltas.png not found.")

    # Plot 5: Robustness Sweep
    st.markdown("### 3. Simulator Robustness & Sensitivity Sweep")
    p5 = Path("experiments/sensitivity_sweep.png")
    if p5.exists():
        st.image(str(p5), caption="Figure 5: Robustness across 12 Perturbed Simulator Axes (±20% bounds, noise, intervention effects).")
    else:
        st.info("sensitivity_sweep.png not found.")

    st.markdown(
        '<div class="disclaimer-box">'
        '<strong>Scientific Validity Note (Roadmap Sec. 39):</strong> '
        'All results shown are derived from simulated learner populations sampled from parametric '
        'distributions. They constitute preliminary simulated evidence, not proof of human student learning gains.'
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================================
# --- Tab 4: Add Curriculum Material ---
# ============================================================================

def render_add_material_tab(provider: LLMProvider) -> None:
    """Render Prompt-6 syllabus/notes ingestion workflow into schema-validated JSON."""
    st.subheader("Add Curriculum Material")
    st.markdown(
        "Ingest unstructured educational notes, lecture slides, or syllabi into a "
        "schema-validated curriculum JSON module. The RL core remains completely untouched."
    )

    with st.form("add_material_form"):
        module_title = st.text_input("Module Title (Optional)", placeholder="e.g. Introduction to Binary Search Trees")
        raw_notes = st.text_area(
            "Raw Educational Notes / Syllabus Text",
            height=240,
            placeholder=(
                "Paste unstructured notes here. Example:\n"
                "Binary Search Trees (BST) are binary tree data structures where each node has at most two children.\n"
                "Left subtree contains only keys less than the node's key.\n"
                "Right subtree contains only keys greater than the node's key.\n"
                "Operations include Search, Insert, and Delete with O(h) complexity."
            ),
        )
        submit_notes = st.form_submit_button("Generate & Validate Curriculum Module", type="primary")

    if submit_notes:
        if not raw_notes.strip():
            st.error("Please provide non-empty educational notes.")
            return

        with st.spinner("Extracting concepts, questions, and pedagogical templates via LLM layer..."):
            try:
                out_path = add_module_from_notes(
                    notes=raw_notes,
                    title=module_title.strip() if module_title else None,
                    provider=provider,
                )
                st.success(f"Curriculum module successfully generated and validated at: `{out_path}`")

                # Reload curriculum in cache
                st.cache_resource.clear()
                st.info("Curriculum repository reloaded. You can now select this module from the sidebar!")
            except Exception as exc:
                st.error(f"Failed to generate curriculum module: {exc}")


# ============================================================================
# --- Script Entry Point ---
# ============================================================================

if __name__ == "__main__":
    main()
