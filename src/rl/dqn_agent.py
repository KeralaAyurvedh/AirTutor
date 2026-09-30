"""Deep Q-Network (DQN) Pedagogical Agent for AirTutor.

Implements an end-to-end lightweight PyTorch MLP Q-network with dual-network
architecture (policy_net & target_net), experience replay, Polyak soft updates,
and epsilon-greedy exploration.

Target execution environment: CPU-only, 8 GB RAM target.
Total parameter footprint: ~5,445 parameters (< 25 KB).
"""

from collections import deque
from pathlib import Path
import random
from typing import Any, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.environment.actions import ACTIONS, NUM_ACTIONS, PedagogicalAction
from src.teachers.base_teacher import BaseTeacher, TeacherObservation


class QNetwork(nn.Module):
    """Lightweight 2-hidden-layer MLP for Q-value estimation."""

    def __init__(self, state_dim: int = 14, num_actions: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """High-performance preallocated circular buffer for experience transitions."""

    def __init__(self, capacity: int = 10000, state_dim: int = 14):
        self.capacity = capacity
        self.state_dim = state_dim
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.idx = 0
        self.size = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.dones[self.idx] = 1.0 if done else 0.0
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int = 32) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = np.random.randint(0, self.size, size=batch_size)
        states_t = torch.from_numpy(self.states[indices])
        actions_t = torch.from_numpy(self.actions[indices])
        rewards_t = torch.from_numpy(self.rewards[indices])
        next_states_t = torch.from_numpy(self.next_states[indices])
        dones_t = torch.from_numpy(self.dones[indices])
        return states_t, actions_t, rewards_t, next_states_t, dones_t

    def __len__(self) -> int:
        return self.size


class DQNAgent(BaseTeacher):
    """Deep Q-Network agent selecting pedagogical actions from 14-D cognitive state vectors."""

    def __init__(
        self,
        state_dim: int = 14,
        num_actions: int = 5,
        lr: float = 1e-3,
        gamma: float = 0.95,
        tau: float = 0.005,
        epsilon_start: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        seed: int = 42,
        name: str = "DQN_Agent",
        device: str = "cpu",
    ):
        super().__init__(name=name)
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.device = torch.device(device)

        # Set seeds for reproducibility
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        # Policy & Target Networks
        self.policy_net = QNetwork(state_dim, num_actions).to(self.device)
        self.target_net = QNetwork(state_dim, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber Loss

        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity, state_dim=state_dim)


    def select_action(
        self,
        state: np.ndarray | torch.Tensor | TeacherObservation,
        greedy: bool = False,
    ) -> int:
        """Select action index in [0, num_actions-1] using epsilon-greedy policy."""
        # Convert state if TeacherObservation or other object
        if hasattr(state, "shape"):
            s_arr = np.asarray(state, dtype=np.float32)
        else:
            s_arr = np.array(state, dtype=np.float32)

        # Exploration branch
        if (not greedy) and (self.rng.random() < self.epsilon):
            return int(self.rng.integers(0, self.num_actions))

        # Exploitation branch
        with torch.no_grad():
            s_tensor = torch.from_numpy(s_arr).unsqueeze(0)
            q_values = self.policy_net(s_tensor)
            chosen_action = int(q_values.argmax(dim=-1).item())

        return chosen_action

    def select_pedagogical_action(
        self,
        state: np.ndarray | torch.Tensor,
        greedy: bool = False,
    ) -> PedagogicalAction:
        """Select discrete PedagogicalAction enum corresponding to Q-argmax."""
        act_idx = self.select_action(state, greedy=greedy)
        return ACTIONS[act_idx]

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store an experience tuple in the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> float | None:
        """Sample mini-batch, compute Smooth L1 loss, update policy net, and soft-update target net."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Q(s, a) from policy net
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q = r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1)[0]
            target_q = rewards + (1.0 - dones) * self.gamma * max_next_q

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        # In-place Polyak soft update: theta_target <- tau * theta_policy + (1 - tau) * theta_target
        with torch.no_grad():
            for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(pp.data, alpha=self.tau)

        return float(loss.item())


    def decay_epsilon(self) -> float:
        """Decay exploration rate epsilon by decay factor down to epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def save(self, path: str | Path) -> None:
        """Serialize policy network weights, optimizer state, and training metadata."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "policy_net_state_dict": self.policy_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "seed": self.seed,
            "state_dim": self.state_dim,
            "num_actions": self.num_actions,
            "gamma": self.gamma,
            "tau": self.tau,
        }
        torch.save(checkpoint, str(p))

    def load(self, path: str | Path) -> None:
        """Load weights from a saved checkpoint."""
        checkpoint = torch.load(str(path), map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        if "optimizer_state_dict" in checkpoint and hasattr(self, "optimizer"):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_min)
        self.policy_net.eval()
        self.target_net.eval()
