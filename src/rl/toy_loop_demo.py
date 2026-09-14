"""Toy Q-Learning Loop Demonstration.

Stage 0 deliverable proving the mechanical RL loop end-to-end:
A hand-rolled discrete MDP (no Gymnasium) trained with tabular Q-learning,
logging progress and exporting a learning curve plot.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np


class ToyChainEnv:
    """A hand-rolled 1D chain environment (~30 lines, zero external framework).

    State space: {0, 1, ..., length - 1}.
    Actions: 0 = Move Left, 1 = Move Right.
    Terminal: reaching goal (state = length - 1) or max_steps reached.
    """

    def __init__(self, length: int = 5, max_steps: int = 20):
        self.length = length
        self.max_steps = max_steps
        self.state = 0
        self.steps = 0

    def reset(self) -> int:
        self.state = 0
        self.steps = 0
        return self.state

    def step(self, action: int) -> tuple[int, float, bool]:
        self.steps += 1
        if action == 1:
            self.state = min(self.length - 1, self.state + 1)
        else:
            self.state = max(0, self.state - 1)

        reached_goal = (self.state == self.length - 1)
        timeout = (self.steps >= self.max_steps)
        done = reached_goal or timeout

        reward = 10.0 if reached_goal else -0.1
        return self.state, reward, done


class TabularQLearningAgent:
    """Tabular Q-learning agent implementing standard Bellman temporal difference updates."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.985,
        epsilon_min: float = 0.05,
        seed: int = 42,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.rng = np.random.default_rng(seed)
        self.q_table = np.zeros((num_states, num_actions), dtype=float)

    def select_action(self, state: int) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.num_actions))
        q_vals = self.q_table[state]
        max_val = np.max(q_vals)
        best_actions = np.flatnonzero(q_vals == max_val)
        return int(self.rng.choice(best_actions))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> float:
        """Bellman update: Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]."""
        current_q = self.q_table[state, action]
        target = reward if done else reward + self.gamma * np.max(self.q_table[next_state])
        td_error = target - current_q
        self.q_table[state, action] += self.lr * td_error
        return td_error

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_toy_agent(
    num_episodes: int = 300,
    seed: int = 42,
) -> tuple[TabularQLearningAgent, list[float]]:
    """Train tabular Q-learning agent on the toy environment."""
    env = ToyChainEnv(length=5, max_steps=20)
    agent = TabularQLearningAgent(
        num_states=5,
        num_actions=2,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.985,
        epsilon_min=0.05,
        seed=seed,
    )

    episode_rewards: list[float] = []

    print(f"--- Starting Toy Q-Learning Training ({num_episodes} episodes) ---")
    for ep in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        if ep % 50 == 0 or ep == num_episodes:
            recent_avg = np.mean(episode_rewards[-50:])
            print(
                f"Episode {ep:3d}/{num_episodes} | "
                f"Recent Avg Reward (last 50): {recent_avg:6.2f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    return agent, episode_rewards


def plot_learning_curve(
    rewards: list[float],
    save_path: Path,
    window: int = 15,
) -> None:
    """Plot episode rewards and rolling average, saving to file."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4.5), dpi=150)
    episodes = np.arange(1, len(rewards) + 1)
    plt.plot(episodes, rewards, color="#90CAF9", alpha=0.5, label="Raw Episode Reward")

    if len(rewards) >= window:
        rolling_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        plt.plot(
            np.arange(window, len(rewards) + 1),
            rolling_avg,
            color="#1565C0",
            linewidth=2.0,
            label=f"{window}-Episode Moving Average",
        )

    plt.title("AirTutor Stage 0: Toy Tabular Q-Learning Learning Curve", fontsize=12, pad=10)
    plt.xlabel("Episode", fontsize=10)
    plt.ylabel("Total Reward", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()
    print(f"Learning curve plot successfully saved to: {save_path}")


def main() -> None:
    agent, rewards = train_toy_agent(num_episodes=300, seed=42)
    output_plot = Path("experiments/toy_learning_curve.png")
    plot_learning_curve(rewards, output_plot)
    print("\nFinal Learned Q-Table:")
    print("State | Q(Left)  | Q(Right)")
    print("----------------------------")
    for s in range(agent.num_states):
        print(f"  {s:2d}  | {agent.q_table[s, 0]:8.3f} | {agent.q_table[s, 1]:8.3f}")
    print("\nStage 0 RL mechanical verification complete.")


if __name__ == "__main__":
    main()
