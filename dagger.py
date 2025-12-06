import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import wandb
from torch.optim.adam import Adam
from torch.distributions import Normal

from buffers import ReplayBuffer


def make_env():
    def _thunk():
        env = gym.make(
            "Walker2d-v5",
            ctrl_cost_weight=1e-3,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return _thunk


LOG_STD_MIN = -5
LOG_STD_MAX = 2


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats
        return mean, log_std


class ExpertAgent(nn.Module):
    def __init__(self, env):
        super(ExpertAgent, self).__init__()
        self.actor = ActorNetwork(
            state_dim=env.single_observation_space.shape[0],
            action_dim=env.single_action_space.shape[0],
            hidden_dim=256,
        )

        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def get_action(self, state):
        state = torch.FloatTensor(state)
        mean, log_std = self.actor(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class Learner(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Learner, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state):
        x = self.net(state)
        return x


def evaluate(learner, num_episodes=5):
    env = gym.make(
        "Walker2d-v5",
        ctrl_cost_weight=1e-3,
    )

    avg_returns = []
    avg_lengths = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        truncated = False
        while not (done or truncated):
            with torch.no_grad():
                action = learner(torch.FloatTensor(obs).unsqueeze(0))
                action = action.cpu().numpy()[0]

            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += float(reward)
            episode_length += 1
        avg_returns.append(episode_reward)
        avg_lengths.append(episode_length)

    return np.mean(avg_returns), np.mean(avg_lengths)


def train(args):
    wandb.init(
        project="Walker2d-DAggers", config=vars(args), entity="l16h7", group=args.group
    )

    os.makedirs("walker2d_dagger_checkpoints", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    envs = gym.vector.SyncVectorEnv([make_env() for _ in range(args.num_envs)])
    expert = ExpertAgent(envs)
    checkpoint = torch.load(
        "checkpoints/best_model.pth", map_location=torch.device("cpu")
    )
    expert.actor.load_state_dict(checkpoint["actor"])
    expert.eval()

    learner = Learner(
        state_dim=envs.single_observation_space.shape[0],  # type: ignore
        action_dim=envs.single_action_space.shape[0],  # type: ignore
    )
    optimizer = Adam(learner.parameters(), lr=args.lr)

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    obs, _ = envs.reset(seed=args.seed)

    timesteps = 0
    iterations = 0
    beta = args.beta
    best_avg_return = -float("inf")

    while timesteps < args.total_timesteps:
        iterations += 1
        for i in range(args.learner_steps):
            with torch.no_grad():
                expert_actions, _, _ = expert.get_action(obs)
            learner_actions = learner(torch.FloatTensor(obs))

            sampling_mask = np.random.rand(args.num_envs) < beta
            sampling_mask = sampling_mask.astype(np.float32).reshape(-1, 1)
            sampled_actions = np.where(
                sampling_mask,
                expert_actions.detach().cpu().numpy(),
                learner_actions.detach().cpu().numpy(),
            )

            next_obs, rewards, dones, truncs, infos = envs.step(sampled_actions)

            rb.add(obs, next_obs, expert_actions, rewards, dones, infos)  # type: ignore

            obs = next_obs

            timesteps += args.num_envs

        # Update learner
        for _ in range(args.n_epochs):
            data = rb.sample(args.batch_size)

            # Ensure data is float32 to match network dtypes
            data = data._replace(
                observations=data.observations.float(),
                actions=data.actions.float()
            )

            learner_actions = learner(data.observations)

            loss = F.mse_loss(learner_actions, data.actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        beta *= args.beta_decay
        beta = max(beta, 0)  # Ensure beta does not go below 0

        if iterations % args.log_interval == 0:
            print(
                f"Iteration: {iterations}, Timesteps: {timesteps}, Loss: {loss.item():.4f}, Beta: {beta:.4f}"
            )
            wandb.log(
                {
                    "iteration": iterations,
                    "timesteps": timesteps,
                    "loss": loss.item(),
                    "beta": beta,
                }
            )

        if iterations % args.eval_interval == 0:
            avg_return, avg_length = evaluate(learner, num_episodes=5)
            print(f"Evaluation over 5 episodes: {avg_return:.2f}")
            wandb.log(
                {
                    "avg_return": avg_return,
                    "avg_length": avg_length,
                    "timesteps": timesteps,
                }
            )

            if avg_return > best_avg_return:
                best_avg_return = avg_return
                torch.save(
                    {
                        "actor": learner.state_dict(),
                    },
                    "dagger_checkpoints/best_model.pth",
                )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--num-envs", type=int, default=4, help="Number of parallel environments"
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=1_000_000, help="Total timesteps"
    )
    parser.add_argument(
        "--learner-steps",
        type=int,
        default=64,
        help="Number of learner steps per iteration",
    )
    parser.add_argument(
        "--buffer-size", type=int, default=1_000_000, help="Replay buffer size"
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument(
        "--n-epochs", type=int, default=5, help="Number of epochs per update"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--log-interval", type=int, default=1, help="Logging interval")
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=10,
        help="Evaluation interval (in iterations)",
    )
    parser.add_argument(
        "--beta-decay",
        type=float,
        default=0.99,
        help="Decay factor for beta (expert action probability)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Initial probability of taking expert actions",
    )
    parser.add_argument(
        "--group",
        type=str,
        default="default",
        help="WandB group name for organizing runs",
    )

    args = parser.parse_args()
    train(args)
