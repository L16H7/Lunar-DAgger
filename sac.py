import pdb
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import wandb
from torch.distributions import Normal
from torch.optim.adam import Adam

from buffers import ReplayBuffer


class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(SoftQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


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


class SACAgent(nn.Module):
    def __init__(self, env):
        super(SACAgent, self).__init__()
        self.actor = ActorNetwork(
            state_dim=env.single_observation_space.shape[0],
            action_dim=env.single_action_space.shape[0],
            hidden_dim=256,
        )
        self.qf1 = SoftQNetwork(
            state_dim=env.single_observation_space.shape[0],
            action_dim=env.single_action_space.shape[0],
            hidden_dim=256,
        )
        self.qf2 = SoftQNetwork(
            state_dim=env.single_observation_space.shape[0],
            action_dim=env.single_action_space.shape[0],
            hidden_dim=256,
        )
        self.qf1_target = SoftQNetwork(
            state_dim=env.single_observation_space.shape[0],
            action_dim=env.single_action_space.shape[0],
            hidden_dim=256,
        )
        self.qf2_target = SoftQNetwork(
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
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=3e-4
        )
        self.actor_optimizer = Adam(list(self.actor.parameters()), lr=3e-4)

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


def evaluate(agent, args, num_episodes=5):
    env = gym.make(
        "LunarLander-v3",
        continuous=True,
        gravity=-10.0,
        enable_wind=True,
        wind_power=15.0,
        turbulence_power=1.5,
    )

    avg_returns = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        while not (done or truncated):
            with torch.no_grad():
                obs_tensor = (
                    torch.FloatTensor(obs)
                    .unsqueeze(0)
                    .to(next(agent.parameters()).device)
                )
                _, _, action = agent.get_action(obs_tensor)
                action = action.cpu().numpy()[0]

            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += float(reward)
        avg_returns.append(episode_reward)

    return np.mean(avg_returns)


def make_env():
    def _thunk():
        env = gym.make(
            "LunarLander-v3",
            continuous=True,
            gravity=-10.0,
            enable_wind=True,
            wind_power=15.0,
            turbulence_power=1.5,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return _thunk


def train(args):
    wandb.init(project="Lunar-DAgger", config=vars(args), entity="l16h7")

    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    best_return = -float("inf")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    envs = gym.vector.SyncVectorEnv([make_env() for i in range(args.num_envs)])

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )

    agent = SACAgent(envs)
    obs, _ = envs.reset(seed=args.seed)

    timesteps = 0
    iterations = 0
    while timesteps < args.total_timesteps:
        iterations += 1

        actions, log_prob, mean = agent.get_action(obs)
        actions = actions.detach().cpu().numpy()
        next_obs, rewards, dones, truncs, infos = envs.step(actions)

        rb.add(obs, next_obs, actions, rewards, dones, infos)  # type: ignore
        obs = next_obs

        timesteps += args.num_envs

        if rb.size() < args.learning_starts:
            continue

        data = rb.sample(batch_size=args.batch_size)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = agent.get_action(
                data.next_observations
            )
            qf1_next_target = agent.qf1_target(
                data.next_observations, next_state_action
            )
            qf2_next_target = agent.qf2_target(
                data.next_observations, next_state_action
            )
            min_qf_next_target = (
                torch.min(qf1_next_target, qf2_next_target)
                - args.alpha * next_state_log_pi
            )
            next_q_value = data.rewards + (1 - data.dones) * args.gamma * (
                min_qf_next_target
            )

        qf1_a_values = agent.qf1(data.observations, data.actions)
        qf2_a_values = agent.qf2(data.observations, data.actions)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss
        agent.q_optimizer.zero_grad()
        qf_loss.backward()
        agent.q_optimizer.step()

        if iterations % args.policy_update_freq == 0:
            pi, log_pi, _ = agent.get_action(data.observations)
            qf1_pi = agent.qf1(data.observations, pi)
            qf2_pi = agent.qf2(data.observations, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            actor_loss = (args.alpha * log_pi - min_qf_pi).mean()

            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()

            if iterations % args.log_interval == 0:
                wandb.log(
                    {
                        "losses/qf_loss": qf_loss.item(),
                        "losses/actor_loss": actor_loss.item(),
                        "losses/alpha": args.alpha,
                        "global_step": timesteps,
                    }
                )

        if iterations % args.target_update_freq == 0:
            for target_param, param in zip(
                agent.qf1_target.parameters(), agent.qf1.parameters()
            ):
                target_param.data.copy_(
                    args.tau * param.data + (1 - args.tau) * target_param.data
                )

            for target_param, param in zip(
                agent.qf2_target.parameters(), agent.qf2.parameters()
            ):
                target_param.data.copy_(
                    args.tau * param.data + (1 - args.tau) * target_param.data
                )

        if iterations % args.log_interval == 0:
            print(f"Timesteps: {timesteps}, " f"QF Loss: {qf_loss.item():.3f}, ")

        if iterations % args.eval_interval == 0:
            eval_return = evaluate(agent, args)
            print(f"Eval at {timesteps}: {eval_return}")
            wandb.log({"charts/eval_return": eval_return, "global_step": timesteps})

            if args.checkpoint_dir:
                checkpoint_path = os.path.join(
                    args.checkpoint_dir, f"checkpoint_{timesteps}.pth"
                )
                torch.save(
                    {
                        "actor": agent.actor.state_dict(),
                        "qf1": agent.qf1.state_dict(),
                        "qf2": agent.qf2.state_dict(),
                        "qf1_target": agent.qf1_target.state_dict(),
                        "qf2_target": agent.qf2_target.state_dict(),
                        "q_optimizer": agent.q_optimizer.state_dict(),
                        "actor_optimizer": agent.actor_optimizer.state_dict(),
                        "args": args,
                        "timesteps": timesteps,
                    },
                    checkpoint_path,
                )
                print(f"Saved checkpoint to {checkpoint_path}")

                if eval_return > best_return:
                    best_return = eval_return
                    best_model_path = os.path.join(
                        args.checkpoint_dir, "best_model.pth"
                    )
                    torch.save(
                        {
                            "actor": agent.actor.state_dict(),
                            "qf1": agent.qf1.state_dict(),
                            "qf2": agent.qf2.state_dict(),
                            "qf1_target": agent.qf1_target.state_dict(),
                            "qf2_target": agent.qf2_target.state_dict(),
                            "q_optimizer": agent.q_optimizer.state_dict(),
                            "actor_optimizer": agent.actor_optimizer.state_dict(),
                            "args": args,
                            "timesteps": timesteps,
                            "eval_return": eval_return,
                        },
                        best_model_path,
                    )
                    print(f"Saved best model to {best_model_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-envs", type=int, default=4, help="Number of parallel environments"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="Total training timesteps",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--buffer-size", type=int, default=1_000_000, help="Replay buffer size"
    )
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=1000,
        help="Timesteps before learning starts",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Batch size for training"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.2, help="Entropy regularization coefficient"
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--tau", type=float, default=0.005, help="Target network update rate"
    )
    parser.add_argument(
        "--policy-update-freq",
        type=int,
        default=1,
        help="Frequency of policy updates",
    )
    parser.add_argument(
        "--target-update-freq",
        type=int,
        default=5,
        help="Frequency of target network updates",
    )
    parser.add_argument("--log-interval", type=int, default=40, help="Logging interval")
    parser.add_argument(
        "--eval-interval", type=int, default=1000, help="Evaluation interval"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    args = parser.parse_args()

    train(args)
