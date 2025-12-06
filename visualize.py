import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from torch.distributions import Normal


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
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            hidden_dim=256,
        )

        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.action_space.high - env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.action_space.high + env.action_space.low) / 2.0,
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


def main():
    # Create environment with human render mode
    env = gym.make(
        "Walker2d-v5",
        ctrl_cost_weight=1e-3,
        render_mode="human",
    )

    # Create agent
    agent = SACAgent(env)

    # Load the best model
    checkpoint = torch.load(
        "checkpoints/best_model.pth", map_location=torch.device("cpu")
    )
    agent.actor.load_state_dict(checkpoint["actor"])
    agent.eval()

    # Run visualization
    obs, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0.0

    while not (done or truncated):
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            _, _, action = agent.get_action(obs_tensor)
            action = action.cpu().numpy()[0]

        obs, reward, done, truncated, _ = env.step(action)
        total_reward += float(reward)
        env.render()  # Render the environment

    print(f"Episode finished with total reward: {total_reward}")
    env.close()


if __name__ == "__main__":
    main()
