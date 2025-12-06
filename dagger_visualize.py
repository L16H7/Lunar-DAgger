import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from torch.distributions import Normal
from dagger import Learner


def main():
    # Create environment with human render mode
    env = gym.make(
        "LunarLander-v3",
        continuous=True,
        gravity=-10.0,
        enable_wind=True,
        wind_power=15.0,
        turbulence_power=1.5,
        render_mode="human",
    )

    # Create agent
    agent = Learner(
        state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0]  # type: ignore
    )

    # Load the best model
    checkpoint = torch.load(
        "dagger_checkpoints/best_model.pth", map_location=torch.device("cpu")
    )
    agent.load_state_dict(checkpoint["actor"])

    # Run visualization
    obs, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0.0

    while not (done or truncated):
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action = agent(obs_tensor)
            action = action.cpu().numpy()[0]

        obs, reward, done, truncated, _ = env.step(action)
        total_reward += float(reward)
        env.render()  # Render the environment

    print(f"Episode finished with total reward: {total_reward}")
    env.close()


if __name__ == "__main__":
    main()
