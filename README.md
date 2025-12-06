# Lunar-DAgger — SAC training on LunarLander

This repository contains a small implementation of Soft Actor-Critic (SAC) used to train a continuous LunarLander environment (based on Gymnasium). The training script logs metrics to Weights & Biases and writes checkpoints into the `checkpoints/` folder.

---

## 📌 Results

Below is an example training run output (rewards over time) captured during one of the experiments in this repo.

![Rewards plot](rewards.png)

---

## ⚙️ Hyperparameters for the provided run

These are the exact parameters you provided that were used to produce the example run shown above:

- alpha: 0.2 (entropy regularization)
- batch_size: 256
- buffer_size: 1,000,000
- checkpoint_dir: "checkpoints"
- eval_interval: 1,000
- gamma: 0.99 (discount factor)
- learning_starts: 1,000
- log_interval: 40
- num_envs: 64
- policy_update_freq: 2
- seed: 42
- target_update_freq: 1
- tau: 0.005

---

## 🚀 Quick start — reproduce a training run

1. Create a Python virtual environment and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Train with the same hyperparameters used for the plot (adjust total timesteps as you see fit):

```bash
python sac.py \
  --num-envs 64 \
  --batch-size 256 \
  --buffer-size 1000000 \
  --alpha 0.2 \
  --gamma 0.99 \
  --learning-starts 1000 \
  --log-interval 40 \
  --eval-interval 1000 \
  --policy-update-freq 2 \
  --target-update-freq 1 \
  --tau 0.005 \
  --checkpoint-dir checkpoints \
  --seed 42 \
  --total-timesteps 5000000
```