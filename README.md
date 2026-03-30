# Sim2Real Transfer in Robotic RL: From Grasping to Locomotion

A reinforcement learning project exploring sim-to-real transfer challenges using NVIDIA Isaac Gym. The project documents a complete engineering journey: a custom Franka Panda grasping environment, iterative reward engineering and curriculum learning, and a pivot to humanoid locomotion with a domain randomization ablation study.

## Project overview

This project tackles the assignment: *"Use a simulated embodiment platform. Perform a task (grasping, navigation, balance). Discuss challenges with sim2real transfer."*

The repository contains two bodies of work:

**Part 1 — Franka mug grasping (custom environment).** A from-scratch RL environment training a Franka Panda arm to grasp a YCB mug. The grasping task was not fully solved despite multiple approaches (standard PPO, reward shaping, curriculum learning), providing firsthand experience with the exploration challenges in high-DOF manipulation.

**Part 2 — Humanoid locomotion (domain randomization ablation).** Using Isaac Gym's built-in Humanoid task, a three-condition ablation study demonstrates the core sim2real tradeoff: training performance vs. deployment robustness under domain randomization.

## Results

| Condition | DR Level | Reward | Interpretation |
|-----------|----------|--------|----------------|
| Baseline | None | 788.24 | Best in-simulation performance |
| Moderate | Friction ±30%, Mass ±50%, Damping ±50% | 363.12 | Conservative but robust gait |
| Aggressive | Friction ±70%, Mass ±80%, Damping ±80% | 406.29 | Robust, variance overlaps moderate |

The moderate DR policy sacrificed 54% of baseline reward to gain robustness against physics perturbation. This tradeoff is the central finding of the sim2real analysis.

## Repository structure

```
├── README.md                              # This file
├── ARCHITECTURE.md                        # Architecture decisions and rationale
├── PROJECT_JOURNEY.md                     # Detailed development narrative
├── franka_grasp.py                        # Custom grasping environment (Part 1)
├── cfg/
│   ├── task/
│   │   ├── FrankaGrasp.yaml               # Grasping task config
│   │   ├── HumanoidModerateDR.yaml        # Moderate DR config
│   │   └── HumanoidAggressiveDR.yaml      # Aggressive DR config
│   └── train/
│       ├── FrankaGraspPPO.yaml            # Grasping PPO config
│       ├── HumanoidModerateDRPPO.yaml     # Moderate DR PPO config
│       └── HumanoidAggressiveDRPPO.yaml   # Aggressive DR PPO config
└── report/
    └── sim2real_analysis.md               # Sim2Real transfer analysis
```

## Setup and reproduction

### Prerequisites
- Windows with WSL2 (Ubuntu 20.04) or native Ubuntu
- NVIDIA GPU (RTX 3060+) with driver >= 470
- NVIDIA Developer account for Isaac Gym download

### Installation

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

# Download Isaac Gym Preview 4 from:
# https://developer.nvidia.com/isaac-gym/download

# Extract and install
cd ~
tar -xf IsaacGym_Preview_4_Package.tar.gz
uv python install 3.8
cd ~/isaacgym/python
uv venv --python 3.8 ~/.venvs/isaacgym
source ~/.venvs/isaacgym/bin/activate
uv pip install -e .
uv pip install "numpy<1.24"

# Install IsaacGymEnvs
cd ~
git clone https://github.com/isaac-sim/IsaacGymEnvs.git
cd IsaacGymEnvs
uv pip install -e .

# WSL2 only: set CUDA library path
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

### Reproducing the ablation study

```bash
cd ~/IsaacGymEnvs/isaacgymenvs

# Copy DR configs into IsaacGymEnvs
cp cfg/task/HumanoidModerateDR.yaml ~/IsaacGymEnvs/isaacgymenvs/cfg/task/
cp cfg/task/HumanoidAggressiveDR.yaml ~/IsaacGymEnvs/isaacgymenvs/cfg/task/
cp cfg/train/HumanoidModerateDRPPO.yaml ~/IsaacGymEnvs/isaacgymenvs/cfg/train/
cp cfg/train/HumanoidAggressiveDRPPO.yaml ~/IsaacGymEnvs/isaacgymenvs/cfg/train/

# Register tasks in tasks/__init__.py:
# from .humanoid import Humanoid as HumanoidModerateDR
# from .humanoid import Humanoid as HumanoidAggressiveDR
# Add to isaacgym_task_map:
# "HumanoidModerateDR": HumanoidModerateDR,
# "HumanoidAggressiveDR": HumanoidAggressiveDR,

# Condition 1: No DR
python train.py task=Humanoid num_envs=1024 headless=True max_iterations=200

# Condition 2: Moderate DR
python train.py task=HumanoidModerateDR num_envs=1024 headless=True max_iterations=200

# Condition 3: Aggressive DR
python train.py task=HumanoidAggressiveDR num_envs=1024 headless=True max_iterations=200
```

## Platform

- **Simulator:** NVIDIA Isaac Gym Preview 4 (GPU-accelerated PhysX)
- **Algorithm:** PPO via rl_games
- **Hardware:** RTX 4060 (8GB VRAM), 1024 parallel environments
- **OS:** Windows 11 + WSL2 (Ubuntu 20.04)
- **Package manager:** uv (astral-sh/uv)
