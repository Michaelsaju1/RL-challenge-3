# Franka Panda Mug Grasping — RL with Domain Randomization

GPU-accelerated reinforcement learning for robotic grasping using NVIDIA Isaac Gym. A Franka Panda robot arm learns to reach, grasp, and lift a YCB mug from a table using PPO, with a domain randomization ablation study analyzing sim-to-real transfer challenges.

## Project Structure

```
├── franka_grasp.py              # Custom Isaac Gym environment (Franka + table + YCB mug)
├── cfg/
│   ├── task/FrankaGrasp.yaml    # Environment config (physics, rewards, DR parameters)
│   └── train/FrankaGraspPPO.yaml # PPO training config (network, hyperparameters)
├── run_ablation.sh              # Runs 3 DR levels × 3 seeds
├── evaluate.py                  # Policy evaluation + metrics collection
├── plot_results.py              # Generates comparison plots
└── report/
    └── sim2real_analysis.md     # Sim2Real transfer analysis writeup
```

## Setup

### Prerequisites
- Windows with WSL2 (Ubuntu 20.04) or native Ubuntu 20.04/22.04
- NVIDIA GPU (RTX 3060+) with driver >= 470
- NVIDIA Developer account (free) for Isaac Gym download

### Installation

```bash
# 1. Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

# 2. Download Isaac Gym Preview 4 from:
#    https://developer.nvidia.com/isaac-gym/download
#    Place IsaacGym_Preview_4_Package.tar.gz in your home directory

# 3. Extract Isaac Gym
cd ~
tar -xf IsaacGym_Preview_4_Package.tar.gz

# 4. Create Python 3.8 environment and install Isaac Gym
uv python install 3.8
cd ~/isaacgym/python
uv venv --python 3.8 ~/.venvs/isaacgym
source ~/.venvs/isaacgym/bin/activate
uv pip install -e .
uv pip install "numpy<1.24"

# 5. Install IsaacGymEnvs
cd ~
git clone https://github.com/isaac-sim/IsaacGymEnvs.git
cd IsaacGymEnvs
uv pip install -e .

# 6. For WSL2: set CUDA library path
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

# 7. Verify installation
cd isaacgymenvs
python train.py task=Cartpole num_envs=1024 headless=True
# Should show reward climbing toward ~494 within 100 epochs
```

### Installing the Custom Task

```bash
# Copy task files into IsaacGymEnvs
cp franka_grasp.py ~/IsaacGymEnvs/isaacgymenvs/tasks/
cp cfg/task/FrankaGrasp.yaml ~/IsaacGymEnvs/isaacgymenvs/cfg/task/
cp cfg/train/FrankaGraspPPO.yaml ~/IsaacGymEnvs/isaacgymenvs/cfg/train/

# Register the task — edit ~/IsaacGymEnvs/isaacgymenvs/tasks/__init__.py:
# Add import:   from .franka_grasp import FrankaGrasp
# Add to dict:  "FrankaGrasp": FrankaGrasp,
```

## Training

```bash
cd ~/IsaacGymEnvs/isaacgymenvs

# Basic training (no domain randomization)
python train.py task=FrankaGrasp num_envs=512 headless=True \
    max_iterations=500 train.params.config.minibatch_size=7680

# With moderate domain randomization
python train.py task=FrankaGrasp num_envs=512 headless=True \
    max_iterations=500 train.params.config.minibatch_size=7680 \
    task.env.drLevel=moderate

# With aggressive domain randomization
python train.py task=FrankaGrasp num_envs=512 headless=True \
    max_iterations=500 train.params.config.minibatch_size=7680 \
    task.env.drLevel=aggressive
```

## Domain Randomization Ablation

The core experiment varies simulation parameters to study policy robustness:

| Parameter        | No DR (Baseline) | Moderate (±20%) | Aggressive (±50%) |
|------------------|------------------|-----------------|--------------------|
| Object friction  | 1.0              | 0.8 – 1.2      | 0.5 – 1.5         |
| Action noise σ   | 0.0              | 0.01            | 0.05               |
| Obs noise σ      | 0.0              | 0.005           | 0.02               |

## Environment Details

**Observation space (23D):**
- Franka joint positions (9): 7 arm joints + 2 gripper fingers
- Franka joint velocities (9)
- End-effector to mug vector (3): direction and distance to target
- Mug height above table (1)
- Gripper opening width (1)

**Action space (9D):**
- Joint position targets for all 9 DOFs

**Reward shaping (reach → grasp → lift):**
- Distance reward: exponential decay based on hand-to-mug distance
- Grasp reward: bonus when gripper is close to mug and closing
- Lift reward: proportional to mug height above table
- Action penalty: penalizes large joint movements for smooth motion
- Success bonus: large reward for lifting mug above 10cm threshold

**Object:** YCB 025_mug (standardized benchmark object for grasping research)

## Platform

- **Simulator:** NVIDIA Isaac Gym Preview 4 (GPU-accelerated PhysX)
- **Algorithm:** PPO via rl_games
- **Robot:** Franka Emika Panda 7-DOF arm
- **GPU:** RTX 4060 (8GB VRAM), 512 parallel environments

## Sim2Real Transfer Discussion

See `report/sim2real_analysis.md` for the full analysis covering:
- Contact dynamics mismatch (simplified friction models vs real-world surface interactions)
- Observation gap (perfect state in sim vs noisy camera perception in reality)
- Actuator modeling (ideal position control vs real servo latency and backlash)
- Domain randomization as mitigation and its limits
- Ablation study results showing the robustness-performance tradeoff
