# Architecture Decisions & Technical Rationale

## Project Goal

Train a robotic arm to grasp and lift objects in simulation using reinforcement learning, then analyze the challenges of transferring that learned behavior to a real robot (sim2real transfer). The deliverable is a trained policy, a domain randomization ablation study, and a written analysis of the reality gap.

## Decision 1: Isaac Gym over PyBullet / MuJoCo / Gazebo

**Context:** We need a physics simulator that supports reinforcement learning with enough speed to run meaningful experiments in one week.

**Options considered:**
- **PyBullet** — Free, easy to install, CPU-based. ~1 environment at a time.
- **MuJoCo** — Best contact physics, now free. CPU-based, single environment.
- **Gazebo/Ignition** — ROS ecosystem standard. Excellent sensor simulation. CPU-based, designed for system integration testing, not RL training.
- **Isaac Gym** — NVIDIA GPU-accelerated physics. Runs thousands of environments in parallel on the GPU.

**Decision:** Isaac Gym Preview 4.

**Rationale:** PPO requires millions of environment steps to learn manipulation tasks. At 1 environment on CPU (PyBullet/MuJoCo), training takes hours to days. Isaac Gym runs 512+ environments simultaneously on the GPU, reducing training from hours to minutes. This speed is essential for running a 9-condition ablation study (3 DR levels × 3 seeds) within a week. Gazebo was rejected because its strengths (ROS integration, sensor plugins) are irrelevant for state-based RL training, and its CPU physics would be prohibitively slow.

**Tradeoff accepted:** Isaac Gym is deprecated in favor of Isaac Lab/Sim. It runs on Linux only (we use WSL2). The viewer doesn't work reliably in WSL2 (we train headless). PhysX contact models are less accurate than MuJoCo for fine manipulation. These are acceptable for a research project focused on RL training and sim2real analysis.

## Decision 2: Franka Panda + YCB Mug over Simple Primitives

**Context:** We need a robot and an object for the grasping task.

**Decision:** Franka Emika Panda arm grasping a YCB 025_mug.

**Rationale:** The Franka Panda is the most widely used robot arm in manipulation research. Its URDF ships with Isaac Gym, so no custom modeling is required. The YCB (Yale-CMU-Berkeley) object set is the standard benchmark for grasping research — using a YCB object rather than a simple cube makes the project more credible and aligns with published work. The mug specifically has interesting geometry (handle vs body) that creates varied grasp strategies.

**Alternative rejected:** A basic cube would be easier to grasp but less interesting. Custom objects (e.g., weapon models) would require creating URDF files with collision meshes, adding a day of work with no scientific value.

## Decision 3: State-Based Observations over Vision-Based

**Context:** The RL policy needs input observations to decide what actions to take.

**Options:**
- **State-based (23D vector):** Joint positions, joint velocities, hand-to-object distance, object height, gripper width. Perfect information from the physics engine.
- **Vision-based (RGB/depth images):** Camera mounted on the robot, raw pixels as input. Requires CNN architecture, much harder to train.

**Decision:** State-based observations.

**Rationale:** Vision-based policies are more realistic (real robots use cameras) but require 10-100x more training time due to the high-dimensional input. For a one-week project, state-based observations let us focus on the grasping behavior and sim2real analysis rather than spending all our time getting a CNN to converge. The sim2real discussion explicitly addresses this as the "observation gap" — noting that real deployment would require a perception pipeline that introduces noise, latency, and occlusion.

## Decision 4: PPO (Proximal Policy Optimization)

**Context:** We need an RL algorithm to train the grasping policy.

**Decision:** PPO via the rl_games library (bundled with IsaacGymEnvs).

**Rationale:** PPO is the default algorithm for robotics RL due to its stability. It clips policy updates to prevent catastrophic divergence, which is critical for high-dimensional continuous control (9-DOF action space). The rl_games implementation is optimized for Isaac Gym's GPU tensor pipeline — observations and actions stay on the GPU throughout training, avoiding CPU-GPU data transfer overhead. SAC (Soft Actor-Critic) was considered but is not included in the IsaacGymEnvs training infrastructure.

**Network architecture:** 3-layer MLP (256 → 128 → 64 neurons) with ELU activation. This is standard for state-based manipulation tasks. No CNN or RNN layers needed since observations are a fixed-size vector with no temporal dependencies.

## Decision 5: Shaped Reward over Sparse Reward

**Context:** The reward function defines what "success" means and guides learning.

**Options:**
- **Sparse reward:** +1 when the mug is lifted above 10cm, 0 otherwise.
- **Shaped reward:** Continuous feedback guiding the arm through reach → grasp → lift stages.

**Decision:** Shaped reward with four components.

**Rationale:** Sparse rewards are elegant but extremely difficult to learn from in manipulation tasks. The robot must randomly stumble upon the exact sequence (move to mug → close gripper → lift) to receive any signal. With 9 DOFs and a 150-step episode, the probability of this happening by chance is essentially zero. Shaped rewards provide continuous gradient information:

1. **Distance reward** (weight: 2.0) — Exponential decay based on hand-to-mug distance. Guides the arm toward the mug.
2. **Grasp reward** (weight: 5.0) — Bonus when the gripper is close to the mug AND closing. Encourages contact.
3. **Lift reward** (weight: 15.0) — Proportional to mug height above the table, only when near the mug. Heaviest weight because this is the actual objective.
4. **Action penalty** (weight: 0.01) — Small penalty on action magnitude for smooth, energy-efficient motion.

The weights were chosen so that the lift reward dominates once the arm reaches the mug, creating a natural curriculum: first learn to approach, then learn to grasp, then learn to lift.

## Decision 6: Domain Randomization for Sim2Real Analysis

**Context:** The core intellectual contribution of the project — analyzing how simulation-trained policies transfer to reality.

**Decision:** Three-level domain randomization ablation (none / moderate / aggressive) across three random seeds.

**Parameters randomized:**
- **Object friction** (moderate: ±20%, aggressive: ±50%) — Real-world friction varies with material, surface condition, humidity, and wear.
- **Action noise** (moderate: σ=0.01, aggressive: σ=0.05) — Real actuators have latency, backlash, and imprecise tracking.
- **Observation noise** (moderate: σ=0.005, aggressive: σ=0.02) — Real sensors have quantization noise, calibration error, and latency.

**Rationale:** Domain randomization is the most widely used technique for sim2real transfer. By training under varied physics parameters, the policy learns to be robust to perturbation. The ablation study quantifies the tradeoff: no DR gives the best in-simulation performance but is brittle to any parameter change; aggressive DR produces robust policies but may converge to overly conservative behavior. This directly answers the assignment's requirement to "discuss challenges with sim2real transfer."

## Decision 7: WSL2 over Native Linux / Docker

**Context:** Isaac Gym requires Linux. Development machine runs Windows.

**Decision:** WSL2 with Ubuntu 20.04.

**Rationale:** WSL2 provides GPU passthrough to Linux via the Windows NVIDIA driver — `nvidia-smi` works natively without additional configuration. Docker with NVIDIA Container Toolkit was considered but adds complexity (driver mapping, volume mounts, GPU device configuration) that is unnecessary for a single-user development setup. Native dual-boot Linux was rejected as too disruptive for a one-week project.

**Limitation accepted:** The Isaac Gym viewer does not reliably render in WSL2. All training runs headless. Video capture (`capture_video=True`) is used for visual verification of learned behavior.

## Decision 8: uv over pip / conda

**Context:** Python package management for the project environment.

**Decision:** uv (astral-sh/uv) for all package installation.

**Rationale:** uv resolves and installs packages 10-100x faster than pip. It handles virtual environment creation, Python version management, and dependency resolution in a single tool. Isaac Gym's `setup.py` install works identically with `uv pip install -e .` as with `pip install -e .`, with no compatibility issues.

## Key Technical Challenges Encountered

### Episode Termination Communication
**Problem:** rl_games (the PPO library) was not detecting completed episodes, resulting in no reward logging. Training appeared to run but produced no learning signal.

**Root cause:** The IsaacGymEnvs base class (`VecTask`) has its own reset mechanism via `reset_done()`. Our environment was also manually calling `reset_idx()` inside `post_physics_step()`, which cleared `reset_buf` and `progress_buf` before the base class could read them. The base class never saw `reset_buf = 1` because we had already zeroed it.

**Fix:** Removed the manual `reset_idx()` call from `post_physics_step()`. The environment only sets `reset_buf = 1` when episodes end; the base class reads this flag and handles the actual reset through its own `reset_done()` method.

### Numpy Compatibility
**Problem:** Isaac Gym's code uses deprecated `np.float` alias, removed in numpy 1.24+.

**Fix:** Pinned numpy < 1.24 via `uv pip install "numpy<1.24"`.

### Batch Size Math
**Problem:** rl_games requires `batch_size` (num_envs × horizon_length) to be divisible by `minibatch_size`. Different environment counts require different minibatch sizes.

**Fix:** Set `minibatch_size: 7680` which divides evenly into 512 × 150 = 76800.
