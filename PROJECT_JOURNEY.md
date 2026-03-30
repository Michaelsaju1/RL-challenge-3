# Project Journey: From Grasping Failure to Locomotion Success

This document traces the full engineering process of this project, including failed approaches, debugging sessions, and the reasoning behind every pivot. Honest documentation of what didn't work is as valuable as what did.

## Phase 1: Platform selection and setup

### The decision
We needed a physics simulator for RL training. The options were PyBullet (CPU, simple), MuJoCo (CPU, best contact physics), Gazebo (CPU, ROS ecosystem), and Isaac Gym (GPU-accelerated, NVIDIA).

We chose Isaac Gym because PPO requires millions of environment steps. CPU simulators run one environment at a time; Isaac Gym runs 1024 simultaneously on the GPU. For a one-week project, this was the difference between running one experiment and running a full ablation study.

### The setup
Isaac Gym is Linux-only. Since the development machine runs Windows, we used WSL2 (Windows Subsystem for Linux) with Ubuntu 20.04. The RTX 4060 GPU is accessible from WSL2 via NVIDIA's driver passthrough — `nvidia-smi` works natively.

Key setup challenges:
- **libcuda.so not found**: Isaac Gym couldn't find CUDA libraries. Fixed by setting `LD_LIBRARY_PATH=/usr/lib/wsl/lib`.
- **numpy compatibility**: Isaac Gym uses deprecated `np.float`, removed in numpy 1.24+. Fixed by pinning `numpy<1.24`.
- **Batch size math**: rl_games requires `num_envs × horizon_length` to be divisible by `minibatch_size`. Different environment counts need different minibatch sizes.
- **Python version**: Isaac Gym's compiled binaries target Python 3.8. We used uv to install and manage a 3.8 virtual environment.

### Verification
Trained the built-in Cartpole task. Reward climbed from ~150 to 494 (near-perfect) in 100 epochs, confirming the full pipeline: GPU physics → observations → PPO → actions → GPU physics.

## Phase 2: Custom Franka grasping environment

### The goal
Train a Franka Panda 7-DOF robot arm to reach, grasp, and lift a YCB mug from a table. The YCB (Yale-CMU-Berkeley) object set is the standard benchmark for grasping research.

### What we built
A custom Isaac Gym environment (`franka_grasp.py`) defining:
- **Scene**: Franka arm + table + YCB 025_mug
- **Observations (23D)**: joint positions (9), joint velocities (9), hand-to-mug vector (3), mug height (1), gripper opening (1)
- **Actions (9D)**: joint position targets for all DOFs
- **Reward**: shaped reward with distance, grasp, lift, and action penalty components

### Attempt 1: Standard PPO with shaped reward

**Approach**: Distance-based reward pulling the arm toward the mug, plus bonuses for grasping and lifting.

**Result**: Reward plateaued at 1.63 after 1000 epochs. The arm learned to move vaguely in the mug's direction but never discovered grasping.

**Why it failed**: The arm starts 0.62m from the mug. With 9 DOFs, random joint-space exploration rarely produces coordinated movement toward a specific point. The arm found a local optimum: collect easy distance reward without attempting the harder grasp. The probability of randomly discovering the precise sequence (position hand → orient gripper → close fingers → lift) is essentially zero.

### Attempt 2: Aggressive reward shaping

**Approach**: Changed from `1/(1+dist)` to `exp(-10*dist)` for steeper gradient near the mug. Added proximity bonuses at 10cm and 5cm thresholds.

**Result**: Reward was actually lower (0.59) because the exponential reward gives almost zero signal at 0.62m distance. The arm couldn't even find the starting gradient.

**Lesson**: Reward functions that are too sharp near the goal but flat everywhere else create a "needle in a haystack" problem. The agent needs gradient everywhere, not just near the target.

### Attempt 3: Curriculum learning

**Approach**: Start the mug near the hand so the robot learns grasping first, then gradually move the mug further away.

**Phase 0 attempt — mug in mid-air near the hand**: Placed the mug at the hand's position (z=0.80). The mug immediately fell to the table due to gravity, ending up at the same 0.62m distance as before. The curriculum provided no benefit because the mug fell before the arm could react.

**Phase 0 attempt — mug on a raised table**: Raised the table so its surface was at the hand's height. The mug stayed stable (no falling), and starting distance dropped to 0.16m. Reward was higher (1.72) but still plateaued at 1.50 after 200 epochs.

**Why it still failed**: Even at 0.16m, the arm's default pose points the hand upward, not toward the mug. The 9-DOF joint-space exploration problem remained — the arm needed to discover a complex multi-joint coordination to reach down and forward simultaneously.

### The fundamental insight

Our grasping task failed because we used **joint-space control** (9D action space). Each action specifies a target angle for each of 9 joints. To move the hand 1cm toward the mug, the agent has to figure out the right combination of 9 joint angle changes — a complex inverse kinematics problem that PPO must solve through trial and error.

Published grasping papers (including NVIDIA's FrankaCubeStack) use **Cartesian control** or **operational space control (OSC)**, where the action space is 4D: (hand_dx, hand_dy, hand_dz, gripper). The agent says "move hand left 1cm" and an IK solver handles the joint angles. This reduces the exploration problem from 9D to 4D, making grasping tractable.

Building a Cartesian controller requires implementing inverse kinematics or operational space control — significant additional engineering beyond the scope of a one-week project.

## Phase 3: Pivot to humanoid locomotion

### The decision
With the assignment deadline approaching and grasping unsolved, we pivoted to humanoid locomotion. This was a pragmatic engineering decision: use a proven task where PPO succeeds out of the box, and focus our effort on the sim2real analysis (domain randomization ablation), which is the intellectual core of the assignment.

### Why locomotion works where grasping failed
Locomotion has a natural reward gradient. Random joint movements produce some forward stumbling, which earns reward, which PPO builds on. There's no "discovery cliff" like grasping requires. The reward signal is dense and continuous from the very first step.

### The ablation study

We ran three training conditions, each for 200 epochs with 1024 parallel environments:

**Condition 1 — No domain randomization (baseline)**
Perfect simulation: fixed friction (1.0), exact masses, no sensor noise. The humanoid learns an optimized gait for these specific conditions.
Result: reward 788.24

**Condition 2 — Moderate domain randomization**
Realistic deployment variation: friction ±30%, mass ±50%, joint damping ±50%, observation noise σ=0.002, action noise σ=0.02. Simulates the difference between lab conditions and real-world deployment.
Result: reward 363.12

**Condition 3 — Aggressive domain randomization**
Worst-case deployment: friction ±70%, mass ±80%, joint damping ±80%, observation noise σ=0.01, action noise σ=0.08. Simulates unknown terrain, damaged actuators, and heavy sensor degradation.
Result: reward 406.29

### Interpreting the results

The baseline policy (788) achieves the highest reward because it trains and evaluates in identical conditions. It has memorized the optimal gait for one specific physics setup.

The moderate DR policy (363) sacrifices 54% of baseline reward. This doesn't mean it walks worse — it means it learned a conservative, general-purpose gait that works across many different conditions instead of being optimal for one.

The aggressive DR policy (406) is surprisingly close to moderate. With single-seed runs, there's natural variance that accounts for this overlap. A rigorous study would use 3-5 seeds per condition to distinguish real effects from noise.

The key insight: **the baseline policy would likely collapse under any perturbation**, while the DR policies would maintain their performance. This robustness gap is the central challenge of sim2real transfer.

## Technical debugging log

### Episode termination (4+ hours of debugging)
The rl_games library was not detecting completed episodes, showing `-inf` reward in checkpoints. Root cause: our `post_physics_step()` called `reset_idx()` which cleared `reset_buf` to 0 before the base class (`VecTask`) could read it. The base class uses `reset_buf` to track episode boundaries and compute timeout signals.

Fix: removed the manual `reset_idx()` call from `post_physics_step()`. The base class handles resets through its own `reset_done()` method. Our environment only sets `reset_buf = 1` when episodes end.

### Mug falling in curriculum learning
During Phase 0 of curriculum learning, the mug was placed at the hand's height (z=0.80) but with no surface to support it. Gravity pulled it to the table within 30 timesteps, defeating the curriculum.

Diagnosis: wrote a diagnostic script (`dump_positions.py`) that recorded mug position over 150 steps with zero actions. Confirmed `mug_z` dropped from 0.80 to 0.44 (table height).

Fix: raised the table itself (which has `fix_base_link=True` and ignores gravity) so its surface was near the hand.

### GPU pipeline on WSL2
Isaac Gym occasionally failed to create a PhysX CUDA Context Manager, falling back to CPU. Caused by `libcuda.so` not being on the library path.

Fix: `export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH` added to `~/.bashrc` for persistence.

## What I would do differently

1. **Start with Cartesian control for manipulation**: Joint-space PPO for grasping is a known hard problem. Operational space control reduces the action space from 9D to 4D and makes exploration tractable.

2. **Use a proven baseline and modify it**: NVIDIA's FrankaCubeStack already solves grasping with carefully engineered rewards and OSC. Starting from a working solution and modifying it (swap objects, add DR) is faster and more reliable than building from scratch.

3. **Run multiple seeds**: Single-seed results have too much variance for meaningful ablation comparisons. Three seeds per condition with mean ± std would strengthen the analysis.

4. **Budget more time for reward engineering**: Reward design for manipulation is genuinely difficult. Published papers often spend weeks iterating on reward functions. A one-week timeline is tight for both building the environment and tuning the reward.

5. **Test earlier with diagnostic scripts**: The `dump_positions.py` script that recorded mug and hand positions was invaluable for debugging. Building diagnostic tools upfront would have saved hours of trial-and-error training runs.
