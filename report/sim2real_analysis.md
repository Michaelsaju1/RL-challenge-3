# Sim2Real Transfer Analysis: Humanoid Locomotion

## 1. Overview

This analysis examines the challenges of transferring a simulation-trained walking policy to real hardware, using a domain randomization ablation study as the primary experimental framework.

**Platform**: NVIDIA Isaac Gym Preview 4 (GPU-accelerated PhysX)
**Task**: Humanoid locomotion (walking/running)
**Algorithm**: PPO via rl_games, 1024 parallel environments
**Experiment**: 3 domain randomization levels, 200 epochs each

## 2. The Sim2Real Problem

A policy trained in simulation operates in a mathematically perfect world. Transferring it to a physical robot exposes mismatches across every aspect of the system.

### 2.1 Ground contact dynamics
PhysX models ground contact with simplified friction cones and rigid collision geometry. Real-world walking involves compliant shoe soles, surface irregularities, moisture, and material-dependent friction coefficients that change with temperature and wear. A policy trained on friction=1.0 may slip on a polished floor (friction ~0.4) or over-grip on rubber mats (friction ~1.8).

### 2.2 Actuator modeling
Simulation assumes ideal torque control: the commanded torque is applied instantly and exactly. Real servo motors introduce response latency (5-20ms command-to-motion delay), gear backlash (dead zone on direction reversal), torque limits that vary with angular velocity and temperature, and cable compliance. Walking policies exploit precise timing of push-off forces — even 10ms of additional latency can destabilize a gait.

### 2.3 Mass distribution and inertia
Simulated robots have exact, symmetric mass distributions defined by their model files. Real robots carry batteries, sensor boards, cabling, and mounting hardware that shift the center of mass. Rotational inertia differs from the model due to internal component placement. A gait optimized for the simulated mass distribution may produce unstable torques on the real system.

### 2.4 Sensor noise and latency
Simulation provides ground-truth joint angles, velocities, and body orientation at machine precision with zero latency. Real IMUs suffer from drift, vibration noise, and quantization. Joint encoders have resolution limits. The full sensor-to-action pipeline (read sensors → compute observation → run neural network → send command) introduces 5-50ms of latency depending on the compute stack. Policies trained on perfect state may be unable to balance with noisy, delayed observations.

### 2.5 Unmodeled dynamics
Simulation omits many real-world effects: air resistance at limb velocity, ground deformation (soft soil, carpet), joint flexibility beyond the modeled DOFs, electromagnetic interference on sensors, and thermal effects on motor performance.

## 3. Domain Randomization as Mitigation

Domain randomization (DR) addresses the reality gap by training across a distribution of physics parameters, so the real world appears as "just another sample" from that distribution.

### 3.1 Experimental design

| Parameter | No DR (baseline) | Moderate DR | Aggressive DR |
|-----------|-----------------|-------------|---------------|
| Friction | 1.0 (fixed) | 0.7 – 1.3 | 0.3 – 1.7 |
| Body mass | 1.0x (fixed) | 0.5x – 1.5x | 0.2x – 2.0x |
| Joint damping | 1.0x (fixed) | 0.5x – 1.5x | 0.2x – 2.5x |
| Joint stiffness | 1.0x (fixed) | 0.5x – 1.5x | 0.2x – 2.5x |
| Observation noise | None | σ = 0.002 | σ = 0.01 |
| Action noise | None | σ = 0.02 | σ = 0.08 |
| Gravity perturbation | None | σ = 0.4 | σ = 0.4 |

Moderate DR ranges represent realistic real-world deployment variation (different floor surfaces, slight hardware wear, sensor calibration drift). Aggressive DR ranges represent worst-case scenarios (unknown terrain, damaged actuators, heavy sensor degradation).

### 3.2 Results

| Condition | Reward | % of Baseline |
|-----------|--------|---------------|
| No DR | 788.24 | 100% |
| Moderate DR | 363.12 | 46% |
| Aggressive DR | 406.29 | 52% |

### 3.3 Analysis

**Finding 1: DR significantly reduces training reward.** The moderate DR policy achieved 46% of baseline reward. This does not indicate a worse walking ability — it reflects the difficulty of maintaining a high-speed gait when physics parameters change every episode. The DR policy learns a conservative, general-purpose walking strategy rather than an optimized-for-one-condition sprint.

**Finding 2: Aggressive DR does not further degrade performance.** The aggressive policy (406.29) scored slightly higher than moderate (363.12). With single-seed runs, this is within expected variance. A multi-seed study would clarify whether aggressive DR genuinely plateaus or whether this is noise. One interpretation: once DR is "hard enough," the policy converges to a similarly conservative gait regardless of the exact perturbation magnitude.

**Finding 3: The baseline policy is brittle by design.** The 788.24 reward assumes perfect conditions. If we tested this policy under moderate perturbation (friction=0.7, mass=1.3x), we would expect significant degradation — potentially falling over entirely. This cross-evaluation (train under condition A, test under condition B) is the definitive sim2real experiment, and would be the next step in a longer study.

## 4. The Robustness-Performance Tradeoff

The central insight of this study: there is an inherent tradeoff between training performance and deployment robustness.

The no-DR policy achieves the highest score because it can exploit specific physics parameters — timing push-offs to exact friction, swinging legs at rates tuned to exact inertia. These optimizations become liabilities when any parameter changes.

The DR policies sacrifice these optimizations for generality. They learn gaits that work "well enough" across a range of conditions, never exploiting any specific parameter. This is analogous to the bias-variance tradeoff in supervised learning: the DR policy has higher bias (suboptimal for any single condition) but lower variance (performs consistently across conditions).

## 5. Implications for Real Deployment

### 5.1 What DR can and cannot solve
DR effectively addresses parametric uncertainty — unknown friction, mass variation, sensor noise within expected ranges. It does not address structural modeling errors: missing degrees of freedom, fundamentally different contact geometry, or dynamics regimes outside the training distribution (e.g., a walking policy cannot handle swimming).

### 5.2 Complementary approaches
- **System identification**: measure real-world parameters and calibrate the simulation to match, reducing the gap DR needs to bridge
- **Sim2real fine-tuning**: transfer the DR-trained policy to real hardware and fine-tune with small amounts of real-world data
- **Adaptive policies**: train policies that estimate environment parameters online and adjust their behavior accordingly
- **Residual learning**: learn a correction on top of a classical controller, combining model-based reliability with learned flexibility

### 5.3 Isaac Gym considerations
Isaac Gym's GPU parallelism makes DR ablation studies practical — running 1024 environments with different randomized parameters is trivially parallel. Training 200 epochs took approximately 5 minutes per condition on an RTX 4060. This speed advantage is critical: DR multiplies the effective training time by the number of randomization samples, so a slow simulator would make DR prohibitively expensive.

## 6. Connection to the Grasping Attempt

This project also attempted a custom Franka Panda grasping task (see PROJECT_JOURNEY.md). The grasping task plateaued at a reaching-only policy despite reward engineering and curriculum learning. This failure itself illustrates a sim2real-adjacent challenge: even within simulation, high-DOF manipulation tasks require careful action space design (Cartesian vs. joint control), demonstrations, or hierarchical learning to overcome the exploration barrier. The gap between "sim works" and "sim solves the task" foreshadows the gap between "sim solves the task" and "real robot solves the task."

## 7. References

1. Tobin et al., "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World," IROS 2017.
2. OpenAI et al., "Learning Dexterous In-Hand Manipulation," arXiv 2018.
3. Makoviychuk et al., "Isaac Gym: High Performance GPU-Based Physics Simulation For Robot Learning," NeurIPS 2021.
4. Peng et al., "Sim-to-Real Transfer of Robotic Control with Dynamics Randomization," ICRA 2018.
5. Hwangbo et al., "Learning Agile and Dynamic Motor Skills for Legged Robots," Science Robotics 2019.
