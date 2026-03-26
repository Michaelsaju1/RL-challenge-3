"""
FrankaGrasp: Franka Panda arm learns to grasp a YCB mug from a table.

This is the core environment for our grasping project. It defines:
- The scene: Franka arm + table + YCB mug
- Observations: what the robot "sees" (joint states + mug position)
- Rewards: shaped reward guiding reach -> grasp -> lift
- Resets: when episodes end and how the world resets

Observation space (23D):
  - Franka DOF positions (9: 7 arm joints + 2 gripper fingers)
  - Franka DOF velocities (9)
  - End-effector to mug vector (3: how far the hand is from the mug)
  - Mug height above table (1)
  - Gripper opening width (1)

Action space (9D):
  - Joint position targets for all 9 DOFs
"""

import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import (
    to_torch, tensor_clamp, torch_rand_float,
)
from isaacgymenvs.tasks.base.vec_task import VecTask


class FrankaGrasp(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id,
                 headless, virtual_screen_capture, force_render):

        self.cfg = cfg
        self.max_episode_length = cfg["env"]["episodeLength"]
        self.action_scale = cfg["env"]["actionScale"]

        # Reward weights
        self.dist_reward_scale = cfg["env"]["distRewardScale"]
        self.lift_reward_scale = cfg["env"]["liftRewardScale"]
        self.grasp_reward_scale = cfg["env"]["graspRewardScale"]
        self.action_penalty_scale = cfg["env"]["actionPenaltyScale"]
        self.lift_height = cfg["env"]["liftHeight"]

        # Domain randomization level: "none", "moderate", "aggressive"
        self.dr_level = cfg["env"].get("drLevel", "none")
        self.dr_cfg = cfg["env"].get("domainRandomization", {})

        # 23 observations, 9 actions
        self.cfg["env"]["numObservations"] = 23
        self.cfg["env"]["numActions"] = 9
        self.num_franka_dofs = 9
        # Call parent constructor (creates sim, envs, buffers)
        super().__init__(
            config=self.cfg, rl_device=rl_device, sim_device=sim_device,
            graphics_device_id=graphics_device_id, headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        # Acquire physics state tensors from Isaac Gym
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # Wrap as PyTorch tensors (these update in-place when we refresh)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.root_state = gymtorch.wrap_tensor(actor_root_state)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_tensor)

        # Franka has 9 DOFs: 7 arm joints + 2 gripper fingers
        dof_per_env = self.dof_state.shape[0] // self.num_envs
        self.franka_dof_pos = self.dof_state[:, 0].view(self.num_envs, dof_per_env)[:, :self.num_franka_dofs]
        self.franka_dof_vel = self.dof_state[:, 1].view(self.num_envs, dof_per_env)[:, :self.num_franka_dofs]

        # Default arm pose: natural pre-grasp position
        self.franka_default_dof_pos = to_torch(
            [0.0, 0.2, 0.0, -1.2, 0.0, 1.5, 0.785, 0.04, 0.04],
        )

        # Track successes
        self.successes = torch.zeros(self.num_envs, device=self.device)

    def create_sim(self):
        """Set up the simulation world: gravity, ground, environments."""
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81

        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id,
            self.physics_engine, self.sim_params,
        )
        self._create_ground_plane()
        self._create_envs(
            self.num_envs, self.cfg["env"]["envSpacing"],
            int(np.sqrt(self.num_envs)),
        )

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # --- Load Franka robot ---
        asset_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../assets",
        )
        franka_file = "urdf/franka_description/robots/franka_panda.urdf"
        franka_opts = gymapi.AssetOptions()
        franka_opts.armature = 0.01
        franka_opts.fix_base_link = True
        franka_opts.disable_gravity = True
        franka_opts.flip_visual_attachments = True
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_file, franka_opts)

        # --- Create table (simple box) ---
        table_dims = gymapi.Vec3(0.6, 0.8, 0.4)
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, table_opts)

        # --- Load YCB mug ---
        mug_file = "urdf/ycb/025_mug/025_mug.urdf"
        mug_opts = gymapi.AssetOptions()
        mug_opts.density = 500.0
        mug_opts.override_com = True
        mug_opts.override_inertia = True
        mug_opts.vhacd_enabled = True
        mug_opts.vhacd_params.resolution = 300000
        mug_asset = self.gym.load_asset(self.sim, asset_root, mug_file, mug_opts)

        # --- Franka DOF properties ---
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        franka_dof_lower = []
        franka_dof_upper = []
        for i in range(self.num_franka_dofs):
            franka_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            franka_dof_props["stiffness"][i] = 400.0 if i < 7 else 800.0
            franka_dof_props["damping"][i] = 40.0
            franka_dof_lower.append(franka_dof_props["lower"][i])
            franka_dof_upper.append(franka_dof_props["upper"][i])

        self.franka_dof_lower_limits = to_torch(franka_dof_lower, device=self.device)
        self.franka_dof_upper_limits = to_torch(franka_dof_upper, device=self.device)

        # --- Starting poses ---
        franka_pose = gymapi.Transform()
        franka_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.5, 0.0, 0.2)  # in front of robot, half-height

        mug_pose = gymapi.Transform()
        mug_pose.p = gymapi.Vec3(0.5, 0.0, 0.44)  # on top of table

        # --- Create all environments ---
        self.envs = []
        self.franka_handles = []

        for i in range(num_envs):
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env)

            # Actor 0: Franka
            franka_handle = self.gym.create_actor(env, franka_asset, franka_pose, "franka", i, 1, 0)
            self.gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)
            self.franka_handles.append(franka_handle)

            # Actor 1: Table
            table_handle = self.gym.create_actor(env, table_asset, table_pose, "table", i, 1, 0)
            # Make table gray
            table_color = gymapi.Vec3(0.5, 0.5, 0.5)
            self.gym.set_rigid_body_color(env, table_handle, 0, gymapi.MESH_VISUAL, table_color)

            # Actor 2: Mug
            mug_handle = self.gym.create_actor(env, mug_asset, mug_pose, "mug", i, 0, 0)

            # Apply friction domain randomization
            if self.dr_level != "none":
                mug_props = self.gym.get_actor_rigid_shape_properties(env, mug_handle)
                for p in mug_props:
                    scale = 0.2 if self.dr_level == "moderate" else 0.5
                    p.friction = 1.0 * (1.0 + np.random.uniform(-scale, scale))
                self.gym.set_actor_rigid_shape_properties(env, mug_handle, mug_props)

        # Find the rigid body index for the hand (end-effector)
        self.hand_handle = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.franka_handles[0], "panda_hand",
        )

        # Count actors per env (should be 3: franka, table, mug)
        self.actors_per_env = self.gym.get_actor_count(self.envs[0])

    def compute_observations(self):
        """Build the observation vector that the policy network sees."""
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # Hand position from rigid body states
        hand_pos = self.rigid_body_state.view(self.num_envs, -1, 13)[:, self.hand_handle, :3]

        # Mug position from root state (actor index 2 in each env)
        mug_pos = self.root_state.view(self.num_envs, self.actors_per_env, 13)[:, 2, :3]

        # Distance from hand to mug
        eef_to_mug = mug_pos - hand_pos

        # Mug height above table surface (table top is at z=0.4)
        mug_height = mug_pos[:, 2:3] - 0.4

        # Gripper opening
        gripper_open = self.franka_dof_pos[:, 7:8] + self.franka_dof_pos[:, 8:9]

        # Add observation noise for domain randomization
        obs = torch.cat([
            self.franka_dof_pos,   # 9
            self.franka_dof_vel,   # 9
            eef_to_mug,            # 3
            mug_height,            # 1
            gripper_open,          # 1
        ], dim=-1)                 # Total: 23

        if self.dr_level != "none":
            noise_std = 0.005 if self.dr_level == "moderate" else 0.02
            obs += torch.randn_like(obs) * noise_std

        self.obs_buf[:] = obs
        return self.obs_buf

    def pre_physics_step(self, actions):
        """Convert policy actions to joint position targets."""
        self.actions = actions.clone().to(self.device)

        # Add action noise for domain randomization
        if self.dr_level != "none":
            noise_std = 0.01 if self.dr_level == "moderate" else 0.05
            self.actions += torch.randn_like(self.actions) * noise_std

        # Scale actions and add to default pose
        targets = self.franka_default_dof_pos + self.action_scale * self.actions
        targets = tensor_clamp(targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)

        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(targets),
        )

    def post_physics_step(self):
        """Called after physics step. Compute obs, rewards, resets."""
        self.progress_buf += 1
        self.compute_observations()
        self.compute_reward()

    def compute_reward(self):
        """Shaped reward: reach -> grasp -> lift."""
        self.gym.refresh_actor_root_state_tensor(self.sim)

        # Get current mug position directly (not from obs_buf which may have noise)
        mug_pos = self.root_state.view(self.num_envs, self.actors_per_env, 13)[:, 2, :3]
        hand_pos = self.rigid_body_state.view(self.num_envs, -1, 13)[:, self.hand_handle, :3]

        eef_to_mug = mug_pos - hand_pos
        dist = torch.norm(eef_to_mug, dim=-1)
        mug_height = mug_pos[:, 2] - 0.4
        gripper_open = self.franka_dof_pos[:, 7] + self.franka_dof_pos[:, 8]

        # 1) Distance reward: steep exponential — really pull toward the mug
        dist_reward = self.dist_reward_scale * (1.0 / (1.0 + dist))

        # 2) Reaching bonus: extra reward for getting very close
        close_bonus = torch.where(dist < 0.1, 0.5 * torch.ones_like(dist), torch.zeros_like(dist))
        very_close_bonus = torch.where(dist < 0.05, 1.0 * torch.ones_like(dist), torch.zeros_like(dist))

        # 3) Grasp reward: close gripper when near mug
        near_mug = (dist < 0.05).float()
        fingers_closing = (gripper_open < 0.06).float()
        grasp_reward = self.grasp_reward_scale * near_mug * fingers_closing

        # 4) Lift reward: raise the mug off the table
        height = mug_height.clamp(min=0.0)
        lift_reward = self.lift_reward_scale * height * near_mug

        # 5) Action penalty
        action_penalty = self.action_penalty_scale * torch.sum(self.actions ** 2, dim=-1)

        # 6) Success bonus
        success = ((height > self.lift_height) & (dist < 0.05)).float()
        success_bonus = 10.0 * success

        # Total reward
        self.rew_buf[:] = dist_reward + close_bonus + very_close_bonus + grasp_reward + lift_reward - action_penalty + success_bonus

        # Reset if episode is done
        self.reset_buf[:] = torch.where(
            self.progress_buf >= self.max_episode_length - 1,
            torch.ones_like(self.reset_buf),
            torch.zeros_like(self.reset_buf),
        )

    def reset_idx(self, env_ids):
        """Reset selected environments to initial state."""
        num_resets = len(env_ids)

        # Randomize starting joint positions slightly
        pos_noise = torch_rand_float(-0.1, 0.1, (num_resets, self.num_franka_dofs), device=self.device)
        dof_pos = self.franka_default_dof_pos + pos_noise * 0.25
        dof_pos = tensor_clamp(dof_pos, self.franka_dof_lower_limits, self.franka_dof_upper_limits)

        # Write new DOF states
        dof_per_env = self.dof_state.shape[0] // self.num_envs
        for i, env_id in enumerate(env_ids):
            start = env_id * dof_per_env
            self.dof_state[start:start + self.num_franka_dofs, 0] = dof_pos[i]
            self.dof_state[start:start + self.num_franka_dofs, 1] = 0.0  # zero velocity

        # Randomize mug position on table
        mug_noise = torch_rand_float(-0.08, 0.08, (num_resets, 2), device=self.device)
        for i, env_id in enumerate(env_ids):
            state_idx = env_id * self.actors_per_env + 2  # mug is actor 2
            self.root_state[state_idx, 0] = 0.5 + mug_noise[i, 0]   # x
            self.root_state[state_idx, 1] = 0.0 + mug_noise[i, 1]   # y
            self.root_state[state_idx, 2] = 0.44                      # z (on table)
            self.root_state[state_idx, 3:7] = torch.tensor([0, 0, 0, 1], device=self.device, dtype=torch.float)
            self.root_state[state_idx, 7:13] = 0  # zero velocities

        # Apply the state changes
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # Reset DOFs
        franka_indices = (env_ids * self.actors_per_env).to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(franka_indices), len(env_ids),
        )

        # Reset mug positions
        mug_indices = (env_ids * self.actors_per_env + 2).to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_state),
            gymtorch.unwrap_tensor(mug_indices), len(env_ids),
        )

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
