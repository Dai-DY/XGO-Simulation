from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from legged_gym.envs import LeggedRobot

class sdog(LeggedRobot):
    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add perceptive inputs if not blind we need measured height but not put into the obs
        # if self.cfg.terrain.measure_heights:
        #     heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
        #     self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
    
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.02, 0.02, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.01).unsqueeze(1)

    # rewrite reward functions
    def _reward_stand_same_pose(self):
        if not hasattr(self, "stand_same_pose_indices"):
            self.stand_same_pose_indices = {
                "hip_pos": [],
                "hip_neg": [],
                "thigh": [],
                "calf": []
            }
            for i, name in enumerate(self.dof_names):
                if "thigh" in name:
                    self.stand_same_pose_indices["thigh"].append(i)
                elif "calf" in name:
                     self.stand_same_pose_indices["calf"].append(i)
                elif "hip" in name:
                    if "fr" in name or "br" in name:
                         self.stand_same_pose_indices["hip_neg"].append(i)
                    else:
                         self.stand_same_pose_indices["hip_pos"].append(i)
        
        hip_pos = self.dof_pos[:, self.stand_same_pose_indices["hip_pos"]]
        hip_neg = self.dof_pos[:, self.stand_same_pose_indices["hip_neg"]]
        
        all_hips = torch.cat([hip_pos, -hip_neg], dim=1)
        all_thighs = self.dof_pos[:, self.stand_same_pose_indices["thigh"]]
        all_calves = self.dof_pos[:, self.stand_same_pose_indices["calf"]]
        
        var_hips = torch.var(all_hips, dim=1)
        var_thighs = torch.var(all_thighs, dim=1)
        var_calves = torch.var(all_calves, dim=1)
        
        total_var = var_hips + var_thighs + var_calves
        
        is_standing = torch.norm(self.commands[:, :2], dim=1) < 0.1
        
        return total_var * is_standing

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        
        # update last feet air time
        self.last_feet_air_time = torch.where(first_contact, self.feet_air_time, self.last_feet_air_time)
        
        # update feet contact time
        first_air = (self.feet_contact_time > 0.) * ~contact_filt
        self.feet_contact_time += self.dt
        self.last_feet_contact_time = torch.where(first_air, self.feet_contact_time, self.last_feet_contact_time)
        self.feet_contact_time *= contact_filt
        
        # Reduced threshold from 0.5 to 0.2 for small robot
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.01)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/0.2)
        
    def _reward_foot_clearance(self):
        # Reward foot height during swing phase
        target_height = -0.06
        tanh_mult = 2.0

        # Get foot positions and velocities in world frame
        foot_pos = self.rigid_body_states[:, self.feet_indices, 0:3]
        foot_vel = self.rigid_body_states[:, self.feet_indices, 7:10]
        
        # Translate to root frame
        foot_pos_rel = foot_pos - self.root_states[:, 0:3].unsqueeze(1)
        foot_vel_rel = foot_vel - self.root_states[:, 7:10].unsqueeze(1)
        
        # Transform to body frame
        num_feet = len(self.feet_indices)
        flat_foot_pos_rel = foot_pos_rel.view(-1, 3)
        flat_foot_vel_rel = foot_vel_rel.view(-1, 3)
        flat_base_quat = self.base_quat.repeat_interleave(num_feet, dim=0)
        
        foot_pos_b = quat_rotate_inverse(flat_base_quat, flat_foot_pos_rel).view(self.num_envs, num_feet, 3)
        foot_vel_b = quat_rotate_inverse(flat_base_quat, flat_foot_vel_rel).view(self.num_envs, num_feet, 3)
        
        foot_z_error = torch.square(foot_pos_b[:, :, 2] - target_height)
        foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(foot_vel_b[:, :, :2], dim=2))
        
        reward = torch.sum(foot_z_error * foot_velocity_tanh, dim=1)
        reward *= (torch.norm(self.commands[:, :2], dim=1) > 0.1)
        
        return reward
    
    def _reward_feet_air_time_variance(self):
        # Penalize variance of air time (encourage symmetry)
        reward = torch.var(torch.clip(self.last_feet_air_time, max=0.5), dim=1) + torch.var(torch.clip(self.last_feet_contact_time, max=0.5), dim=1)
        return reward
    
    def sync_helper(self, foot0, foot1):
        air_time = self.feet_air_time
        contact_time = self.feet_contact_time
        se_air = torch.clip(torch.square(air_time[:, foot0] - air_time[:, foot1]), max=0.04)
        se_contact = torch.clip(torch.square(contact_time[:, foot0] - contact_time[:, foot1]), max=0.04)
        return torch.exp(-(se_air + se_contact)/0.7)
    
    def async_helper(self, foot0, foot1):
        air_time = self.feet_air_time
        contact_time = self.feet_contact_time
        se_act0 = torch.clip(torch.square(air_time[:, foot0] - contact_time[:, foot1]), max=0.04)
        se_act1 = torch.clip(torch.square(contact_time[:, foot0] - air_time[:, foot1]), max=0.04)
        return torch.exp(-(se_act0 + se_act1)/0.7)
    
    def _reward_feet_gait(self):
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward = self.sync_helper(0, 3) * self.sync_helper(1, 2)
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward0 = self.async_helper(0, 1)
        async_reward1 = self.async_helper(0, 2)
        async_reward2 = self.async_helper(1, 3)
        async_reward3 = self.async_helper(2, 3)
        async_reward = async_reward0 * async_reward1 * async_reward2 * async_reward3
        # only enforce gait if cmd > 0
        cmd = torch.linalg.norm(self.commands[:, :2], dim=1)
        reward = torch.where(cmd > 0.05, sync_reward * async_reward, 0)
        return reward
    
    def _reward_joint_mirror(self):
        if not hasattr(self, "mirror_joint_indices"):
            mirror_joints = [
                ["fl_thigh_joint", "br_thigh_joint"],
                ["fl_calf_joint",  "br_calf_joint"],
                ["fr_thigh_joint", "bl_thigh_joint"],
                ["fr_calf_joint",  "bl_calf_joint"],
            ]
            self.mirror_joint_indices = []
            for n1, n2 in mirror_joints:
                try:
                    # Map string name to integer index using dof_names
                    idx1 = self.dof_names.index(n1)
                    idx2 = self.dof_names.index(n2)
                    self.mirror_joint_indices.append([idx1, idx2])
                except ValueError:
                    print(f"Warning: Joint pair {n1}, {n2} not found in dof_names")
        # print(self.mirror_joint_indices)
        reward = torch.zeros(self.num_envs, device=self.device)
        for idx1, idx2 in self.mirror_joint_indices:
            diff = torch.square(self.dof_pos[:, idx1] - self.dof_pos[:, idx2])
            reward += diff
        reward /= len(self.mirror_joint_indices)
        projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        upright_scale = torch.clamp(-projected_gravity[:, 2], 0.0, 0.7) / 0.7
        reward *= upright_scale
        return reward

