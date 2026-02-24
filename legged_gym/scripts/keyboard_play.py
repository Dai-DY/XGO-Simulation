# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from isaacgym import gymapi
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
from isaacgym.torch_utils import get_euler_xyz
import multiprocessing as mp
import time

def plotter_process(data_queue):
    # Initialize buffers for plotting
    plot_len = 200
    time_buf = deque(maxlen=plot_len)
    vel_x_buf = deque(maxlen=plot_len)
    vel_y_buf = deque(maxlen=plot_len)
    roll_buf = deque(maxlen=plot_len)
    pitch_buf = deque(maxlen=plot_len)
    yaw_buf = deque(maxlen=plot_len)
    cmd_x_buf = deque(maxlen=plot_len)
    cmd_y_buf = deque(maxlen=plot_len)
    height_buf = deque(maxlen=plot_len)

    # Initialize Plot
    plt.ion()
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    
    line_vel_x, = axes[0].plot([], [], label='Vel X', color='b')
    line_cmd_x, = axes[0].plot([], [], label='Cmd X', color='b', linestyle='--')
    line_vel_y, = axes[0].plot([], [], label='Vel Y', color='r')
    line_cmd_y, = axes[0].plot([], [], label='Cmd Y', color='r', linestyle='--')
    axes[0].set_ylabel('Velocity (m/s)')
    axes[0].set_title('Body Velocity')
    axes[0].legend(loc='upper right')
    axes[0].grid(True)
    axes[0].set_ylim(-1.5, 1.5)

    line_roll, = axes[1].plot([], [], label='Roll', color='r')
    line_pitch, = axes[1].plot([], [], label='Pitch', color='g')
    line_yaw, = axes[1].plot([], [], label='Yaw', color='b')
    axes[1].set_ylabel('Angle (rad)')
    axes[1].set_title('Body Orientation')
    axes[1].legend(loc='upper right')
    axes[1].grid(True)
    axes[1].set_ylim(-1.5, 1.5)

    line_height, = axes[2].plot([], [], label='Base Height', color='m')
    axes[2].set_ylabel('Height (m)')
    axes[2].set_title('Base Height')
    axes[2].legend(loc='upper right')
    axes[2].grid(True)
    axes[2].set_xlabel('Time (s)')

    while True:
        try:
            # Get all available data from the queue
            while not data_queue.empty():
                data = data_queue.get(block=False)
                if data is None:
                    plt.close(fig)
                    return # Exit signal
                
                t, vx, vy, cx, cy, roll, pitch, yaw, height = data
                time_buf.append(t)
                vel_x_buf.append(vx)
                vel_y_buf.append(vy)
                cmd_x_buf.append(cx)
                cmd_y_buf.append(cy)
                roll_buf.append(roll)
                pitch_buf.append(pitch)
                yaw_buf.append(yaw)
                height_buf.append(height)

            # Update plot
            line_vel_x.set_data(time_buf, vel_x_buf)
            line_vel_y.set_data(time_buf, vel_y_buf)
            line_cmd_x.set_data(time_buf, cmd_x_buf)
            line_cmd_y.set_data(time_buf, cmd_y_buf)
            
            line_roll.set_data(time_buf, roll_buf)
            line_pitch.set_data(time_buf, pitch_buf)
            line_yaw.set_data(time_buf, yaw_buf)
            
            line_height.set_data(time_buf, height_buf)

            if len(time_buf) > 1:
                axes[0].set_xlim(min(time_buf), max(time_buf) + 1e-3)
                
                # Dynamic scaling for height
                if len(height_buf) > 0:
                    min_h = min(height_buf)
                    max_h = max(height_buf)
                    margin = 0.1
                    if max_h - min_h < 1e-6:
                        margin = 0.1
                    else:
                        margin = (max_h - min_h) * 0.2
                    axes[2].set_ylim(min_h - margin, max_h + margin)
            
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.01)
            
        except Exception as e:
            print(f"Plotting error: {e}")
            time.sleep(0.1)

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    
    # Disable command curriculum and resampling to allow manual control
    env_cfg.commands.curriculum = False
    env_cfg.commands.resampling_time = 1e10 # Set to a very large value
    env_cfg.env.episode_length_s = 100000 # Set episode length to a very large value to prevent reset

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # Register keyboard events
    if env.viewer:
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_UP, "FORWARD")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_DOWN, "BACKWARD")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_LEFT, "LEFT")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_RIGHT, "RIGHT")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_W, "FORWARD")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_S, "BACKWARD")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_A, "LEFT")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_D, "RIGHT")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_Q, "TURN_LEFT")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_E, "TURN_RIGHT")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_SPACE, "STOP")

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    
    # Set camera to look at the robot
    if env.viewer:
        robot_root_state = env.root_states[robot_index, :3].cpu().numpy()
        cam_pos = robot_root_state + np.array([0.0, 1.0, 0.5]) # Behind and above
        cam_target = robot_root_state
        env.set_camera(cam_pos, cam_target)

    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    
    # Command tracking
    target_vel_x = 0.0
    target_vel_y = 0.0
    target_vel_yaw = 0.0
    
    # Setup data queue for plotter
    data_queue = mp.Queue()
    plot_process = mp.Process(target=plotter_process, args=(data_queue,))
    plot_process.start()

    plot_refresh_rate = 10 # Update plot every N steps

    print("Use arrow keys or WASD to move the robot.")
    print("Use Q/E to turn.")
    print("Use SPACE to stop.")

    try:
        for i in range(10*int(env.max_episode_length)):
            # Handle keyboard events
            if env.viewer:
                events = env.gym.query_viewer_action_events(env.viewer)
                for evt in events:
                    if evt.action == "FORWARD" and evt.value > 0:
                        target_vel_x += 0.1
                    elif evt.action == "BACKWARD" and evt.value > 0:
                        target_vel_x -= 0.1
                    elif evt.action == "LEFT" and evt.value > 0:
                        target_vel_y += 0.1
                    elif evt.action == "RIGHT" and evt.value > 0:
                        target_vel_y -= 0.1
                    elif evt.action == "TURN_LEFT" and evt.value > 0:
                        target_vel_yaw += 0.1
                    elif evt.action == "TURN_RIGHT" and evt.value > 0:
                        target_vel_yaw -= 0.1
                    elif evt.action == "STOP" and evt.value > 0:
                        target_vel_x = 0.0
                        target_vel_y = 0.0
                        target_vel_yaw = 0.0
            
            # Clip commands
            target_vel_x = np.clip(target_vel_x, -0.2, 0.2)
            target_vel_y = np.clip(target_vel_y, -0.2, 0.2)
            target_vel_yaw = np.clip(target_vel_yaw, -1.0, 1.0)

            # Set commands
            env.commands[:, 0] = target_vel_x
            env.commands[:, 1] = target_vel_y
            env.commands[:, 2] = target_vel_yaw

            # if i % 10 == 0:
            #     names = ["Omega", "Grav", "Cmd", "DofPos", "DofVel", "LastAct"]
            #     starts = [0, 3, 6, 9, 21, 33]
            #     ends = [3, 6, 9, 21, 33, 45]
            #     obs_data = obs[0, :].detach().cpu().numpy()
            #     print(f"\n=== Step {i} Observation Debug (Isaac Frame) ===")
            #     for name, s, e in zip(names, starts, ends):
            #         print(f"{name:<8}: {obs_data[s:e]}")
            #     print(f"{'Torques':<8}: {env.torques.detach().cpu().numpy()[0, :]}")
            #     print("====================================================")

            actions = policy(obs.detach())
            obs, _, rews, dones, infos = env.step(actions.detach())
            
            if i % plot_refresh_rate == 0:
                base_lin = env.base_lin_vel[robot_index, :].detach().cpu().numpy()
                roll, pitch, yaw = get_euler_xyz(env.base_quat[robot_index:robot_index+1, :])
                base_height = env.root_states[robot_index, 2].item()
                
                # Send data to plotter
                # t, vx, vy, cx, cy, roll, pitch, yaw, height
                data = (
                    i * env.dt,
                    base_lin[0],
                    base_lin[1],
                    target_vel_x,
                    target_vel_y,
                    roll.item(),
                    pitch.item(),
                    yaw.item(),
                    base_height
                )
                data_queue.put(data)
                

            if RECORD_FRAMES:
                if i % 2:
                    filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                    env.gym.write_viewer_image_to_file(env.viewer, filename)
                    img_idx += 1 
            if MOVE_CAMERA:
                camera_position += camera_vel * env.dt
                env.set_camera(camera_position, camera_position + camera_direction)

    finally:
        data_queue.put(None)
        plot_process.join()

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True) # Essential for PyTorch/CUDA compatibility
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
