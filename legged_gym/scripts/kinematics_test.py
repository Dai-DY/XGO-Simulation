
from legged_gym import LEGGED_GYM_ROOT_DIR
import isaacgym
from isaacgym import gymtorch, gymapi
from legged_gym.envs import *
from legged_gym.utils import  get_args, task_registry
import torch
from isaacgym.torch_utils import quat_rotate_inverse
import numpy as np

def get_kinematics(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # Force 1 environment
    env_cfg.env.num_envs = 1
    
    # Disable noise and randomization for deterministic testing
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    # Subscribe to keyboard events
    if env.viewer:
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_SPACE, "NEXT_JOINT")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_UP, "MOVE_UP")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_DOWN, "MOVE_DOWN")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_LEFT, "DEC_ANGLE")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_RIGHT, "INC_ANGLE")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_R, "RESET")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_Q, "QUIT")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_ESCAPE, "QUIT")
    
    print(f"Joint names: {env.dof_names}")
    
    # Create target joint angles needed
    # Using existing default positions as a starting point
    target_dof_pos = env.default_dof_pos.clone()
    base_height = 0.3
    
    selected_joint_idx = 0
    num_dofs = len(env.dof_names)
    
    print("\nInstructions:")
    print("  SPACE: Select next joint")
    print("  UP/DOWN: Move Base Up/Down")
    print("  LEFT/RIGHT: Decrease/Increase angle")
    print("  R: Reset to default")
    print("  ESC/Q: Quit\n")

    # Initial camera setup flag
    camera_set = False
    
    last_output = ""

    while env.viewer and not env.gym.query_viewer_has_closed(env.viewer):
        # Handle events
        events = env.gym.query_viewer_action_events(env.viewer)
        for event in events:
            if event.action == "QUIT" and event.value > 0:
                print("Quitting...")
                return
            elif event.action == "NEXT_JOINT" and event.value > 0:
                selected_joint_idx = (selected_joint_idx + 1) % num_dofs
                print(f"\nSelected joint: {env.dof_names[selected_joint_idx]}")
            
            elif event.action == "MOVE_UP" and event.value > 0:
                base_height += 0.01
            elif event.action == "MOVE_DOWN" and event.value > 0:
                base_height -= 0.01
                
            elif event.action == "INC_ANGLE" and event.value > 0: # Check if key is pressed (value=1) or repeated
                target_dof_pos[0, selected_joint_idx] += 0.05
                # print(f"Increased {env.dof_names[selected_joint_idx]} at {target_dof_pos[0, selected_joint_idx].item():.2f}")
                
            elif event.action == "DEC_ANGLE" and event.value > 0:
                target_dof_pos[0, selected_joint_idx] -= 0.05
                # print(f"Decreased {env.dof_names[selected_joint_idx]} at {target_dof_pos[0, selected_joint_idx].item():.2f}")

            elif event.action == "RESET" and event.value > 0:
                target_dof_pos = env.default_dof_pos.clone()
                base_height = 0.3
                print("Reset to default positions")
        
        # Override DOFs with target
        # Update both position and velocity to 0 to hold it steady
        gym_dof_state = env.dof_state.view(env.num_envs, env.num_dofs, 2)
        gym_dof_state[0, :, 0] = target_dof_pos[0]
        gym_dof_state[0, :, 1] = 0.0 # velocity
        
        # Override Root State to PIN the robot in the air (upright)
        env.root_states[0, 0:3] = torch.tensor([0.0, 0.0, base_height], device=env.device) # Fix height
        env.root_states[0, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device) # Identity Quat (Upright)
        env.root_states[0, 7:13] = 0.0 # Zero velocities

        # Apply states
        env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))
        env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
        
        # Step simulation to update viewer and physics
        env.gym.simulate(env.sim)
        env.gym.fetch_results(env.sim, True)
        
        # Update Camera once to focus on robot
        if not camera_set:
            # Look at robot from a nice angle
            cam_pos = gymapi.Vec3(1.0, 1.0, 0.5)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.25)
            env.gym.viewer_camera_look_at(env.viewer, env.envs[0], cam_pos, cam_target)
            camera_set = True

        env.gym.refresh_dof_state_tensor(env.sim)
        env.gym.refresh_actor_root_state_tensor(env.sim)
        env.gym.refresh_rigid_body_state_tensor(env.sim)
        env.gym.refresh_net_contact_force_tensor(env.sim)
        
        # Get states
        base_pos = env.root_states[0, 0:3]
        base_quat = env.root_states[0, 3:7]
        base_z = base_pos[2].item()
        
        # Feet positions
        feet_pos_world = env.rigid_body_states[0, env.feet_indices, 0:3]

        # Feet forces
        feet_forces = env.contact_forces[0, env.feet_indices, 2]
        
        # Transform to base frame
        feet_pos_rel_world = feet_pos_world - base_pos.unsqueeze(0)
        
        # Inverse rotate
        base_quat_expanded = base_quat.unsqueeze(0).repeat(len(env.feet_indices), 1)
        feet_pos_base = quat_rotate_inverse(base_quat_expanded, feet_pos_rel_world)
        
        # Draw on screen
        env.gym.clear_lines(env.viewer)
        
        # Print info only on change
        current_output = (
            f"Selected: {env.dof_names[selected_joint_idx]} ({target_dof_pos[0, selected_joint_idx]:.2f})\n"
            f"Base Z: {base_z:.4f}\n"
            f"Feet Z (Base): " + " | ".join([f"{feet_pos_base[i,2]:.4f}" for i in range(len(env.feet_indices))]) + "\n"
            f"Feet Force Z: " + " | ".join([f"{feet_forces[i]:.2f}" for i in range(len(env.feet_indices))])
        )
        
        if current_output != last_output:
            print(current_output)
            last_output = current_output
        
        # Render
        env.gym.step_graphics(env.sim)
        env.gym.draw_viewer(env.viewer, env.sim, True)
        # Sync frame time (approx 60fps)
        env.gym.sync_frame_time(env.sim)

if __name__ == '__main__':
    args = get_args()
    get_kinematics(args)
