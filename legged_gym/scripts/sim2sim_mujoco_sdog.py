import time
import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
import os

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation

def pd_control(target_q, q, kp, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp - dq * kd

if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming config is in legged_gym/config/sdog.yaml relative to XGO-Simulation root
    # If script is in XGO-Simulation/, then:
    config_path = os.path.join(script_dir, "legged_gym/config/sdog.yaml")

    print(f"Loading config from: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = os.path.join(script_dir, config["policy_path"])
        xml_path = os.path.join(script_dir, config["xml_path"])

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        log_on = config["log_on"]
        log_interval = config["log_interval"]

        kps = np.full(12, config["kps"], dtype=np.float32)
        kds = np.full(12, config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        num_obs_frame = config["num_obs_frame"]

        hip_reduction = config["hip_reduction"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

        # Mapping Explanation:
        # Isaac Gym (URDF) Order: FR, FL, BR, BL  (Based on sdog.urdf joint order)
        # MuJoCo (XML) Order:     BL, BR, FL, FR  (Based on sdog_copy.xml body order)
        
        # We need to map between these two conventions.
        
        # Isaac Indices:
        # FR: 0, 1, 2
        # FL: 3, 4, 5
        # BR: 6, 7, 8
        # BL: 9, 10, 11
        
        # MuJoCo Indices:
        # BL: 0, 1, 2
        # BR: 3, 4, 5
        # FL: 6, 7, 8
        # FR: 9, 10, 11

        # Map MuJoCo vector to Isaac vector (for Observations):
        # We want Isaac[0] (FR) to come from MuJoCo[9] (FR).
        # We want Isaac[3] (FL) to come from MuJoCo[6] (FL).
        # We want Isaac[6] (BR) to come from MuJoCo[3] (BR).
        # We want Isaac[9] (BL) to come from MuJoCo[0] (BL).
        mujoco_2_isaac_idx = [9, 10, 11, 6, 7, 8, 3, 4, 5, 0, 1, 2]
        
        # Map Isaac vector to MuJoCo vector (for Actions/Targets):
        # We want MuJoCo[0] (BL) to come from Isaac[9] (BL).
        # We want MuJoCo[3] (BR) to come from Isaac[6] (BR).
        # We want MuJoCo[6] (FL) to come from Isaac[3] (FL).
        # We want MuJoCo[9] (FR) to come from Isaac[0] (FR).
        isaac_2_mujoco_idx = [9, 10, 11, 6, 7, 8, 3, 4, 5, 0, 1, 2]

        # IMPORTANT: Convert default_angles (Isaac Order) to MuJoCo Order
        # The YAML file defines default angles in FR, FL, BR, BL order (see comments in yaml).
        # The simulation control loop runs in MuJoCo order.
        default_angles_isaac = default_angles.copy()
        default_angles = default_angles_isaac[isaac_2_mujoco_idx]
        print("Default Angles (MuJoCo Order):", default_angles)

    # observations
    obs_buff = torch.zeros(1, num_obs*num_obs_frame, dtype=torch.float)
    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    
    # Initialize target_dof_pos with default angles in MuJoCo order
    target_dof_pos = default_angles.copy()
    
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    print(f"Loading MuJoCo model from: {xml_path}")
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    
    # Initialize state
    d.qpos[2] = 0.16 # Set initial height to 0.15m (sim_to_mujoco used 0.16)
    
    # Set initial joint positions (MuJoCo order)
    d.qpos[7:] = default_angles
    
    m.opt.timestep = simulation_dt

    # load policy
    print(f"Loading policy from: {policy_path}")
    policy = torch.jit.load(policy_path)

    print("Starting simulation...")
    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            
            # Apply PD Control
            # d.qpos[7:] is current joint pos (MuJoCo order)
            # d.qvel[6:] is current joint vel (MuJoCo order)
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, d.qvel[6:], kds)
            d.ctrl[:] = tau
            
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # Raw observations from MuJoCo
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                # Prepare for Observation
                # Note: Observation computation expects (val - default) * scale
                # qj is in MuJoCo order. default_angles is in MuJoCo order. 
                # Subtraction is safe.
                qj_obs = (qj - default_angles) * dof_pos_scale
                dqj_obs = dqj * dof_vel_scale
                
                gravity_orientation = get_gravity_orientation(quat)
                omega_obs = omega * ang_vel_scale
                
                # Construct the observation in the order expected by the policy (Isaac Order)
                # Sdog Policy Observation Structure (45 dim):
                # [AngVel(3), Gravity(3), Commands(3), DofPos(12), DofVel(12), LastActions(12)]
                
                obs[:3] = omega_obs
                obs[3:6] = gravity_orientation
                obs[6:9] = cmd * cmd_scale
                
                # Reorder Joint Pos/Vel from MuJoCo to Isaac
                obs[9 : 9 + num_actions] = qj_obs[mujoco_2_isaac_idx]
                obs[9 + num_actions : 9 + 2 * num_actions] = dqj_obs[mujoco_2_isaac_idx]
                obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action # Last action (already Isaac order)

                if log_on and counter % log_interval == 0:
                    names = ["Omega", "Grav", "Cmd", "DofPos", "DofVel", "LastAct"]
                    starts = [0, 3, 6, 9, 21, 33]
                    ends = [3, 6, 9, 21, 33, 45]
                    print(f"\n=== Step {counter} Observation Debug (Isaac Frame) ===")
                    for name, s, e in zip(names, starts, ends):
                        print(f"{name:<8}: {obs[s:e]}")
                    print(f"tau output: {tau}")
                    print("====================================================")

                # Policy inference
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action = policy(obs_tensor).detach().numpy().squeeze()
                
                # Hip reduction (Isaac Order indices 0,3,6,9 = FR_hip, FL_hip, BR_hip, BL_hip)
                action[[0, 3, 6, 9]] *= hip_reduction
                
                # Convert Action (Isaac) to Target Pos (MuJoCo)
                # Action is delta from default.
                # Target = Default(MuJoCo) + Action(MuJoCo) * Scale
                
                # First reorder action to MuJoCo:
                action_mujoco = action[isaac_2_mujoco_idx]
                target_dof_pos = action_mujoco * action_scale + default_angles

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
