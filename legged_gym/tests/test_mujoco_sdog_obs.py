
import mujoco
import mujoco.viewer
import numpy as np
import yaml
import os
import time

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

def run_test():
    # Path handling
    # Assumes running from workspace root
    # Use absolute paths or relative to the script location if needed, 
    # but based on current context, we are running from root.
    base_path = "/home/daidy/RL/XGO-Simulation"
    config_path = os.path.join(base_path, "legged_gym/config/sdog.yaml")
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return

    print(f"Loading config from {config_path}")
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    xml_path = os.path.join(base_path, config["xml_path"])
    if not os.path.exists(xml_path):
        print(f"Error: XML file not found at {xml_path}")
        return

    # Load parameters
    default_angles = np.array(config["default_angles"], dtype=np.float32)
    
    ang_vel_scale = config["ang_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
    
    num_actions = config["num_actions"]
    num_obs = config["num_obs"]
    
    # Using cmd_init from config or default to 0
    cmd = np.array(config.get("cmd_init", [0, 0, 0]), dtype=np.float32)

    # Index Mappings (Copied from sim_to_mujoco.py)
    # Map MuJoCo (FR, FL, BR, BL) -> Isaac (BL, BR, FL, FR)
    mujoco_2_isaac_idx = [9, 10, 11, 6, 7, 8, 3, 4, 5, 0, 1, 2]

    # Initialize Mujoco
    print(f"Loading MuJoCo model from {xml_path}")
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    # Set Initial State
    # 1. Set Joint Angles to Default
    # MuJoCo qpos structure: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z, joint_1, ..., joint_n]
    # Free joint is qpos[0:7]
    
    # Set base position (suspended in air)
    d.qpos[0] = 0.0
    d.qpos[1] = 0.0
    d.qpos[2] = 0.5 # High enough to be suspended - 0.5m
    
    # Set base orientation (Identity quaternion: w=1, x=0, y=0, z=0)
    d.qpos[3] = 1.0
    d.qpos[4] = 0.0
    d.qpos[5] = 0.0
    d.qpos[6] = 0.0

    # Set joints to default angles
    # Note: default_angles in config seems to be in MuJoCo order based on usage in sim_to_mujoco.py: 
    # tau = pd_control(..., d.qpos[7:], ...) which uses target_dof_pos initialized from default_angles
    d.qpos[7:] = default_angles
    
    # Set velocities to 0
    d.qvel[:] = 0.0

    # Forward kinematics to update sensors/site positions (without stepping time)
    mujoco.mj_forward(m, d)

    print("\nState initialized:")
    print(f"Base Pos: {d.qpos[0:3]}")
    print(f"Base Quat: {d.qpos[3:7]}")
    print(f"Joint Pos (MuJoCo Order): {d.qpos[7:]}")

    # Compute Observations
    # --------------------
    
    # 1. Base Angular Velocity (Omega)
    # qvel definitions for free joint: [lin_vel_x, lin_vel_y, lin_vel_z, ang_vel_x, ang_vel_y, ang_vel_z]
    # Indices: 0,1,2 (linear), 3,4,5 (angular)
    omega = d.qvel[3:6]
    
    # 2. Gravity Orientation (Projected Gravity)
    # Using logic from sim_to_mujoco.py
    quat = np.array([d.qpos[4], d.qpos[5], d.qpos[6], d.qpos[3]]) # [x, y, z, w]
    gravity_orientation = get_gravity_orientation(quat)

    # 3. Joint Positions (qj)
    qj_mujoco = d.qpos[7:]
    qj_sim = (qj_mujoco - default_angles) * dof_pos_scale
    qj_isaac = qj_sim[mujoco_2_isaac_idx]

    # 4. Joint Velocities (dqj)
    dqj_mujoco = d.qvel[6:] # Joints start at qvel[6] in mujoco (0-5 are base 6DOF)
    dqj_sim = dqj_mujoco * dof_vel_scale
    dqj_isaac = dqj_sim[mujoco_2_isaac_idx]

    # 5. Last Action
    # Initialized to zero
    last_action = np.zeros(num_actions, dtype=np.float32)

    # Scale other things
    omega_scaled = omega * ang_vel_scale
    cmd_scaled = cmd * cmd_scale

    # Construct Observation Vector
    obs = np.zeros(num_obs, dtype=np.float32)
    # [omega, gravity_orientation, cmd, qj, dqj, last_action]
    
    current_idx = 0
    
    obs[current_idx : current_idx+3] = omega_scaled
    print(f"Omega (Scaled): {omega_scaled}")
    current_idx += 3
    
    obs[current_idx : current_idx+3] = gravity_orientation
    print(f"Gravity Orientation: {gravity_orientation}")
    current_idx += 3
    
    obs[current_idx : current_idx+3] = cmd_scaled
    print(f"Command (Scaled): {cmd_scaled}")
    current_idx += 3
    
    obs[current_idx : current_idx+num_actions] = qj_isaac
    print(f"DOF Pos (Scaled, Isaac Order): {qj_isaac}")
    current_idx += num_actions
    
    obs[current_idx : current_idx+num_actions] = dqj_isaac
    print(f"DOF Vel (Scaled, Isaac Order): {dqj_isaac}")
    current_idx += num_actions
    
    obs[current_idx : current_idx+num_actions] = last_action
    print(f"Last Action: {last_action}")
    current_idx += num_actions
    
    print("-" * 20)
    print(f"Total Obs Length: {len(obs)}")
    print(f"Is 45? {len(obs) == 45}")
    print("\nStarting Viewer... Close window to exit or press Ctrl+C.")
    print("The robot will be kept static suspended in the air.")
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            # Keep sync with viewer
            viewer.sync()
            # Loop delay
            time.sleep(1/60.0)

    print(f"Full Observation Vector:\n{obs}")

if __name__ == "__main__":
    run_test()
