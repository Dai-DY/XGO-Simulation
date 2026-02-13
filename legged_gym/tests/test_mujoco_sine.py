
import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque
import threading
import yaml
import os

# Configuration for the sine wave
AMPLITUDE = 0.5  # rad
FREQUENCY = 1.0  # Hz
OFFSET = 0.0     # rad (center of oscillation)
JOINT_NAME = "fr_thigh_joint" # Joint to control
HISTORY_LEN = 500 # Number of points to plot

def run_sine_test():
    # Load config to get XML path
    base_path = "/home/daidy/RL/XGO-Simulation"
    config_path = os.path.join(base_path, "legged_gym/config/sdog.yaml")
    
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    xml_path = os.path.join(base_path, config["xml_path"])
    
    # Load Model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    # Get Joint ID and Actuator ID
    try:
        joint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, JOINT_NAME)
        # Assuming actuator name matches joint name as per XML
        actuator_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, JOINT_NAME)
        print(f"Controlling Joint: {JOINT_NAME} (ID: {joint_id})")
        print(f"Actuator ID: {actuator_id}")
    except Exception as e:
        print(f"Error finding joint/actuator: {e}")
        return

    # Find the qpos address for this joint
    qpos_adr = m.jnt_qposadr[joint_id]
    qvel_adr = m.jnt_dofadr[joint_id]

    # Get Sensor ID and Address
    try:
        sensor_name = "FR_thigh_vel"
        sensor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        if sensor_id == -1:
             print(f"Warning: Sensor {sensor_name} not found!")
             sensor_adr = -1
        else:
             sensor_adr = m.sensor_adr[sensor_id]
             print(f"Sensor '{sensor_name}': ID {sensor_id}, Adr {sensor_adr}")
    except Exception as e:
        print(f"Error finding sensor: {e}")
        sensor_adr = -1
    
    # --- Real-time Plotting Setup ---
    # We'll use matplotlib in interactive mode or a separate thread if needed.
    # For simplicity and stability with MuJoCo viewer, let's use a non-blocking plot update.
    
    x_data = deque(maxlen=HISTORY_LEN)
    pos_data = deque(maxlen=HISTORY_LEN)
    vel_data = deque(maxlen=HISTORY_LEN)
    target_data = deque(maxlen=HISTORY_LEN)
    target_vel_data = deque(maxlen=HISTORY_LEN)
    diff_vel_data = deque(maxlen=HISTORY_LEN)
    sensor_vel_data = deque(maxlen=HISTORY_LEN)
    
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    
    line_pos, = ax1.plot([], [], 'b-', label='Actual Pos')
    line_target, = ax1.plot([], [], 'r--', label='Target Pos')
    
    line_vel, = ax2.plot([], [], 'g-', label='Actual Vel (qvel)')
    line_target_vel, = ax2.plot([], [], 'm--', label='Target Vel')
    line_diff_vel, = ax2.plot([], [], 'k:', label='Diff Vel (dq/dt)', alpha=0.5)
    line_sensor_vel, = ax2.plot([], [], 'c-.', label='Sensor Vel')
    
    ax1.set_title(f"Joint Position: {JOINT_NAME}")
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_title(f"Joint Velocity: {JOINT_NAME}")
    ax2.legend()
    ax2.grid(True)
    
    # Set PD gains manually for this test if needed, or use model defaults
    # XML defines actuators as "motor" with gear=1.
    # Usually we need a PD controller computed in python for torque control 
    # OR we rely on equality constraints/position actuators.
    # The config says "control_type = 'P'" and defines kp/kd.
    kp = 0.5 # Stiff enough for tracking
    kd = 0.005
    
    # Set base to fixed height to avoid falling during test
    d.qpos[2] = 0.5
    
    # Disable Gravity
    m.opt.gravity[:] = 0
    
    print("\nStarting Simulation...")
    print(f"Joint: {JOINT_NAME}, QPos Adr: {qpos_adr}, QVel Adr: {qvel_adr}")
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start_time = time.time()
        step_count = 0
        last_q = None
        last_time = None
        
        while viewer.is_running():
            step_start = time.time()
            current_time = d.time
            
            # --- Fix Base and Freeze Other Joints ---
            # 1. Save active joint state (preserve physics for the target joint)
            saved_q = d.qpos[qpos_adr]
            saved_v = d.qvel[qvel_adr]
            
            # 2. Reset Base (Suspended)
            d.qpos[0:3] = [0, 0, 0.5]
            d.qpos[3:7] = [1, 0, 0, 0] # Identity Quaternion
            d.qvel[0:6] = 0
            
            # 3. Reset All Joints to 0 (Freeze)
            d.qpos[7:] = 0.0
            d.qvel[6:] = 0.0
            
            # 4. Restore Active Joint
            d.qpos[qpos_adr] = saved_q
            d.qvel[qvel_adr] = saved_v
            # ----------------------------------------
            
            # 1. Compute Sine Wave Target
            target_pos = OFFSET + AMPLITUDE * np.sin(2 * np.pi * FREQUENCY * current_time)
            target_vel = AMPLITUDE * 2 * np.pi * FREQUENCY * np.cos(2 * np.pi * FREQUENCY * current_time)
            # target_pos = 0.3 * current_time
            # target_vel = 0.3

            # 2. Get Current State
            current_q = d.qpos[qpos_adr]
            current_v = d.qvel[qvel_adr]
            
            # 3. PD Control Calculation
            # Torque = Kp * (Target - Current) + Kd * (TargetVel - CurrentVel)
            torque = kp * (target_pos - current_q) + kd * (target_vel - current_v)
            
            # Apply Control
            d.ctrl[actuator_id] = torque
            
            # Apply gravity compensation or hold other joints? 
            # For now, let's just control this one and let others dangle or stay at 0 if initialized.
            # (Ideally we should hold others, but user asked for one joint sine wave)
            
            # Step Physics
            mujoco.mj_step(m, d)
            
            # Read Sensor
            sensor_vel = 0.0
            if sensor_adr != -1:
                sensor_vel = d.sensordata[sensor_adr]

            # Update Plot Data (every ~10 steps to save overhead)
            if step_count % 10 == 0:
                # Calc Diff Vel
                diff_vel = 0.0
                if last_q is not None and last_time is not None:
                     dt_sim = current_time - last_time
                     if dt_sim > 1e-6:
                         diff_vel = (current_q - last_q) / dt_sim
                
                last_q = current_q
                last_time = current_time

                x_data.append(current_time)
                pos_data.append(current_q)
                target_data.append(target_pos)
                vel_data.append(current_v)
                target_vel_data.append(target_vel)
                
                diff_vel_data.append(diff_vel)
                sensor_vel_data.append(sensor_vel)
                
                # Update Plots
                line_pos.set_data(x_data, pos_data)
                line_target.set_data(x_data, target_data)
                
                line_vel.set_data(x_data, vel_data)
                line_target_vel.set_data(x_data, target_vel_data)
                line_diff_vel.set_data(x_data, diff_vel_data)
                line_sensor_vel.set_data(x_data, sensor_vel_data)
                
                ax1.relim()
                ax1.autoscale_view()
                ax2.relim()
                ax2.autoscale_view()
                
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

            viewer.sync()
            step_count += 1
            
            # Time synchronization
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    plt.close()

if __name__ == "__main__":
    run_sine_test()
