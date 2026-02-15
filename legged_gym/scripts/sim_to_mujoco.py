import time

import mujoco.viewer
import mujoco
from mujoco import mjtTrn, mjtObj
import numpy as np
import torch
import yaml
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


def pd_control(target_q, q, kp, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp - dq * kd


if __name__ == "__main__":

    with open("legged_gym/config/sdog.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"]
        xml_path = config["xml_path"]

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

        hip_reduction = config["hip_reduction"]

        init_base_height = config["init_base_height"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)

    # 遍历所有执行器
    # print("执行器顺序及其控制的关节：")
    # for actuator_id in range(m.nu):
    #     # 获取执行器名称
    #     actuator_name = mujoco.mj_id2name(m, mjtObj.mjOBJ_ACTUATOR, actuator_id)
        
    #     # 获取执行器类型和关联的关节ID
    #     trn_type = m.actuator_trntype[actuator_id]
    #     trn_id = m.actuator_trnid[actuator_id, 0]  # 假设第一个目标为关节
        
    #     # 检查是否为关节型执行器
    #     if trn_type == mjtTrn.mjTRN_JOINT:
    #         joint_name = mujoco.mj_id2name(m, mjtObj.mjOBJ_JOINT, trn_id)
    #     else:
    #         joint_name = "N/A (非关节执行器)"
        
    #     # 输出信息
    #     print(f"执行器索引 {actuator_id} - 名称: {actuator_name}")
    #     print(f"    控制关节: {joint_name} (关节索引: {trn_id})")
    #     print("-----------------------------------------")


    d = mujoco.MjData(m)
    # Init
    d.qpos[2] = init_base_height
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0] 
    d.qpos[7:] = default_angles
    m.opt.timestep = simulation_dt

    # Forward Compute
    mujoco.mj_forward(m, d)

    # load policy
    policy = torch.jit.load(policy_path)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            tau = np.clip(pd_control(target_dof_pos, d.qpos[7:], kps, d.qvel[6:], kds), -0.45, 0.45)
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # Raw observations from MuJoCo
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                # quat = np.array([d.qpos[4], d.qpos[5], d.qpos[6], d.qpos[3]])
                omega = d.qvel[3:6]

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale
                
                # Construct the observation in the order expected by the policy: 
                # [omega, gravity_orientation, cmd, qj, dqj, last_action]
                obs[:3] = omega
                obs[3:6] = gravity_orientation
                obs[6:9] = cmd * cmd_scale
                obs[9 : 9 + num_actions] = qj
                obs[9 + num_actions : 9 + 2 * num_actions] = dqj
                obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action

                # Print obs debug information
                if log_on and counter % (log_interval * control_decimation) == 0:
                    print(f"\n=== Step {counter} Observation Debug Mujoco Order ======")
                    print(f"{'Omega':15}: {omega}")
                    print(f"{'Omega':15}: [{' '.join(f'{x:6.4f}' for x in omega)}]")
                    print(f"{'Grav':15}: [{' '.join(f'{x:6.4f}' for x in gravity_orientation)}]")
                    print(f"{'Vel':15}: [{' '.join(f'{x:6.4f}' for x in d.qvel[6:])}]")
                    print(f"{'Pos':15}: [{' '.join(f'{x:6.4f}' for x in d.qpos[7:])}]")
                    print(f"{'Pos_cmd':15}: [{' '.join(f'{x:6.4f}' for x in target_dof_pos)}]")  
                    print(f"{'Pos_err':15}: [{' '.join(f'{x:6.4f}' for x in target_dof_pos-d.qpos[7:])}]")  
                    print(f"{'Torque_output':15}: [{' '.join(f'{x:6.4f}' for x in tau)}]") 
                    print("====================================================")
                # Policy inference
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action = policy(obs_tensor).detach().numpy().squeeze()
                action[[0, 3, 6, 9]] *= hip_reduction
                target_dof_pos = action * action_scale + default_angles

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
