import time

import mujoco.viewer
import mujoco
from mujoco import mjtTrn, mjtObj
import numpy as np
import torch
import yaml


class MujocoSim:
    class Config:
        def __init__(self, config_path):
            with open(config_path, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            
            self.policy_path = config["policy_path"]
            self.xml_path = config["xml_path"]

            self.simulation_duration = config["simulation_duration"]
            self.simulation_dt = config["simulation_dt"]
            self.control_decimation = config["control_decimation"]

            self.log_on = config["log_on"]
            self.log_interval = config["log_interval"]

            self.kps = np.full(12, config["kps"], dtype=np.float32)
            self.kds = np.full(12, config["kds"], dtype=np.float32)

            self.default_angles = np.array(config["default_angles"], dtype=np.float32)

            self.ang_vel_scale = config["ang_vel_scale"]
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.action_scale = config["action_scale"]
            self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]

            self.hip_reduction = config["hip_reduction"]

            self.init_base_height = config["init_base_height"]
            
            self.cmd = np.array(config["cmd_init"], dtype=np.float32)

    def __init__(self, config_path="legged_gym/config/sdog.yaml"):
        self.cfg = self.Config(config_path)
        self.init_simulation()
        self.load_policy()

    def init_simulation(self):
        # define context variables
        self.action = np.zeros(self.cfg.num_actions, dtype=np.float32)
        self.target_dof_pos = self.cfg.default_angles.copy()
        self.obs = np.zeros(self.cfg.num_obs, dtype=np.float32)
        self.counter = 0

        # Load robot model
        self.model = mujoco.MjModel.from_xml_path(self.cfg.xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Init
        self.data.qpos[2] = self.cfg.init_base_height
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0] 
        self.data.qpos[7:] = self.cfg.default_angles
        self.model.opt.timestep = self.cfg.simulation_dt

        # Forward Compute
        mujoco.mj_forward(self.model, self.data)

    def load_policy(self):
        self.policy = torch.jit.load(self.cfg.policy_path)

    @staticmethod
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

    @staticmethod
    def pd_control(target_q, q, kp, dq, kd):
        """Calculates torques from position commands"""
        return (target_q - q) * kp - dq * kd
    
    def show_joint_order(self):
        for actuator_id in range(self.model.nu):
            actuator_name = mujoco.mj_id2name(self.model, mjtObj.mjOBJ_ACTUATOR, actuator_id)
            
            trn_type = self.model.actuator_trntype[actuator_id]
            trn_id = self.model.actuator_trnid[actuator_id, 0] 
            
            if trn_type == mjtTrn.mjTRN_JOINT:
                joint_name = mujoco.mj_id2name(self.model, mjtObj.mjOBJ_JOINT, trn_id)
            else:
                joint_name = "N/A (Not a joint)"
            
            print(f"Actuator id {actuator_id} - Name: {actuator_name}")
            print(f"Controlled Joint: {joint_name} (Joint id: {trn_id})")
            print("---------------------------------------------------")

    def get_obs(self):
        # Raw observations from MuJoCo
        qj = self.data.qpos[7:]
        dqj = self.data.qvel[6:]
        quat = self.data.qpos[3:7]
        omega = self.data.qvel[3:6]

        qj = (qj - self.cfg.default_angles) * self.cfg.dof_pos_scale
        dqj = dqj * self.cfg.dof_vel_scale
        gravity_orientation = self.get_gravity_orientation(quat)
        omega = omega * self.cfg.ang_vel_scale
        
        # Construct the observation in the order expected by the policy: 
        # [omega, gravity_orientation, cmd, qj, dqj, last_action]
        self.obs[:3] = omega
        self.obs[3:6] = gravity_orientation
        self.obs[6:9] = self.cfg.cmd * self.cfg.cmd_scale
        self.obs[9 : 9 + self.cfg.num_actions] = qj
        self.obs[9 + self.cfg.num_actions : 9 + 2 * self.cfg.num_actions] = dqj
        self.obs[9 + 2 * self.cfg.num_actions : 9 + 3 * self.cfg.num_actions] = self.action
        
        return self.obs, omega, gravity_orientation

    def step(self):
        tau = np.clip(
            self.pd_control(
                self.target_dof_pos, 
                self.data.qpos[7:], 
                self.cfg.kps, 
                self.data.qvel[6:], 
                self.cfg.kds
            ), 
            -0.45, 
            0.45
        )
        self.data.ctrl[:] = tau
        mujoco.mj_step(self.model, self.data)
        return tau

    def run(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # Close the viewer automatically after simulation_duration wall-seconds.
            start = time.time()
            while viewer.is_running() and time.time() - start < self.cfg.simulation_duration:
                step_start = time.time()
                
                tau = self.step()

                self.counter += 1
                if self.counter % self.cfg.control_decimation == 0:
                    obs, omega, gravity_orientation = self.get_obs()

                    # Print obs debug information
                    if self.cfg.log_on and self.counter % (self.cfg.log_interval * self.cfg.control_decimation) == 0:
                        print(f"\n=== Step {self.counter} Observation Debug Mujoco Order ======")
                        print(f"{'Omega':15}: [{' '.join(f'{x:6.4f}' for x in omega)}]")
                        print(f"{'Grav':15}: [{' '.join(f'{x:6.4f}' for x in gravity_orientation)}]")
                        print(f"{'Vel':15}: [{' '.join(f'{x:6.4f}' for x in self.data.qvel[6:])}]")
                        print(f"{'Pos':15}: [{' '.join(f'{x:6.4f}' for x in self.data.qpos[7:])}]")
                        print(f"{'Pos_cmd':15}: [{' '.join(f'{x:6.4f}' for x in self.target_dof_pos)}]")  
                        print(f"{'Pos_err':15}: [{' '.join(f'{x:6.4f}' for x in self.target_dof_pos-self.data.qpos[7:])}]")  
                        print(f"{'Torque_output':15}: [{' '.join(f'{x:6.4f}' for x in tau)}]") 
                        print("====================================================")
                    
                    # Policy inference
                    obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                    action = self.policy(obs_tensor).detach().numpy().squeeze()
                    
                    # Store raw action for next observation
                    self.action = action.copy()
                    
                    action[[0, 3, 6, 9]] *= self.cfg.hip_reduction
                    self.target_dof_pos = action * self.cfg.action_scale + self.cfg.default_angles

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

if __name__ == "__main__":
    sim = MujocoSim()
    # sim.show_joint_order()
    sim.run()
