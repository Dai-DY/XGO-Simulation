
import isaacgym
from isaacgym import gymapi
from legged_gym.envs import *
from legged_gym.utils import task_registry
import torch
import numpy as np

# Mock Args class
class Args:
    task = "sdog"
    resume = False
    experiment_name = "test_sdog_obs_isaac"
    run_name = ""
    load_run = ""
    checkpoint = -1
    headless = False # Use False to verify if window opens, though user might be headless
    horovod = False
    rl_device = "cuda"
    sim_device = "cuda"
    num_envs = 1
    seed = 42
    use_jit = False
    
    # Sim params
    physics_engine = gymapi.SIM_PHYSX
    use_gpu = True
    use_gpu_pipeline = True
    subscenes = 0
    num_threads = 0
    device = "cuda"

def run_test():
    args = Args()
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # Modify setup for testing static suspended state
    env_cfg.env.num_envs = 1
    # Set initial position high enough to be suspended
    env_cfg.init_state.pos = [0.0, 0.0, 0.5] 
    
    # Fix base link to keep it suspended statically
    # This assumes 'fix_base_link' is supported in the asset config. 
    # If not supported by LeggedRobotCfg directly, we rely on asset settings.
    # checking sdog_config.py, it inherits from LeggedRobotCfg.
    # We can try setting it.
    env_cfg.asset.fix_base_link = True
    
    # Disable randomization
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.noise.add_noise = False
    
    # Create env
    print("Creating Environment...")
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    # Reset to ensure init state
    obs = env.reset()
    
    # Manually set commands to match MuJoCo test [0.1, 0, 0]
    # Commands shape is (num_envs, 3 or 4). Based on sdog, likely 3 or ranges has 4 elements
    # But usually commands tensor is fixed size. sdog uses 3 scales in config 'cmd_scale': [10.0, 10.0, 0.5]
    # Let's check env.commands shape if possible, but hard to check before running.
    # We'll just set the first 3 elements.
    env.commands[:, 0] = 0.1
    env.commands[:, 1] = 0.0
    env.commands[:, 2] = 0.0
    # If there's a 4th, set to 0
    if env.commands.shape[1] > 3:
         env.commands[:, 3] = 0.0

    print("\nEnvironment Created.")
    
    # Apply zero actions - this will run simulation and compute observations
    # obs, _, _, _, _ = env.step(zero_actions)
    
    # To compare with MuJoCo T=0 static state, we need observations immediately after reset
    # But reset() randomizes commands. We need to fix commands and re-compute observations.
    env.commands[:, 0] = 0.1
    env.commands[:, 1] = 0.0
    env.commands[:, 2] = 0.0 
    if env.commands.shape[1] > 3:
         env.commands[:, 3] = 0.0
         
    # Re-compute observations to reflect the manual command change
    env.compute_observations()
    obs = env.obs_buf
    
    # Extract observation (first env)
    obs_cpu = obs[0].cpu().numpy()
    
    # Observation structure: [omega(3), gravity(3), cmd(3), joint_pos(12), joint_vel(12), last_action(12)]
    # Total = 45
    
    idx = 0
    omega = obs_cpu[idx:idx+3]
    idx += 3
    grav = obs_cpu[idx:idx+3]
    idx += 3
    cmd = obs_cpu[idx:idx+3]
    idx += 3
    dof_pos = obs_cpu[idx:idx+12]
    idx += 12
    dof_vel = obs_cpu[idx:idx+12]
    idx += 12
    last_act = obs_cpu[idx:idx+12]
    
    print("\n=== Isaac Gym Observation (First Env, T=0 Static) ===")
    print(f"Omega (Scaled): {omega}")
    print(f"Gravity Orientation: {grav}")
    print(f"Command (Scaled): {cmd}")
    print(f"DOF Pos (Scaled): {dof_pos}")
    print(f"DOF Vel (Scaled): {dof_vel}")
    print(f"Last Action: {last_act}")
    print("-" * 30)
    print(f"Total Length: {len(obs_cpu)}")
    print(f"Full Vector:\n{obs_cpu}")

    print("\n=== Joint Names Order ===")
    print(env.dof_names)
    print("-" * 30)
    
    print("\nStarting Viewer Loop... Press Ctrl+C to exit.")
    
    zero_actions = torch.zeros(env.num_envs, env.num_actions, device=env.device, requires_grad=False)
    # Simple loop to keep window open
    while True:
        env.step(zero_actions)

if __name__ == '__main__':
    run_test()
