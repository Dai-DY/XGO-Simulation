
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, task_registry
import torch

def check_joints(args):
    # Get configuration (using the task name passed in args)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # Override some parameters to avoid full simulation overhead if possible
    env_cfg.env.num_envs = 1
    
    # Create environment
    # This will load the asset and populate dof_names
    print(f"Creating environment for task: {args.task}")
    try:
        env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    except Exception as e:
        print(f"Failed to create environment: {e}")
        return

    # Print actuator/joint information
    print("\n" + "="*50)
    print("执行器顺序及其控制的关节 (Isaac Gym Check):")
    print("="*50)
    
    if hasattr(env, 'dof_names'):
        for i, name in enumerate(env.dof_names):
            # In Isaac Gym, the DOF list order corresponds to the action tensor order.
            # Typically 1 DOF = 1 Actuator.
            print(f"执行器索引 {i} - 名称: {name}") 
            print(f"    控制关节: {name} (关节索引: {i})")
            print("-----------------------------------------")
    else:
        print("Error: Environment does not have 'dof_names' attribute.")
        
    print(f"Total DOFs: {len(env.dof_names)}")

if __name__ == '__main__':
    args = get_args()
    check_joints(args)
