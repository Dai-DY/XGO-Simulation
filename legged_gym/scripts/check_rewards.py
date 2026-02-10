
import os
import sys
import inspect

# Add the current directory to sys.path to ensure modules can be imported
sys.path.append(os.getcwd())

import isaacgym
import legged_gym.envs
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import class_to_dict

def check_reward_functions():
    args = get_args()
    try:
        env_cfg, _ = task_registry.get_cfgs(name=args.task)
        task_class = task_registry.get_task_class(name=args.task)
    except ValueError as e:
        print(f"Error: {e}")
        print(f"Available tasks: {list(task_registry.task_classes.keys())}")
        return

    # Get reward scales
    reward_scales = class_to_dict(env_cfg.rewards.scales)
    
    print("-" * 100)
    print(f"Checking rewards for task: {args.task}")
    print("-" * 100)
    print(f"{'Reward Name':<30} | {'Scale':<10} | {'Function Name':<30} | {'Status'}")
    print("-" * 100)
    
    all_good = True
    
    for name, scale in reward_scales.items():
        if scale == 0:
            continue
            
        func_name = '_reward_' + name
        has_func = hasattr(task_class, func_name)
        
        status = "OK" if has_func else "MISSING"
        if not has_func:
            all_good = False
            
        print(f"{name:<30} | {scale:<10.2f} | {func_name:<30} | {status}")

    print("-" * 100)
    
    # Also check if there are reward functions defined in task_class that are NOT in the config
    # This helps identify available rewards that are unused
    print(f"Available unused reward functions in {task_class.__name__} (scale=0 or missing in config):")
    print("-" * 100)
    
    robot_methods = [m for m in dir(task_class) if m.startswith('_reward_')]
    active_reward_funcs = ['_reward_' + name for name, scale in reward_scales.items() if scale != 0]
    
    for method in robot_methods:
        if method not in active_reward_funcs:
            print(method)

if __name__ == "__main__":
    try:
        check_reward_functions()
    except Exception as e:
        print(f"Error checking rewards: {e}")
        import traceback
        traceback.print_exc()
