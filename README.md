## Training A Model ##
### Installation ###
1. Create a new python virtual env with python 3.8
2. Install pytorch 1.10 with cuda-11.3:
    - `pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
3. Install Isaac Gym
   - Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym
   - `cd isaacgym/python && pip install -e .`
   - Try running an example `cd examples && python 1080_balls_of_solitude.py`
   - For troubleshooting check docs `isaacgym/docs/index.html`)
4. Install rsl_rl (PPO implementation)
   - Clone https://github.com/leggedrobotics/rsl_rl
   -  `cd rsl_rl && git checkout v1.0.2 && pip install -e .` 
5. Install xgo_gym
    - In this repository, run `pip install -e .`


### Usage ###
1. Train:  
  ```python legged_gym/scripts/train.py --task=xgo```
    -  To run on CPU add following arguments: `--sim_device=cpu`, `--rl_device=cpu` (sim on CPU and rl on GPU is possible).
    -  To run headless (no rendering) add `--headless`.
1. Play a trained policy:  
```python legged_gym/scripts/play.py --task=xgo```
    - By default, the loaded policy is the last model of the last run of the experiment folder.
    - Other runs/model iteration can be selected by setting `load_run` and `checkpoint` in the train config.
1. Keyboard Play:
   ```python legged_gym/scripts/keyboard_play.py --task=xgo```
   - Control the robot using keyboard inputs (WASD for movement, QE for turning, Space to stop) and visualize real-time state plots.
1. Sim to Mujoco:
   ```python legged_gym/scripts/sim_to_mujoco.py --task=xgo```
   - Load a policy trained in Isaac Gym and run it in Mujoco simulation.
1. Check Rewards:
   ```python legged_gym/scripts/check_rewards.py --task=xgo```
   - List all active reward functions and their scales for a specific task configuration.
1. Stand Test:
   ```python legged_gym/scripts/stand_test.py --task=xgo```
   - Test standing behavior (runs the policy but zeroes out actions or modifies behavior for standing tests).

### Tests ###
1. Test Environment:
   ```python legged_gym/tests/test_env.py --task=xgo```
   - Basic environment test. runs the environment with zero actions to verify stability and throughput.
1. Check Joints (Isaac):
   ```python legged_gym/tests/check_joints_isaac.py --task=xgo```
   - Checks joint configuration and asset loading in Isaac Gym context.
1. Kinematics Test:
   ```python legged_gym/tests/kinematics_test.py --task=xgo```
   - Tests robot kinematics (e.g., forward kinematics) with deterministic settings.




