# ant-env
Ant Gather and Ant Maze Mujoco envs, separated from RLLab. Reproduced for use with RL algorithms implemented with separately or other frameworks.

## Usage
```
import gym
from envs.ant_maze_env import AntMazeEnv
from envs.ant_gather_env import AntGatherEnv

if __name__ == '__main__':
    env = AntGatherEnv()
    # env = AntMazeEnv()
    env.reset()
    while True:
        obs, r, done, _ = env.step(env.action_space.sample())
        env.render()
```

## ToDo
* Double Check Reward Functions
* Fix Observations for both
* Reward and Terminal Functions