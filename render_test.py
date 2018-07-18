import gym
from envs.ant_maze_env import AntMazeEnv
from envs.ant_gather_env import AntGatherEnv

if __name__ == '__main__':
    env = AntMazeEnv()
    env.reset()
    while True:
        obs, r, done, _ = env.step(env.action_space.sample())
        # print(obs, r, done)
        env.render()