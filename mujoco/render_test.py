import gym
from ant_maze_env import AntMazeEnv

if __name__ == '__main__':
    env = AntMazeEnv()
    env.reset()
    while True:
        obs, r, done, _ = env.step(env.action_space.sample())
        print(obs, r, done)
        env.render()