import gym
import ant_env

if __name__ == '__main__':
    env = ant_env.AntEnv()
    env.reset()
    while True:
        env.render()