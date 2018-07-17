from gym.envs.registration import register

import os, sys
sys.path.append(os.path.dirname(__file__)))

register(
    id='AntEnv-v1',
    entry_point='.ant_env:AntEnv',
)