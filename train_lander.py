# environment related

import gymnasium as gym
from shutil import rmtree

video_folder = './recordings'
record_every = 250

env = gym.make(
    'LunarLander-v3',
    render_mode = 'rgb_array'
)

rmtree(video_folder, ignore_errors = True)

env = gym.wrappers.RecordVideo(
    env = env,
    video_folder = video_folder,
    name_prefix = 'lunar-video',
    episode_trigger = lambda eps_num: (eps_num % record_every) == 0,
    disable_logger = True
)

state_dim = env.observation_space.shape[0]
num_actions = env.action_space.n
reward_range = (-100, 100)

# world-model-actor-critic + learning wrapper

from x_transformers_rl import Learner

learner = Learner(
    state_dim = state_dim,
    num_actions = num_actions,
    reward_range = reward_range,
    world_model = dict(
        attn_dim_head = 16,
        heads = 4,
        depth = 4,
        attn_gate_values = True,
        add_value_residual = True,
        learned_value_residual_mix = True
    )
)

learner(env, 50000)
