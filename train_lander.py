# environment related

import gymnasium as gym
from shutil import rmtree

evolutionary = True
continuous_actions = False

env = gym.make(
    'LunarLander-v3',
    render_mode = 'rgb_array',
    continuous = continuous_actions
)

state_dim = env.observation_space.shape[0]
num_actions = env.action_space.n if not continuous_actions else env.action_space.shape[0]
reward_range = (-5, 5)
continuous_actions_clamp = (-1., 1.)

# world-model-actor-critic + learning wrapper

from x_transformers_rl import Learner

learner = Learner(
    state_dim = state_dim,
    num_actions = num_actions,
    continuous_actions = continuous_actions,
    reward_range = reward_range,
    continuous_actions_clamp = continuous_actions_clamp,
    squash_continuous = True,
    evolutionary = evolutionary,
    batch_size = 8,
    num_episodes_per_update = 64,
    evolve_every = 5,
    evolve_after_step = 10,
    latent_gene_pool = dict(
        dim = 32,
        num_genes_per_island = 3,
        num_selected = 2,
        tournament_size = 2
    ),
    world_model = dict(
        attn_dim_head = 16,
        heads = 4,
        depth = 4,
        attn_gate_values = True,
        add_value_residual = True,
        learned_value_residual_mix = True
    ),
    agent_kwargs = dict(
        actor_loss_weight = 0.5
    ),
    frac_actor_critic_head_gradient = 1e-1
)

if learner.accelerator.is_main_process:

    video_folder = './recordings'
    record_every = len(learner.episode_genes_for_process) * 2 # record every 2 learning updates
    rmtree(video_folder, ignore_errors = True)

    env = gym.wrappers.RecordVideo(
        env = env,
        video_folder = video_folder,
        name_prefix = 'lunar-video',
        episode_trigger = lambda eps_num: (eps_num % record_every) == 0,
        disable_logger = True
    )

learner(env, 2500)
