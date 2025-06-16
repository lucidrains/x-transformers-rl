import pytest
import numpy as np

@pytest.mark.parametrize('evolutionary', (False, True))
@pytest.mark.parametrize('continuous_actions', (False, True))
@pytest.mark.parametrize('world_model_gru', (False, True))
def test_e2e(
    evolutionary,
    continuous_actions,
    world_model_gru
):
    class Sim:
        def reset(self, seed = None):
            return np.random.randn(5) # state

        def step(self, actions):
            return np.random.randn(5), np.random.randn(1), False # state, reward, done

    sim = Sim()

    # learning

    from x_transformers_rl import Learner

    learner = Learner(
        state_dim = 5,
        num_actions = 2,
        reward_range = (-1., 1.),
        max_timesteps = 10,
        batch_size = 2,
        num_episodes_per_update = 2,
        continuous_actions = continuous_actions,
        continuous_actions_clamp = (-1., 1.),
        evolutionary = evolutionary,
        latent_gene_pool = dict(
            dim = 32,
            num_genes_per_island = 3,
            num_selected = 2,
            tournament_size = 2
        ),
        agent_kwargs = dict(
            world_model_attn_hybrid_gru = world_model_gru
        ),
        world_model = dict(
            depth = 1,
        )
    )

    learner(sim, 1)

    # deploying

    agent = learner.agent

    hiddens = None
    actions, hiddens = agent(np.random.randn(5), hiddens = hiddens)
    actions, hiddens = agent(np.random.randn(5), hiddens = hiddens)

# action spaces related

def test_multi_discrete():
    import torch

    from x_transformers_rl.x_transformers_rl import (
        Discrete,
        MultiDiscrete
    )

    logits = [
        torch.randn(3, 4, 16, 5),
        torch.randn(3, 4, 16, 12),
        torch.randn(3, 4, 16, 7)
    ]

    lens = [t.shape[-1] for t in logits]

    dists = [Discrete(logit) for logit in logits]

    samples = torch.stack([dist.sample() for dist in dists], dim = -1)
    log_probs = torch.stack([dist.log_prob(value) for dist, value in zip(dists, samples.unbind(dim = -1))], dim = -1)
    entropies = torch.stack([dist.entropy() for dist in dists], dim = -1)

    multi_dist = MultiDiscrete(logits)

    assert multi_dist.sample().shape == (3, 4, 16, 3)
    assert torch.allclose(multi_dist.entropy(), entropies)
    assert torch.allclose(multi_dist.log_prob(samples), log_probs)
