from __future__ import annotations

from math import ceil
from pathlib import Path
from copy import deepcopy
from functools import partial, wraps
from collections import deque, namedtuple
from random import randrange

import numpy as np
from tqdm import tqdm

import torch
from torch import nn, tensor, is_tensor, cat, stack
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn import Module
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions import Categorical, Normal
from torch.utils._pytree import tree_map

from torch.nn.utils.rnn import pad_sequence
pad_sequence = partial(pad_sequence, batch_first = True)

import einx
from einops import reduce, repeat, einsum, rearrange, pack, unpack
from einops.layers.torch import Rearrange

"""
ein notation:

b - batch
n - sequence
d - dimension
a - actions
"""

from ema_pytorch import EMA

from adam_atan2_pytorch import AdoptAtan2

from hl_gauss_pytorch import HLGaussLoss

from x_transformers import (
    Decoder,
    ContinuousTransformerWrapper
)

from assoc_scan import AssocScan

from accelerate import Accelerator

from x_transformers_rl.evolution import (
    LatentGenePool
)

# memory tuple

Memory = namedtuple('Memory', [
    'state',
    'action',
    'action_log_prob',
    'reward',
    'is_boundary',
    'value',
    'latent_gene_id'
])

# helpers

def exists(val):
    return val is not None

def default(v, d):
    return v if exists(v) else d

def identity(t, *args, **kwargs):
    return t

def divisible_by(num, den):
    return (num % den) == 0

def normalize(t, eps = 1e-5):
    return (t - t.mean()) / (t.std() + eps)

def softclamp(t, value):
    return (t / value).tanh() * value

def frac_gradient(t, frac = 1.):
    assert 0 <= frac <= 1.
    return t.detach() * (1. - frac) + t * frac

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def entropy(prob):
    return (-prob * log(prob)).sum()

def pad_at_dim(
    t,
    pad: tuple[int, int],
    dim = -1,
    value = 0.
):
    if pad == (0, 0):
        return t

    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def temp_batch_dim(fn):
    @wraps(fn)
    def inner(*args, **kwargs):
        args, kwargs = tree_map(lambda t: rearrange(t, '... -> 1 ...') if is_tensor(t) else t, (args, kwargs))

        out = fn(*args, **kwargs)

        out = tree_map(lambda t: rearrange(t, '1 ... -> ...') if is_tensor(t) else t, out)
        return out

    return inner

def pack_with_inverse(t, pattern):
    packed, shapes = pack(t, pattern)

    def inverse(out, inv_pattern = None):
        inv_pattern = default(inv_pattern, pattern)
        return unpack(out, shapes, inv_pattern)

    return packed, inverse

def from_numpy(
    t,
    dtype = torch.float32
):
    if is_tensor(t):
        return t

    if isinstance(t, np.float64):
        t = np.array(t)

    if isinstance(t, np.ndarray):
        t = torch.from_numpy(t)

    if exists(dtype):
        t = t.type(dtype)

    return t

# distributed helpers

def is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1

def maybe_distributed_mean(t):
    if not is_distributed():
        return t

    dist.all_reduce(t)
    t = t / dist.get_world_size()
    return t

# action related

class Discrete:
    def __init__(
        self,
        raw_actions: Tensor
    ):
        self.raw_actions = raw_actions
        self.probs = raw_actions.softmax(dim = -1)
        self.dist = Categorical(self.probs)

    @classmethod
    def Linear(
        self,
        dim,
        num_actions,
        bias = True
    ) -> nn.Linear:
        return nn.Linear(dim, num_actions, bias = bias)

    def sample(self):
        return self.dist.sample()

    def log_prob(self, value):
        return self.dist.log_prob(value)

    def entropy(self):
        return self.dist.entropy()

class Continuous:
    def __init__(
        self,
        raw_actions: Tensor,
    ):
        raw_actions = rearrange(raw_actions, '... (d muvar) -> ... d muvar', muvar = 2)
        self.raw_actions = raw_actions

        mean, log_variance = raw_actions.unbind(dim = -1)

        variance = log_variance.exp()

        self.mean_variance = stack((mean, variance))
        self.dist = Normal(mean, variance)

    @classmethod
    def Linear(
        self,
        dim,
        num_actions,
        bias = True
    ) -> Module:
        return nn.Sequential(
            nn.RMSNorm(dim),
            nn.Linear(dim, num_actions * 2, bias = bias)
        )

    def sample(self):
        return self.dist.sample()

    def log_prob(self, value):
        return self.dist.log_prob(value)

    def entropy(self):
        return self.dist.entropy()

# world model + actor / critic in one

class WorldModelActorCritic(Module):
    def __init__(
        self,
        transformer: Module,
        num_actions,
        critic_dim_pred,
        critic_min_max_value: tuple[float, float],
        state_dim,
        continuous_actions = False,
        frac_actor_critic_head_gradient = 0.5,
        entropy_weight = 0.02,
        reward_dropout = 0.5, # dropout the prev reward conditioning half the time, so the world model can still operate without previous rewards
        eps_clip = 0.2,
        value_clip = 0.4,
        evolutionary = False,
        dim_latent_gene = None
    ):
        super().__init__()
        self.transformer = transformer
        dim = transformer.attn_layers.dim

        self.reward_embed = nn.Parameter(torch.ones(dim) * 1e-2)

        if not continuous_actions:
            self.action_embeds = nn.Embedding(num_actions, dim)
        else:
            self.action_embeds = nn.Linear(num_actions, dim)

        self.reward_dropout = nn.Dropout(reward_dropout)

        dim = transformer.attn_layers.dim

        self.to_state_embed = nn.Linear(state_dim, dim)

        # world modeling related

        self.to_pred_done = nn.Sequential(
            nn.Linear(dim * 2, 1),
            Rearrange('... 1 -> ...'),
            nn.Sigmoid()
        )

        state_dim_and_reward = state_dim + 1

        self.to_pred = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            Continuous.Linear(dim, state_dim_and_reward)
        )

        # evolutionary

        self.evolutionary = evolutionary

        if evolutionary:
            assert exists(dim_latent_gene)
            self.latent_to_embed = nn.Linear(dim_latent_gene, dim)

        # actor critic

        actor_critic_input_dim = dim * 2  # gets the embedding from the world model as well as a direct projection from the state

        if evolutionary:
            actor_critic_input_dim += dim

        self.critic_head = nn.Sequential(
            nn.Linear(actor_critic_input_dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, critic_dim_pred)
        )

        # https://arxiv.org/abs/2403.03950

        self.critic_hl_gauss_loss = HLGaussLoss(
            min_value = critic_min_max_value[0],
            max_value = critic_min_max_value[1],
            num_bins = critic_dim_pred,
            clamp_to_range = True
        )

        self.is_discrete = not continuous_actions

        action_type_klass = Discrete if not continuous_actions else Continuous

        self.action_type_klass = action_type_klass

        self.action_head = nn.Sequential(
            nn.Linear(actor_critic_input_dim, dim * 2),
            nn.SiLU(),
            action_type_klass.Linear(dim * 2, num_actions)
        )

        self.frac_actor_critic_head_gradient = frac_actor_critic_head_gradient

        # ppo loss related

        self.eps_clip = eps_clip
        self.entropy_weight = entropy_weight

        # clipped value loss related

        self.value_clip = value_clip

        self.register_buffer('dummy', tensor(0), persistent = False)

    @property
    def device(self):
        return self.dummy.device

    def compute_autoregressive_loss(
        self,
        pred,
        real
    ):
        pred_mean, pred_var = pred[..., :-1, :] # todo: fix truncation scenario
        return F.gaussian_nll_loss(pred_mean, real[:, 1:], pred_var, reduction = 'none')

    def compute_done_loss(
        self,
        done_pred,
        dones
    ):
        return F.binary_cross_entropy(done_pred, dones.float(), reduction = 'none')

    def compute_actor_loss(
        self,
        raw_actions,
        actions,
        old_log_probs,
        returns,
        old_values
    ):
        dist = self.action_type_klass(raw_actions)
        action_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        scalar_old_values = self.critic_hl_gauss_loss(old_values)

        # calculate clipped surrogate objective, classic PPO loss

        ratios = (action_log_probs - old_log_probs).exp()
        clipped_ratios = ratios.clamp(1 - self.eps_clip, 1 + self.eps_clip)

        advantages = normalize(returns - scalar_old_values.detach())

        surr1 = einx.multiply('b n ..., b n ->  b n ...', ratios, advantages)
        surr2 = einx.multiply('b n ..., b n ->  b n ...', clipped_ratios, advantages)

        actor_loss = - torch.min(surr1, surr2) - self.entropy_weight * entropy

        actor_loss = reduce(actor_loss, 'b n ... -> b n', 'sum')

        return actor_loss

    def compute_critic_loss(
        self,
        values,
        returns,
        old_values
    ):
        clip, hl_gauss = self.value_clip, self.critic_hl_gauss_loss

        scalar_old_values = hl_gauss(old_values)
        scalar_values = hl_gauss(values)

        # using the proposal from https://www.authorea.com/users/855021/articles/1240083-on-analysis-of-clipped-critic-loss-in-proximal-policy-gradient

        clipped_returns = returns.clamp(-clip, clip)

        clipped_loss = hl_gauss(values, clipped_returns)
        loss = hl_gauss(values, returns)

        old_values_lo = scalar_old_values - clip
        old_values_hi = scalar_old_values + clip

        def is_between(mid, lo, hi):
            return (lo < mid) & (mid < hi)

        critic_loss = torch.where(
            is_between(scalar_values, returns, old_values_lo) |
            is_between(scalar_values, old_values_hi, returns),
            0.,
            torch.min(loss, clipped_loss)
        )

        return critic_loss

    def forward(
        self,
        state,
        *args,
        actions = None,
        rewards = None,
        next_actions = None,
        latent_gene = None,
        **kwargs
    ):
        device = self.device
        sum_embeds = 0.

        state_embed = self.to_state_embed(state)

        if exists(actions):
            if self.is_discrete:
                has_actions = actions >= 0.
                actions = torch.where(has_actions, actions, 0)

            action_embeds = self.action_embeds(actions)

            if self.is_discrete:
                action_embeds = einx.where('b n, b n d, ', has_actions, action_embeds, 0.)

            sum_embeds = sum_embeds + action_embeds

        if exists(rewards):
            reward_embeds = einx.multiply('..., d -> ... d', rewards, self.reward_embed)

            maybe_dropout = self.reward_dropout(torch.ones((), device = device)) > 0.

            sum_embeds = sum_embeds + reward_embeds * maybe_dropout.float()

        embed, cache = self.transformer(
            state,
            *args,
            **kwargs,
            sum_embeds = sum_embeds,
            return_embeddings = True,
            return_intermediates = True
        )

        # if `next_actions` from agent passed in, use it to predict the next state + truncated / terminated signal

        embed_with_actions = None
        if exists(next_actions):
            next_action_embeds = self.action_embeds(next_actions)
            embed_with_actions = cat((embed, next_action_embeds), dim = -1)

        # predicting state and dones, based on agent's action

        state_pred = None
        dones = None

        if exists(embed_with_actions):
            raw_state_pred = self.to_pred(embed_with_actions)
            state_pred = Continuous(raw_state_pred).mean_variance
            dones = self.to_pred_done(embed_with_actions)

        # actor critic heads living on top of transformer - basically approaching online decision transformer except critic learn discounted returns

        embed = frac_gradient(embed, self.frac_actor_critic_head_gradient) # what fraction of the gradient to pass back to the world model from the actor / critic head

        # actor critic input

        actor_critic_input = cat((embed, state_embed), dim = -1)

        # maybe evolutionary

        if self.evolutionary:
            assert exists(latent_gene)

            latent_embed = self.latent_to_embed(latent_gene)

            if latent_embed.ndim == 2:
                latent_embed = repeat(latent_embed, 'b d -> b n d', n = actor_critic_input.shape[1])

            actor_critic_input = cat((actor_critic_input, latent_embed), dim = -1)

        # actions

        raw_actions = self.action_head(actor_critic_input)

        # values

        values = self.critic_head(actor_critic_input)

        return raw_actions, values, state_pred, dones, cache

# RS Norm (not to be confused with RMSNorm from transformers)
# this was proposed by SimBa https://arxiv.org/abs/2410.09754
# experiments show this to outperform other types of normalization

class RSNorm(Module):
    def __init__(
        self,
        dim,
        eps = 1e-5
    ):
        # equation (3) in https://arxiv.org/abs/2410.09754
        super().__init__()
        self.dim = dim
        self.eps = eps

        self.register_buffer('step', tensor(1))
        self.register_buffer('running_mean', torch.zeros(dim))
        self.register_buffer('running_variance', torch.ones(dim))

    def forward(
        self,
        x
    ):
        assert x.shape[-1] == self.dim, f'expected feature dimension of {self.dim} but received {x.shape[-1]}'

        time = self.step.item()

        mean = self.running_mean
        variance = self.running_variance

        normed = (x - mean) / variance.sqrt().clamp(min = self.eps)

        if not self.training:
            return normed

        # update running mean and variance

        with torch.no_grad():

            new_obs_mean = reduce(x, '... d -> d', 'mean')
            new_obs_mean = maybe_distributed_mean(new_obs_mean)

            delta = new_obs_mean - mean

            new_mean = mean + delta / time
            new_variance = (time - 1) / time * (variance + (delta ** 2) / time)

            self.step.add_(1)
            self.running_mean.copy_(new_mean)
            self.running_variance.copy_(new_variance)

        return normed

# GAE

@torch.no_grad()
def calc_gae(
    rewards,
    values,
    masks,
    gamma = 0.99,
    lam = 0.95,
    use_accelerated = None
):
    assert values.shape[-1] == rewards.shape[-1]
    use_accelerated = default(use_accelerated, rewards.is_cuda)

    values = F.pad(values, (0, 1), value = 0.)
    values, values_next = values[..., :-1], values[..., 1:]

    delta = rewards + gamma * values_next * masks - values
    gates = gamma * lam * masks

    scan = AssocScan(reverse = True, use_accelerated = use_accelerated)

    gae = scan(gates, delta)

    returns = gae + values

    return returns

# agent

class Agent(Module):
    def __init__(
        self,
        state_dim,
        num_actions,
        reward_range: tuple[float, float],
        epochs,
        max_timesteps,
        batch_size,
        lr,
        betas,
        lam,
        gamma,
        beta_s,
        regen_reg_rate,
        cautious_factor,
        eps_clip,
        value_clip,
        ema_decay,
        continuous_actions = False,
        critic_pred_num_bins = 100,
        hidden_dim = 48,
        evolutionary = False,
        evolve_every = 1,
        latent_gene_pool: dict = dict(
            dim = 128,
            num_genes_per_island = 3,
            num_selected = 2,
            tournament_size = 2
        ),
        world_model: dict = dict(
            attn_dim_head = 16,
            heads = 4,
            depth = 4,
            attn_gate_values = True,
            add_value_residual = True,
            learned_value_residual_mix = True
        ),
        dropout = 0.25,
        max_grad_norm = 0.5,
        frac_actor_critic_head_gradient = 0.5,
        ema_kwargs: dict = dict(
            update_model_with_ema_every = 1250
        ),
        save_path = './ppo.pt',
        accelerator: Accelerator | None = None
    ):
        super().__init__()

        self.model_dim = hidden_dim

        self.gene_pool = None

        self.evolutionary = evolutionary
        self.evolve_every = evolve_every

        if evolutionary:
            self.gene_pool = LatentGenePool(**latent_gene_pool)

        self.model = WorldModelActorCritic(
            num_actions = num_actions,
            continuous_actions = continuous_actions,
            critic_dim_pred = critic_pred_num_bins,
            critic_min_max_value = reward_range,
            state_dim = state_dim,
            entropy_weight = beta_s,
            eps_clip = eps_clip,
            value_clip = value_clip,
            evolutionary = evolutionary,
            dim_latent_gene = self.gene_pool.dim_gene if evolutionary else None,
            transformer = ContinuousTransformerWrapper(
                dim_in = state_dim,
                dim_out = None,
                max_seq_len = max_timesteps,
                probabilistic = True,
                attn_layers = Decoder(
                    dim = hidden_dim,
                    rotary_pos_emb = True,
                    attn_dropout = dropout,
                    ff_dropout = dropout,
                    **world_model
                )
            )
        )

        self.frac_actor_critic_head_gradient = frac_actor_critic_head_gradient

        # action related

        self.continuous_actions = continuous_actions

        # state + reward normalization

        self.rsnorm = RSNorm(state_dim + 1)

        self.ema_model = EMA(self.model, beta = ema_decay, include_online_model = False, forward_method_names = {'action_type_klass'}, **ema_kwargs)

        self.optimizer = AdoptAtan2(self.model.parameters(), lr = lr, betas = betas, regen_reg_rate = regen_reg_rate, cautious_factor = cautious_factor)

        self.max_grad_norm = max_grad_norm

        self.ema_model.add_to_optimizer_post_step_hook(self.optimizer)

        # accelerator

        self.accelerator = accelerator

        if not exists(accelerator):
            self.clip_grad_norm_ = nn.utils.clip_grad_norm_
        else:
            self.clip_grad_norm_ = accelerator.clip_grad_norm_

        # learning hparams

        self.batch_size = batch_size

        self.epochs = epochs

        self.lam = lam
        self.gamma = gamma
        self.beta_s = beta_s

        self.eps_clip = eps_clip
        self.value_clip = value_clip

        self.save_path = Path(save_path)

        self.register_buffer('step', tensor(0))

    @property
    def device(self):
        return self.step.device

    def save(self):
        torch.save({
            'model': self.model.state_dict(),
        }, str(self.save_path))

    def load(self):
        if not self.save_path.exists():
            return

        data = torch.load(str(self.save_path), weights_only = True)

        self.model.load_state_dict(data['model'])

    def learn(self, memories, episode_lens, fitnesses = None):

        model, optimizer = self.model, self.optimizer

        hl_gauss = self.model.critic_hl_gauss_loss

        # retrieve and prepare data from memory for training - list[list[Memory]]

        def stack_and_to_device(t):
            return stack(t).to(self.device)

        def stack_memories(episode_memories):
            return tuple(map(stack_and_to_device, zip(*episode_memories)))

        memories = map(stack_memories, memories)

        (
            states,
            actions,
            old_log_probs,
            rewards,
            is_boundaries,
            values,
            gene_ids
        ) = tuple(map(pad_sequence, zip(*memories)))

        masks = ~is_boundaries

        # calculate generalized advantage estimate

        scalar_values = hl_gauss(values)

        returns = calc_gae(
            rewards = rewards,            
            masks = masks,
            lam = self.lam,
            gamma = self.gamma,
            values = scalar_values,
            use_accelerated = False
        )

        # transformer world model is trained on all states per episode all at once
        # will slowly incorporate other ssl objectives + regularizations from the transformer field

        dataset = TensorDataset(
            states,
            actions,
            rewards,
            old_log_probs,
            returns,
            values,
            is_boundaries,
            gene_ids,
            episode_lens
        )

        dl = DataLoader(dataset, batch_size = self.batch_size, shuffle = True)

        model.train()

        rsnorm_copy = deepcopy(self.rsnorm) # learn the state normalization alongside in a copy of the state norm module, copy back at the end
        rsnorm_copy.train()

        # maybe wrap

        if exists(self.accelerator):
            model, optimizer, dl = self.accelerator.prepare(model, optimizer, dl)

        for _ in range(self.epochs):
            for (
                states,
                actions,
                rewards,
                old_log_probs,
                returns,
                old_values,
                dones,
                gene_ids,
                episode_lens
             ) in dl:

                latent_gene = None

                if self.evolutionary:
                    latent_gene = self.gene_pool[gene_ids]

                seq = torch.arange(states.shape[1], device = self.device)
                mask = einx.less('n, b -> b n', seq, episode_lens)

                prev_actions = pad_at_dim(
                    actions,
                    (1, -1),
                    dim = 1,
                    value = 0. if self.continuous_actions else -1
                )

                rewards = F.pad(rewards, (1, -1), value = 0.)

                states_with_rewards, inverse_pack = pack_with_inverse((states, rewards), 'b n *')

                with torch.no_grad():
                    self.rsnorm.eval()
                    states_with_rewards = self.rsnorm(states_with_rewards)

                states, rewards = inverse_pack(states_with_rewards)

                raw_actions, values, states_with_rewards_pred, done_pred, _ = model(
                    states,
                    rewards = rewards,
                    actions = prev_actions,
                    latent_gene = latent_gene,
                    next_actions = actions, # prediction of the next state needs to be conditioned on the agent's chosen action on that state, and will make the world model interactable
                    mask = mask
                )

                # autoregressive loss for transformer world modeling - there's nothing better atm, even if deficient

                world_model_loss = model.compute_autoregressive_loss(
                    states_with_rewards_pred,
                    states_with_rewards
                )

                world_model_loss = world_model_loss[mask[:, :-1]]

                # predicting termination head

                pred_done_loss = model.compute_done_loss(done_pred, dones)
                pred_done_loss = pred_done_loss[mask]

                # update actor and critic

                actor_loss = model.compute_actor_loss(
                    raw_actions,
                    actions,
                    old_log_probs,
                    returns,
                    old_values
                )

                critic_loss = model.compute_critic_loss(
                    values,
                    returns,
                    old_values,
                )

                # add world modeling loss + ppo actor / critic loss

                actor_critic_loss = (actor_loss + critic_loss)[mask]

                loss = world_model_loss.mean() + actor_critic_loss.mean() + pred_done_loss.mean()

                if exists(self.accelerator):
                    self.accelerator.backward(loss)
                else:
                    loss.backward()

                self.clip_grad_norm_(model.parameters(), self.max_grad_norm)

                self.optimizer.step()
                self.optimizer.zero_grad()

                rsnorm_copy(states_with_rewards[mask])

                # finally update the gene pool, moving the fittest individual to the very left

                if (
                    self.evolutionary and
                    exists(fitnesses) and
                    divisible_by(self.step.item(), self.evolve_every)
                ):
                    self.gene_pool.evolve_(fitnesses)

                    fitnesses.zero_()

        self.rsnorm.load_state_dict(rsnorm_copy.state_dict())

        self.step.add_(1)

    def forward(
        self,
        state,
        reward = None,
        hiddens = None,
        latent_gene_id = 0
    ):

        latent_gene = None
        if self.evolutionary:
            latent_gene = self.gene_pool[latent_gene_id]
            latent_gene = rearrange(latent_gene, 'd -> 1 d')

        state = from_numpy(state)
        state = state.to(self.device)
        state = rearrange(state, 'd -> 1 1 d')

        has_reward = exists(reward)

        if not has_reward:
            state = F.pad(state, (0, 1), value = 0.)

        with torch.no_grad():
            self.rsnorm.eval()
            normed_state_with_reward = self.rsnorm(state)

        normed_state = normed_state_with_reward[..., :-1]

        if has_reward:
            reward = normed_state_with_reward[..., -1:]

        raw_actions, *_, next_hiddens = self.model(
            normed_state,
            rewards = reward,
            latent_gene = latent_gene,
            cache = hiddens,
        )

        raw_actions = rearrange(raw_actions, '1 1 d -> d')

        return raw_actions, next_hiddens

# main

class Learner(Module):
    def __init__(
        self,
        state_dim,
        num_actions,
        reward_range,
        world_model: dict,
        continuous_actions = False,
        continuous_actions_clamp: tuple[float, float] | None = None,
        evolutionary = False,
        evolve_every = 10,
        latent_gene_pool: dict | None = None,
        max_timesteps = 500,
        batch_size = 8,
        update_episodes = 64,
        lr = 0.0008,
        betas = (0.9, 0.99),
        lam = 0.95,
        gamma = 0.99,
        eps_clip = 0.2,
        value_clip = 0.4,
        beta_s = .01,
        regen_reg_rate = 1e-4,
        cautious_factor = 0.1,
        epochs = 4,
        ema_decay = 0.9,
        save_every = 100,
        accelerate_kwargs: dict = dict()
    ):
        super().__init__()

        assert divisible_by(update_episodes, batch_size)

        self.accelerator = Accelerator(**accelerate_kwargs)

        self.agent = Agent(
            state_dim = state_dim,
            num_actions = num_actions,
            continuous_actions = continuous_actions,
            reward_range = reward_range,
            world_model = world_model,
            evolutionary = evolutionary,
            evolve_every = evolve_every,
            latent_gene_pool = latent_gene_pool,
            epochs = epochs,
            max_timesteps = max_timesteps,
            batch_size = batch_size,
            lr = lr,
            betas = betas,
            lam = lam,
            gamma = gamma,
            beta_s = beta_s,
            regen_reg_rate = regen_reg_rate,
            cautious_factor = cautious_factor,
            eps_clip = eps_clip,
            value_clip = value_clip,
            ema_decay = ema_decay,
            accelerator = self.accelerator,
        )
        
        self.update_episodes = update_episodes

        self.max_timesteps = max_timesteps

        # environment

        self.num_actions = num_actions
        self.continuous_actions = continuous_actions
        self.continuous_actions_clamp = continuous_actions_clamp

        # saving agent

        self.save_every = save_every

        # move to device

        self.to(self.device)

    @property
    def device(self):
        return self.accelerator.device

    def forward(
        self,
        env: object,
        num_episodes: int,
        seed = None,
        max_timesteps = None
    ):
        max_timesteps = default(max_timesteps, self.max_timesteps)
        num_episodes = ceil(num_episodes / self.update_episodes) * self.update_episodes

        agent, device, is_main = self.agent, self.device, self.accelerator.is_main_process

        memories = deque([])
        episode_lens = []

        if exists(seed):
            torch.manual_seed(seed)
            np.random.seed(seed)

        time = 0
        num_policy_updates = 0

        agent.eval()
        model = agent.ema_model

        # maybe evolutionary

        num_genes = 1
        fitnesses = None
        maybe_gene_tqdm = identity

        if agent.evolutionary:
            num_genes = agent.gene_pool.num_genes

            fitnesses = torch.zeros((num_genes,), device = device) # keeping track of fitness of each gene

            # episode seeds

            episode_seeds = torch.randint(0, int(1e7), (num_episodes,))
            episode_seeds = self.accelerator.reduce(episode_seeds)

            maybe_gene_tqdm = tqdm

        # interact with environment for experience

        for episode in tqdm(range(num_episodes), desc = 'episodes', position = 0, disable = not is_main):

            for gene_id in maybe_gene_tqdm(range(num_genes), desc = 'gene', position = 1, disable = not is_main, leave = False):

                latent_gene = None
                reset_kwargs = dict()

                if agent.evolutionary:
                    latent_gene = agent.gene_pool[gene_id]
                    episode_seed = episode_seeds[episode]
                    reset_kwargs.update(seed = episode_seed.item())

                one_episode_memories = deque([])

                reset_out = env.reset(**reset_kwargs)

                if isinstance(reset_out, tuple):
                    state, *_ = reset_out
                else:
                    state = reset_out

                state = from_numpy(state).to(device)

                if self.continuous_actions:
                    prev_action = torch.zeros((self.num_actions,), device = device)
                else:
                    prev_action = tensor(-1).to(device)

                prev_reward = tensor(0.).to(device)

                world_model_cache = None

                @torch.no_grad()
                def state_to_pred_action_and_value(state, prev_action, prev_reward, latent_gene = None):
                    nonlocal world_model_cache

                    state_with_reward, inverse_pack = pack_with_inverse((state, prev_reward), '*')

                    self.agent.rsnorm.eval()
                    normed_state_reward = self.agent.rsnorm(state_with_reward)

                    normed_state, normed_reward = inverse_pack(normed_state_reward)

                    model.eval()

                    if exists(latent_gene):
                        latent_gene = rearrange(latent_gene, 'd -> 1 d')

                    normed_state = rearrange(normed_state, 'd -> 1 1 d')
                    prev_action = rearrange(prev_action, '... -> 1 1 ...')
                    
                    raw_actions, values, _, _, world_model_cache = model.forward_eval(
                        normed_state,
                        rewards = normed_reward,
                        latent_gene = latent_gene,
                        cache = world_model_cache,
                        actions = prev_action
                    )

                    raw_actions = rearrange(raw_actions, '1 1 d -> d')
                    values = rearrange(values, '1 1 d -> d')

                    return model.action_type_klass(raw_actions), values

                cumulative_rewards = 0.

                for timestep in range(max_timesteps):
                    time += 1

                    dist, value = state_to_pred_action_and_value(state, prev_action, prev_reward, latent_gene)

                    action = dist.sample()
                    action_log_prob = dist.log_prob(action)

                    if self.continuous_actions and exists(self.continuous_actions_clamp):
                        # environment clamping for now, before incorporating squashed gaussian etc

                        clamp_min, clamp_max = self.continuous_actions_clamp
                        action.clamp_(clamp_min, clamp_max)

                    env_step_out = env.step(action.tolist())

                    if len(env_step_out) >= 4:
                        next_state, reward, terminated, truncated, *_ = env_step_out
                    elif len(env_step_out) == 3:
                        next_state, reward, terminated = env_step_out
                        truncated = False
                    else:
                        raise RuntimeError('invalid number of returns from environment .step')

                    next_state = from_numpy(next_state).to(device)

                    reward = float(reward)
                    cumulative_rewards += reward

                    prev_action = action
                    prev_reward = tensor(reward).to(device) # from the xval paper, we know pre-norm transformers can handle scaled tokens https://arxiv.org/abs/2310.02989

                    memory = Memory(state, action, action_log_prob, tensor(reward), tensor(terminated), value, tensor(gene_id))

                    one_episode_memories.append(memory)

                    state = next_state

                    # determine if truncating or terminated

                    done = terminated or truncated

                    # take care of truncated by adding a non-learnable memory storing the next value for GAE

                    if done and not terminated:
                        _, next_value, *_ = state_to_pred_action_and_value(state, prev_action, prev_reward, latent_gene)

                        bootstrap_value_memory = memory._replace(
                            state = state,
                            is_boundary = tensor(True),
                            value = next_value
                        )

                        memories.append(bootstrap_value_memory)

                    # break if done

                    if done:
                        break

                # add cumulative reward entry for fitness calculation

                if agent.evolutionary:
                    fitnesses[gene_id] += cumulative_rewards

                # add episode len for training world model actor critic

                episode_lens.append(timestep + 1)

                # add list[Memory] to all episode memories list[list[Memory]]

                memories.append(one_episode_memories)

            # updating of the agent

            if divisible_by(len(memories), self.update_episodes):

                self.agent.learn(
                    memories,
                    tensor(episode_lens, device = device),
                    fitnesses
                )

                num_policy_updates += 1

                memories.clear()
                episode_lens.clear()

            if divisible_by(episode, self.save_every):
                self.agent.save()

        self.agent.save()

        print(f'training complete')
