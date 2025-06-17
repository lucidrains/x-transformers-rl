from __future__ import annotations

from typing import Literal

from math import ceil
from pathlib import Path
from copy import deepcopy
from functools import partial, wraps
from collections import namedtuple
from random import random

import numpy as np
from tqdm import tqdm

import torch
from torch import nn, tensor, Tensor, is_tensor, cat, stack, zeros, ones, full, arange
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn import Module, GRU
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions import Categorical, Normal
from torch.utils._pytree import tree_map
from torch.nested import nested_tensor

from torch.nn.utils.rnn import pad_sequence
pad_sequence = partial(pad_sequence, batch_first = True)

import einx
from einx import multiply, less, where
from einops import reduce, repeat, einsum, rearrange, pack, unpack
from einops.layers.torch import Rearrange

"""
ein notation:

b - batch
n - sequence
d - dimension
a - actions
sr - state rewards
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

from x_transformers_rl.distributed import (
    maybe_distributed_mean,
    maybe_sync_seed,
    is_distributed,
    all_gather_variable_dim,
    gather_sizes_and_pad_to
)

from x_transformers_rl.evolution import (
    LatentGenePool
)

from x_mlps_pytorch.nff import nFeedforwards, norm_weights_
from x_mlps_pytorch.ff import Feedforwards

from x_mlps_pytorch.mlp import MLP, create_mlp

# memory tuple

Memory = namedtuple('Memory', [
    'state',
    'action',
    'action_log_prob',
    'reward',
    'is_boundary',
    'value',
])

def map_if_tensor(fn, args):
    return tree_map(lambda t: fn(t) if is_tensor(t) else t, args)

def create_memory(*args) -> Memory:
    args = map_if_tensor(lambda t: t.cpu(), args)
    return Memory(*args)

# helpers

def exists(val):
    return val is not None

def default(v, d):
    return v if exists(v) else d

def first(arr):
    return arr[0]

def is_zero_to_one(n):
    return 0. <= n <= 1.

def identity(t, *args, **kwargs):
    return t

def is_empty(t):
    return t.numel() == 0

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

def divisible_by(num, den):
    return (num % den) == 0

def normalize_with_mean_var(t, mean, var, eps = 1e-5):
    return (t - mean) / var.clamp(min = eps).sqrt()

def get_mean_var(
    t,
    mask = None,
    distributed = False
):
    device = t.device
    stats = t[mask] if exists(mask) else t # the statistics need to exclude padding

    if is_empty(stats):
        return t

    if distributed and is_distributed():
        numel = tensor(stats.numel(), device = device)
        dist.all_reduce(numel)

        summed_stats = stats.sum()
        dist.all_reduce(summed_stats)

        mean = summed_stats / numel
        centered = (t - mean)

        centered_squared_sum = centered[mask].square().sum()
        dist.all_reduce(centered_squared_sum)

        var = centered_squared_sum / numel # forget about correction for now
    else:
        mean = stats.mean()
        var = stats.var()
        centered = (t - mean)

    return mean, var

def frac_gradient(t, frac = 1.):
    assert 0 <= frac <= 1.
    return t.detach().lerp(t, frac)

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def entropy(logits, eps = 1e-20):
    prob = logits.softmax(dim = -1)
    return - (log(prob, eps) * prob).sum(dim = -1)

def softclamp(t, value):
    return (t / value).tanh() * value

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
    pack_one = is_tensor(t)

    if pack_one:
        t = [t]

    packed, shapes = pack(t, pattern)

    def inverse(out, inv_pattern = None):
        inv_pattern = default(inv_pattern, pattern)
        out = unpack(out, shapes, inv_pattern)

        if pack_one:
            out = first(out)

        return out

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

# schedule related

def clamp(num, min_val, max_val):
    return min(max(num, min_val), max_val)

def linear_schedule(min_step, max_step, max_val):
    slope = max_val / (max_step - min_step)

    def calc_value(step):
        return clamp(slope * (step - min_step), 0., max_val)

    return calc_value

# action related

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def gumbel_noise(t):
    noise = torch.rand_like(t)
    return -log(-log(noise))

def to_nested_tensor_and_inverse(tensors):
    jagged_dims = tuple(t.shape[-1] for t in tensors)

    tensor = cat(tensors, dim = -1)

    tensor, inverse_batch_pack = pack_with_inverse(tensor, '* n')

    batch = tensor.shape[0]

    tensor = rearrange(tensor, 'b n -> (b n)')

    nt = nested_tensor(tensor.split(jagged_dims * batch), layout = torch.jagged, device = tensor.device, requires_grad = True)

    def inverse(t):
        t = cat(t.unbind())
        out = rearrange(t, '(b n) -> b n', b = batch)
        return inverse_batch_pack(out)

    return nt, inverse, inverse_batch_pack

def nested_sum(t, lens: tuple[int, ...]):
    # needed as backwards not supported for sum with nested tensor

    batch, device = t.shape[0], t.device

    lens = tensor(lens, device = device)
    indices = lens.cumsum(dim = -1) - 1
    indices = repeat(indices, 'n -> b n', b = batch)

    cumsum = t.cumsum(dim = -1)
    sum_at_indices = cumsum.gather(-1, indices)
    sum_at_indices = F.pad(sum_at_indices, (1, 0), value = 0.)

    return sum_at_indices[..., 1:] - sum_at_indices[..., :-1]

def nested_argmax(t, lens: tuple[int, ...]):
    batch, device = t.shape[0], t.device

    t, inverse_batch_pack = pack_with_inverse(t, '* nl')

    t = rearrange(t, 'b nl -> nl b')
    split_tensors = t.split(lens, dim = 0)

    padded_tensors = pad_sequence(split_tensors, batch_first = True, padding_value = max_neg_value(t))
    padded_tensors = rearrange(padded_tensors, 'n l b -> b n l')

    out = padded_tensors.argmax(dim = -1)

    return inverse_batch_pack(out)

class SafeEmbedding(Module):
    def __init__(
        self,
        num_tokens,
        dim,
        num_set_actions = 1
    ):
        super().__init__()
        self.embed = nn.Embedding(num_set_actions * num_tokens, dim)
        self.num_set_actions = num_set_actions

        self.register_buffer('offsets', arange(num_set_actions) * num_tokens, persistent = False)

    def forward(
        self,
        actions
    ):
        if self.num_set_actions == 1:
            actions = rearrange(actions, '... -> ... 1')

        has_actions = actions >= 0.
        actions = torch.where(has_actions, actions, 0)

        actions = actions + self.offsets
        embeds = self.embed(actions)
        embeds = where('... a, ... a d, ', has_actions, embeds, 0.)

        return reduce(embeds, '... a d -> ... d', 'sum')

class Discrete:
    def __init__(
        self,
        raw_actions: Tensor
    ):
        self.raw_actions = raw_actions
        self.probs = raw_actions.softmax(dim = -1)
        self.dist = Categorical(self.probs)

    @staticmethod
    def dim_out(num_actions):
        return num_actions

    def sample(self):
        return self.dist.sample()

    def log_prob(self, value):
        return self.dist.log_prob(value)

    def entropy(self):
        return self.dist.entropy()

class MultiDiscrete:
    def __init__(
        self,
        raw_actions: Tensor | list[Tensor],
        bin_lens: list[int] | None = None
    ):
        if is_tensor(raw_actions):
            assert exists(bin_lens)
            raw_actions = raw_actions.split(bin_lens, dim = -1)

        if not exists(bin_lens):
            bin_lens = [t.shape[-1] for t in raw_actions]

        self.raw_actions = cat(raw_actions, dim = -1)
        self.bin_lens = bin_lens

        nested_raw_actions, self.inv_nested, self.inv_batch_pack = to_nested_tensor_and_inverse(raw_actions)
        self.nested_raw_actions = nested_raw_actions

    def sample(self):
        noise = gumbel_noise(self.raw_actions)
        noised_logits = self.raw_actions + noise
        return nested_argmax(noised_logits, self.bin_lens)

    def log_prob(self, values):
        prob = self.nested_raw_actions.softmax(dim = -1)
        log_prob = log(prob)
        log_prob = self.inv_nested(log_prob)

        offsets = tensor(self.bin_lens, device = values.device).cumsum(dim = -1)
        offsets = F.pad(offsets, (1, -1), value = 0.)

        indices = values + offsets
        return log_prob.gather(-1, indices)

    def entropy(self):
        prob = self.nested_raw_actions.softmax(dim = -1)
        entropies = (-log(prob) * prob).sum(dim = -1)
        entropies = rearrange(entropies, '(b nl) -> b nl', nl = len(self.bin_lens))
        return self.inv_batch_pack(entropies, '* nl')

class Continuous:
    def __init__(
        self,
        raw_actions: Tensor,
        squash = False,
        eps = 1e-5
    ):
        raw_actions = rearrange(raw_actions, '... (d muvar) -> ... d muvar', muvar = 2)
        self.raw_actions = raw_actions

        mean, log_variance = raw_actions.unbind(dim = -1)

        variance = log_variance.exp()

        self.mean_variance = stack((mean, variance))

        std = variance.clamp(min = eps).sqrt()
        self.dist = Normal(mean, std)

        self.squash = squash    

    @staticmethod
    def dim_out(num_actions):
        return num_actions * 2

    def sample(self):
        sampled = self.dist.sample()

        if not self.squash:
            return sampled

        return sampled.tanh()

    def log_prob(self, value, eps = 1e-3):
        log_prob = self.dist.log_prob(value)

        if not self.squash:
            return log_prob

        return log_prob - log(1. - value.pow(2), eps = eps)

    def entropy(self):

        assert not self.squash

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
        state_pred_num_bins = 50,
        continuous_actions = False,
        squash_continuous = False,
        num_set_actions = 1,
        frac_actor_head_gradient = 5e-2,
        frac_critic_head_gradient = 5e-2,
        entropy_weight = 0.02,
        reward_dropout = 0.5, # dropout the prev reward conditioning half the time, so the world model can still operate without previous rewards
        eps_clip = 0.2,
        value_clip = 0.4,
        evolutionary = False,
        dim_latent_gene = None,
        latent_mapper_depth = 2,
        normalize_advantages = True,
        distributed_normalize = True,
        norm_advantages_stats_momentum = 0.25, # 1. would mean not to use exponential smoothing
        actor_ff_depth = 1,
        critic_ff_depth = 2, # certain paper say critic needs to be larger than actor, although in this setting where both draws and slowly shapes the world model, not sure
        actor_use_norm_ff = False,
        critic_use_norm_ff = False,
    ):
        super().__init__()
        self.transformer = transformer
        dim = transformer.attn_layers.dim

        self.reward_embed = nn.Parameter(ones(dim) * 1e-2)

        if not continuous_actions:
            self.action_embeds = SafeEmbedding(num_actions, dim, num_set_actions)
        else:
            self.action_embeds = nn.Sequential(
                MLP(num_actions, dim * 2, dim, activation = nn.SiLU()),
                nn.RMSNorm(dim)
            )

        self.reward_dropout = nn.Dropout(reward_dropout)

        dim = transformer.attn_layers.dim

        self.to_state_embed = nn.Sequential(
            MLP(state_dim, dim * 2, dim, activation = nn.SiLU()),
            nn.RMSNorm(dim),
        )

        # world modeling related

        self.to_pred_done = nn.Sequential(
            nn.Linear(dim * 2, 1),
            Rearrange('... 1 -> ...'),
            nn.Sigmoid()
        )

        state_dim_and_reward = state_dim + 1

        self.state_hl_gauss_loss = HLGaussLoss(
            min_value = -5., # since state is normalized, this should be sufficient
            max_value = 5.,
            num_bins = state_pred_num_bins,
            clamp_to_range = True
        )

        self.world_model_sos_token = nn.Parameter(torch.randn(dim) * 1e-2)
        self.null_next_action_token = nn.Parameter(torch.randn(dim) * 1e-2)

        self.to_pred = nn.Sequential(
            MLP(
                dim * 2,
                dim,
                state_dim_and_reward * state_pred_num_bins,
                activation = nn.SiLU()
            ),
            Rearrange('... (states_rewards bins) -> ... states_rewards bins', bins = state_pred_num_bins)
        )

        # evolutionary

        self.evolutionary = evolutionary

        if evolutionary:
            assert exists(dim_latent_gene)

            self.latent_to_embed = create_mlp(
                dim_in = dim_latent_gene,
                dim = dim,
                depth = latent_mapper_depth,
                activation = nn.SiLU()
            )

        # actor critic

        actor_net_klass = partial(nFeedforwards, input_preserve_magnitude = True) if actor_use_norm_ff else partial(Feedforwards, final_norm = True)
        critic_net_klass = partial(nFeedforwards, input_preserve_magnitude = True) if critic_use_norm_ff else partial(Feedforwards, final_norm = True)

        actor_critic_input_dim = dim * 2  # gets the embedding from the world model as well as a direct projection from the state

        if evolutionary:
            actor_critic_input_dim += dim

        self.critic_head = critic_net_klass(
            dim,
            depth = critic_ff_depth,
            dim_in = actor_critic_input_dim,
            dim_out = critic_dim_pred,
        )

        # https://arxiv.org/abs/2403.03950

        self.critic_hl_gauss_loss = HLGaussLoss(
            min_value = critic_min_max_value[0],
            max_value = critic_min_max_value[1],
            num_bins = critic_dim_pred,
            clamp_to_range = True
        )

        action_type_klass = Discrete if not continuous_actions else Continuous

        self.action_head = actor_net_klass(
            dim,
            depth = actor_ff_depth,
            dim_in = actor_critic_input_dim,
            dim_out = action_type_klass.dim_out(num_actions) * num_set_actions,
        )

        self.num_set_actions = num_set_actions # refactor later, make it work first

        if continuous_actions and squash_continuous:
            action_type_klass = partial(action_type_klass, squash = True)

        self.action_type_klass = action_type_klass
        self.squash_continuous = squash_continuous and continuous_actions

        self.frac_actor_head_gradient = frac_actor_head_gradient
        self.frac_critic_head_gradient = frac_critic_head_gradient

        # advantage normalization

        self.normalize_advantages = normalize_advantages
        self.normalize_advantages_use_batch_stats = distributed_normalize

        assert is_zero_to_one(norm_advantages_stats_momentum)
        norm_advantages_use_ema = norm_advantages_stats_momentum < 1.

        self.norm_advantages_use_ema = norm_advantages_use_ema
        self.norm_advantages_momentum = norm_advantages_stats_momentum

        self.register_buffer('running_advantages_mean', tensor(0.), persistent = norm_advantages_use_ema)
        self.register_buffer('running_advantages_var', tensor(1.), persistent = norm_advantages_use_ema)

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
        batch = pred.shape[0]

        pred = rearrange(pred[:, :-1], 'b n sr l -> (b sr) n l')
        real = rearrange(real, 'b n sr -> (b sr) n')

        loss = self.state_hl_gauss_loss(pred, real, reduction = 'none')

        return reduce(loss, '(b sr) n -> b n', 'sum', b = batch)

    def compute_done_loss(
        self,
        done_pred,
        dones
    ):
        dones = F.pad(dones, (1, 0), value = False)
        return F.binary_cross_entropy(done_pred, dones.float(), reduction = 'none')

    def compute_actor_loss(
        self,
        raw_actions,
        actions,
        old_log_probs,
        returns,
        old_values,
        mask = None
    ):
        dist = self.action_type_klass(raw_actions)
        action_log_probs = dist.log_prob(actions)

        entropy = dist.entropy() if not self.squash_continuous else -action_log_probs

        scalar_old_values = self.critic_hl_gauss_loss(old_values)

        # advantages

        advantages = returns - scalar_old_values.detach()

        # normalization of advantages

        if self.normalize_advantages:
            use_ema, momentum = (self.norm_advantages_use_ema, self.norm_advantages_momentum)

            mean, var = get_mean_var(advantages, mask = mask, distributed = self.normalize_advantages_use_batch_stats)

            if use_ema:
                if self.training:
                    self.running_advantages_mean.lerp_(mean, momentum)
                    self.running_advantages_var.lerp_(var, momentum)

                mean, var = (self.running_advantages_mean, self.running_advantages_var)

            advantages = normalize_with_mean_var(advantages, mean, var)

        # clipped surrogate objective

        ratios = (action_log_probs - old_log_probs).exp()
        clipped_ratios = ratios.clamp(1 - self.eps_clip, 1 + self.eps_clip)

        surr1 = multiply('b n ..., b n ->  b n ...', ratios, advantages)
        surr2 = multiply('b n ..., b n ->  b n ...', clipped_ratios, advantages)

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
        reward_dropout: bool | Tensor | None = None,
        prev_action_dropout: bool | Tensor | None = None,
        next_actions = None,
        latent_gene = None,
        world_model_embed_mult = 1.,
        cache = None,
        **kwargs
    ):
        batch, device = state.shape[0], self.device
        sum_embeds = 0.

        state_embed = self.to_state_embed(state)

        if exists(actions):
            action_embeds = self.action_embeds(actions)

            if not is_tensor(prev_action_dropout):
                prev_action_dropout = tensor(prev_action_dropout, device = device)

            if prev_action_dropout.ndim == 0:
                prev_action_dropout = repeat(prev_action_dropout, '-> b', b = batch)

            if exists(prev_action_dropout):
                action_embeds = multiply('b ..., b -> b ...', action_embeds, (~prev_action_dropout).float())

            sum_embeds = sum_embeds + action_embeds

        if exists(rewards):
            reward_embeds = multiply('..., d -> ... d', rewards, self.reward_embed)

            if reward_embeds.ndim == 1:
                reward_embeds = rearrange(reward_embeds, 'd -> 1 d')

            if reward_embeds.ndim == 2:
                reward_embeds = repeat(reward_embeds, 'n d -> b n d', b = batch)

            if not exists(reward_dropout):
                reward_dropout = self.reward_dropout(ones((batch,), device = device)) == 0.

            if not is_tensor(reward_dropout):
                reward_dropout = tensor(reward_dropout, device = device)

            if reward_dropout.ndim == 0:
                reward_dropout = repeat(reward_dropout, ' -> b', b = batch)

            sum_embeds = sum_embeds + multiply('b ..., b -> b ...', reward_embeds, (~reward_dropout).float())

        # if not inferencing with cache, prepend the sos token

        inference_with_cache = exists(cache)

        prepend_embeds = None
        if not inference_with_cache:
            prepend_embeds = repeat(self.world_model_sos_token, 'd -> b 1 d', b = batch)

        # attend

        embed, cache = self.transformer(
            state,
            *args,
            **kwargs,
            cache = cache,
            input_not_include_cache = True,
            sum_embeds = sum_embeds,
            prepend_embeds = prepend_embeds,
            return_embeddings = True,
            return_intermediates = True
        )

        # if `next_actions` from agent passed in, use it to predict the next state + truncated / terminated signal

        embed_with_actions = None

        if random() < 0.5:
            next_actions = None # random dropout of next actions

        if exists(next_actions):

            next_actions = pad_at_dim(next_actions, (1, 0), value = 0, dim = 1)

            next_action_embeds = self.action_embeds(next_actions)
        else:
            seq_len = embed.shape[1]
            next_action_embeds = repeat(self.null_next_action_token, 'd -> b n d', b = batch, n = seq_len)

        embed_with_actions = cat((embed, next_action_embeds), dim = -1)

        # predicting state and dones, based on agent's action

        state_pred = self.to_pred(embed_with_actions)
        dones = self.to_pred_done(embed_with_actions)

        # actor critic input

        head_input = state_embed

        # maybe evolutionary

        if self.evolutionary:
            assert exists(latent_gene)

            latent_embed = l2norm(self.latent_to_embed(latent_gene))

            if latent_embed.ndim == 2:
                seq_len = head_input.shape[1]
                latent_embed = repeat(latent_embed, 'b d -> b n d', n = seq_len)

            latent_embed = frac_gradient(latent_embed, 0.1)
            head_input = cat((head_input, latent_embed), dim = -1)

        # actor critic heads living on top of transformer - basically approaching online decision transformer except critic learn discounted returns
        # wm stands for 'world model'

        wm_sched_scale = world_model_embed_mult
        embed_for_actor_critic = embed * wm_sched_scale

        if not inference_with_cache:
            # remove sos token if needed
            embed_for_actor_critic = embed_for_actor_critic[:, 1:]

        actor_embed = frac_gradient(embed_for_actor_critic, self.frac_actor_head_gradient * wm_sched_scale) # what fraction of the gradient to pass back to the world model from the actor / critic head

        critic_embed = frac_gradient(embed_for_actor_critic, self.frac_critic_head_gradient * wm_sched_scale)

        # actions

        raw_actions = self.action_head(cat((head_input, actor_embed), dim = -1))

        if self.num_set_actions > 1:
            raw_actions = rearrange(raw_actions, '... (action_sets actions) -> ... action_sets actions', action_sets = self.num_set_actions)

        # values

        values = self.critic_head(cat((head_input, critic_embed), dim = -1))

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
        self.register_buffer('running_mean', zeros(dim))
        self.register_buffer('running_variance', ones(dim))

    def inverse_norm(
        self,
        normalized
    ):
        mean = self.running_mean
        variance = self.running_variance
        std = variance.clamp(min = self.eps).sqrt()

        return normalized * std + mean

    @torch.no_grad()
    def forward_eval(self, x):
        self.eval()
        return self.forward(x)

    def forward(
        self,
        x
    ):
        assert x.shape[-1] == self.dim, f'expected feature dimension of {self.dim} but received {x.shape[-1]}'

        time = self.step.item()

        mean = self.running_mean
        variance = self.running_variance

        normed = normalize_with_mean_var(x, mean, variance)

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
        entropy_weight,
        regen_reg_rate,
        cautious_factor,
        eps_clip,
        value_clip,
        ema_decay,
        continuous_actions = False,
        squash_continuous = True,
        critic_pred_num_bins = 100,
        hidden_dim = 48,
        evolutionary = False,
        evolve_every = 1,
        evolve_after_step = 20,
        num_set_actions = 1,
        latent_gene_pool: dict = dict(
            dim = 128,
            num_genes_per_island = 3,
            num_selected = 2,
            tournament_size = 2
        ),
        world_model_attn_dim_head = 16,
        world_model_heads = 4,
        world_model_attn_hybrid_gru = False,
        world_model: dict = dict(
            depth = 4,
            attn_gate_values = True,
            add_value_residual = True,
            learned_value_residual_mix = True
        ),
        actor_critic_world_model: dict = dict(),
        dropout = 0.,
        max_grad_norm = 0.5,
        ema_kwargs: dict = dict(
            update_model_with_ema_every = 1250
        ),
        save_path = './ppo.pt',
        accelerator: Accelerator | None = None,
        actor_loss_weight = 1.,
        critic_loss_weight = 1.,
        autoregressive_loss_weight = 1.,
        world_model_embed_linear_schedule: (
            float |                 # 1. fully at this step
            tuple[float, float] |   # min step to this max step from 0. to 1.
            None
        ) = None
    ):
        super().__init__()

        self.model_dim = hidden_dim

        self.gene_pool = None

        self.evolutionary = evolutionary
        self.evolve_every = evolve_every
        self.evolve_after_step = evolve_after_step

        if evolutionary:
            self.gene_pool = LatentGenePool(**latent_gene_pool)

        maybe_gru = GRU(hidden_dim, world_model_attn_dim_head * world_model_heads, batch_first = True) if world_model_attn_hybrid_gru else None

        self.model = WorldModelActorCritic(
            num_actions = num_actions,
            num_set_actions = num_set_actions,
            continuous_actions = continuous_actions,
            squash_continuous = squash_continuous,
            critic_dim_pred = critic_pred_num_bins,
            critic_min_max_value = reward_range,
            state_dim = state_dim,
            entropy_weight = entropy_weight,
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
                    verbose = False,
                    attn_hybrid_fold_axial_dim = 1,
                    attn_hybrid_module = maybe_gru,
                    attn_dim_head = world_model_attn_dim_head,
                    heads = world_model_heads,
                    **world_model
                )
            ),
            **actor_critic_world_model
        )

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
        self.entropy_weight = entropy_weight

        self.eps_clip = eps_clip
        self.value_clip = value_clip

        self.save_path = Path(save_path)

        self.register_buffer('step', tensor(0))
        self.to(accelerator.device)

        # slowly titrate in the world model embed into actor / critic, as initially it won't provide much value

        self.world_model_sched = None

        if exists(world_model_embed_linear_schedule):
            embed_mult_range = world_model_embed_linear_schedule

            if isinstance(embed_mult_range, float):
                embed_mult_range = (0., embed_mult_range)

            self.world_model_sched = linear_schedule(*embed_mult_range, 1.)

        # loss weights

        self.actor_loss_weight = actor_loss_weight
        self.critic_loss_weight = critic_loss_weight
        self.autoregressive_loss_weight = autoregressive_loss_weight

    @property
    def device(self):
        return self.step.device

    @property
    def world_model_embed_mult(self):
        if not exists(self.world_model_sched):
            return 1.

        return self.world_model_sched(self.step.item())

    def save(self):
        if not self.accelerator.is_main_process:
            return

        torch.save({
            'model': self.model.state_dict(),
        }, str(self.save_path))

    def load(self):
        if not self.save_path.exists():
            return

        data = torch.load(str(self.save_path), weights_only = True)

        self.model.load_state_dict(data['model'])

    def learn(
        self,
        memories: list[list[Memory]],
        episode_lens,
        gene_ids,
        reward_dropouts,
        prev_action_dropouts,
        fitnesses = None
    ):

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

        # all gather

        data_tensors = (
            states,
            actions,
            rewards,
            old_log_probs,
            returns,
            values,
            is_boundaries,
            gene_ids,
            episode_lens,
            reward_dropouts,
            prev_action_dropouts
        )

        if is_distributed():
            data_tensors = tuple(gather_sizes_and_pad_to(t, dim = 1) if t.ndim > 1 else t for t in data_tensors)

            data_tensors = tuple(first(all_gather_variable_dim(t, dim = 0)) for t in data_tensors)

        # transformer world model is trained on all states per episode all at once
        # will slowly incorporate other ssl objectives + regularizations from the transformer field

        dataset = TensorDataset(*data_tensors)

        dl = DataLoader(dataset, batch_size = self.batch_size, shuffle = True)

        rsnorm_copy = deepcopy(self.rsnorm) # learn the state normalization alongside in a copy of the state norm module, copy back at the end

        # maybe wrap

        if exists(self.accelerator):
            wrapped_model, wrapped_rsnorm_copy, optimizer, dl = self.accelerator.prepare(model, rsnorm_copy, optimizer, dl)

        wrapped_model.train()
        wrapped_rsnorm_copy.train()

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
                episode_lens,
                reward_dropout,
                prev_action_dropout
             ) in dl:

                latent_gene = None

                if self.evolutionary:
                    latent_gene = self.gene_pool[gene_ids]

                seq = arange(states.shape[1], device = self.device)
                mask = less('n, b -> b n', seq, episode_lens)

                prev_actions = pad_at_dim(
                    actions,
                    (1, -1),
                    dim = 1,
                    value = 0. if self.continuous_actions else -1
                )

                rewards = F.pad(rewards, (1, -1), value = 0.)

                states_with_rewards, inverse_pack = pack_with_inverse((states, rewards), 'b n *')

                states_with_rewards = self.rsnorm.forward_eval(states_with_rewards)

                states, rewards = inverse_pack(states_with_rewards)

                raw_actions, values, states_with_rewards_pred, done_pred, _ = wrapped_model(
                    states,
                    rewards = rewards,
                    reward_dropout = reward_dropout,
                    prev_action_dropout = prev_action_dropout,
                    actions = prev_actions,
                    latent_gene = latent_gene,
                    next_actions = actions, # prediction of the next state needs to be conditioned on the agent's chosen action on that state, and will make the world model interactable
                    mask = mask,
                    world_model_embed_mult = self.world_model_embed_mult
                )

                # autoregressive loss for transformer world modeling - there's nothing better atm, even if deficient

                world_model_loss = model.compute_autoregressive_loss(
                    states_with_rewards_pred,
                    states_with_rewards
                )

                world_model_loss = world_model_loss[mask]

                # predicting termination head

                done_pred_mask = F.pad(mask, (1, 0), value = True)

                pred_done_loss = model.compute_done_loss(done_pred, dones)
                pred_done_loss = pred_done_loss[done_pred_mask]

                # update actor and critic

                actor_loss = model.compute_actor_loss(
                    raw_actions,
                    actions,
                    old_log_probs,
                    returns,
                    old_values,
                    mask = mask
                )

                critic_loss = model.compute_critic_loss(
                    values,
                    returns,
                    old_values,
                )

                # add world modeling loss + ppo actor / critic loss

                actor_critic_loss = (
                    actor_loss * self.actor_loss_weight +
                    critic_loss * self.critic_loss_weight
                )[mask]

                loss = (
                    actor_critic_loss.mean() +
                    (world_model_loss.mean() + pred_done_loss.mean()) * self.autoregressive_loss_weight
                )

                if exists(self.accelerator):
                    self.accelerator.backward(loss)
                else:
                    loss.backward()

                # gradient clipping

                self.clip_grad_norm_(wrapped_model.parameters(), self.max_grad_norm)

                # update

                optimizer.step()
                optimizer.zero_grad()

                # norm any weights

                norm_weights_(self)

                # log losses

                logs = dict(
                    actor_loss = actor_loss.mean(),
                    critic_loss = critic_loss.mean(),
                    autoreg_loss = world_model_loss.mean(),
                    pred_done_loss = pred_done_loss.mean()
                )

                # update state norm

                wrapped_rsnorm_copy(states_with_rewards[mask])

                # finally update the gene pool, moving the fittest individual to the very left

                if (
                    self.evolutionary and
                    exists(fitnesses) and
                    self.step.item() > self.evolve_after_step and
                    divisible_by(self.step.item(), self.evolve_every)
                ):
                    self.gene_pool.evolve_(fitnesses)

                    logs.update(fitnesses = fitnesses)

                self.accelerator.log(logs)

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

        normed_state_with_reward = self.rsnorm.forward_eval(state)

        normed_state = normed_state_with_reward[..., :-1]

        if has_reward:
            reward = normed_state_with_reward[..., -1:]

        raw_actions, *_, next_hiddens = self.model(
            normed_state,
            rewards = reward,
            latent_gene = latent_gene,
            cache = hiddens,
        )

        raw_actions = rearrange(raw_actions, '1 1 ... -> ...')

        return raw_actions, next_hiddens

# main

class Learner(Module):
    def __init__(
        self,
        state_dim,
        num_actions,
        reward_range,
        world_model: dict,
        num_set_actions = 1,
        continuous_actions = False,
        discretize_continuous = True,
        continuous_discretized_bins = 50,
        squash_continuous = True,
        continuous_actions_clamp: tuple[float, float] | None = None,
        evolutionary = False,
        evolve_every = 10,
        evolve_after_step = 20,
        reward_dropout_prob = 0.5,
        action_dropout_prob = 0.5,
        curiosity_reward_weight = 0.,
        latent_gene_pool: dict | None = None,
        max_timesteps = 500,
        batch_size = 8,
        num_episodes_per_update = 64,
        lr = 0.0008,
        betas = (0.9, 0.99),
        lam = 0.95,
        gamma = 0.99,
        eps_clip = 0.2,
        value_clip = 0.4,
        entropy_weight = .01,
        regen_reg_rate = 1e-4,
        cautious_factor = 0.1,
        epochs = 3,
        ema_decay = 0.9,
        save_every = 100,
        accelerate_kwargs: dict = dict(),
        agent_kwargs: dict = dict()
    ):
        super().__init__()

        assert divisible_by(num_episodes_per_update, batch_size)

        self.accelerator = Accelerator(**accelerate_kwargs)

        # if doing continuous actions but want it discretized, have the learning orchestrator take care of converting the discrete to continuous from agent -> env

        # todo - move logic onto Agent so during deployment mapping is taken care of there

        self.discrete_to_continuous = None

        if continuous_actions and discretize_continuous:
            continuous_actions = False
            num_set_actions = num_actions  # fix this later, num actions for continuous should be 1
            num_actions = continuous_discretized_bins
            assert exists(continuous_actions_clamp)

            min_val, max_val = continuous_actions_clamp
            self.discrete_to_continuous = torch.linspace(min_val, max_val, continuous_discretized_bins)

        self.num_set_actions = num_set_actions

        self.agent = Agent(
            state_dim = state_dim,
            num_set_actions = num_set_actions,
            num_actions = num_actions,
            continuous_actions = continuous_actions,
            squash_continuous = squash_continuous,
            reward_range = reward_range,
            world_model = world_model,
            evolutionary = evolutionary,
            evolve_every = evolve_every,
            evolve_after_step = evolve_after_step,
            latent_gene_pool = latent_gene_pool,
            epochs = epochs,
            max_timesteps = max_timesteps,
            batch_size = batch_size,
            lr = lr,
            betas = betas,
            lam = lam,
            gamma = gamma,
            entropy_weight = entropy_weight,
            regen_reg_rate = regen_reg_rate,
            cautious_factor = cautious_factor,
            eps_clip = eps_clip,
            value_clip = value_clip,
            ema_decay = ema_decay,
            accelerator = self.accelerator,
            **agent_kwargs
        )
        
        self.num_episodes_per_update = num_episodes_per_update

        self.max_timesteps = max_timesteps

        # parallelizing gene / episodes

        num_processes = self.accelerator.num_processes
        process_index = self.accelerator.process_index

        genes_arange = arange(self.agent.gene_pool.num_genes if exists(self.agent.gene_pool) else 1)
        episodes_arange = arange(num_episodes_per_update)

        episode_genes = torch.cartesian_prod(episodes_arange, genes_arange)

        num_rollouts = episode_genes.size(0)

        assert episode_genes.size(0) >= num_processes

        # evenly group across processes, say 5 episode genes across 3 processes should be (2, 2, 1)

        split_sizes = (
            full((num_processes,), num_rollouts // num_processes) +
            (arange(num_processes) < (num_rollouts % num_processes))
        )

        sharded_episode_genes = episode_genes.split(split_sizes.tolist(), dim = 0)

        self.episode_genes_for_process = sharded_episode_genes[process_index].tolist()

        # environment

        self.num_actions = num_actions
        self.continuous_actions = continuous_actions
        self.continuous_actions_clamp = continuous_actions_clamp

        # reward dropout for the world models

        self.reward_dropout_prob = reward_dropout_prob

        # action dropout for world models

        self.action_dropout_prob = action_dropout_prob

        # weight for curiosity reward, based on entropy of state prediction

        self.curiosity_reward_weight = curiosity_reward_weight

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
        num_learning_updates: int,
        seed = None,
        max_timesteps = None
    ):
        has_curiosity_reward = self.curiosity_reward_weight > 0.
        max_timesteps = default(max_timesteps, self.max_timesteps)

        agent, device, num_episodes_per_update, is_main = self.agent, self.device, self.num_episodes_per_update, self.accelerator.is_main_process

        memories = []
        episode_lens = []
        gene_ids = []
        one_update_reward_dropouts = []
        one_update_action_dropouts = []

        if exists(seed):
            torch.manual_seed(seed)
            np.random.seed(seed)

        agent.eval()
        model = agent.ema_model

        # maybe evolutionary

        num_genes = 1
        fitnesses = None

        # interact with environment for experience

        for learning_update in tqdm(range(num_learning_updates), desc = 'updates', position = 0, disable = not is_main):

            seed = maybe_sync_seed(device)
            torch.manual_seed(seed)

            if agent.evolutionary:
                num_genes = agent.gene_pool.num_genes

                fitnesses = zeros((num_genes,), device = device) # keeping track of fitness of each gene

                # episode seeds

                episode_seeds = torch.randint(0, int(1e7), (num_episodes_per_update,))

            # precompute reward dropouts

            reward_dropouts = torch.rand((num_episodes_per_update,), device = device) < self.reward_dropout_prob

            # precompute prev action dropouts

            action_dropouts = torch.rand((num_episodes_per_update,), device = device) < self.action_dropout_prob

            # roll out episode for a world model (optionally of a given gene) against environment

            for episode, gene_id in tqdm(self.episode_genes_for_process, desc = 'episodes' if not agent.evolutionary else 'episodes-genes', position = 1, disable = not is_main, leave = False):

                latent_gene = None
                reset_kwargs = dict()

                if agent.evolutionary:
                    latent_gene = agent.gene_pool[gene_id]
                    episode_seed = episode_seeds[episode]
                    reset_kwargs.update(seed = episode_seed.item())

                one_episode_memories = []

                reward_dropout = reward_dropouts[episode]
                prev_action_dropout = action_dropouts[episode]

                reset_out = env.reset(**reset_kwargs)

                if isinstance(reset_out, tuple):
                    state, *_ = reset_out
                else:
                    state = reset_out

                state = from_numpy(state).to(device)

                if self.continuous_actions:
                    prev_action = zeros((self.num_actions,), device = device)
                else:
                    prev_action = -1 if self.num_set_actions == 1 else ([-1] * self.num_set_actions)
                    prev_action = tensor(prev_action, device = device)

                prev_reward = tensor(0.).to(device)

                world_model_cache = None

                @torch.no_grad()
                def state_to_pred_action_and_value(state, prev_action, prev_reward, latent_gene = None):
                    nonlocal world_model_cache

                    state_with_reward, inverse_pack = pack_with_inverse((state, prev_reward), '*')

                    normed_state_reward = self.agent.rsnorm.forward_eval(state_with_reward)

                    normed_state, normed_reward = inverse_pack(normed_state_reward)

                    model.eval()

                    if exists(latent_gene):
                        latent_gene = rearrange(latent_gene, 'd -> 1 d')

                    normed_state = rearrange(normed_state, 'd -> 1 1 d')
                    prev_action = rearrange(prev_action, '... -> 1 1 ...')

                    raw_actions, values, state_pred, _, world_model_cache = model.forward_eval(
                        normed_state,
                        rewards = normed_reward,
                        reward_dropout = reward_dropout,
                        prev_action_dropout = prev_action_dropout,
                        latent_gene = latent_gene,
                        cache = world_model_cache,
                        actions = prev_action,
                        world_model_embed_mult = agent.world_model_embed_mult
                    )

                    raw_actions = rearrange(raw_actions, '1 1 ... -> ...')
                    values = rearrange(values, '1 1 d -> d')

                    state_pred = state_pred[:, -1] # last token, in the case of sos token added
                    state_pred_entropy = entropy(state_pred).mean(dim = -1)
                    state_pred_entropy = rearrange(state_pred_entropy, '1 ->')

                    return model.action_type_klass(raw_actions), values, state_pred_entropy

                cumulative_rewards = 0.

                for timestep in range(max_timesteps):

                    dist, value, state_pred_entropy = state_to_pred_action_and_value(state, prev_action, prev_reward, latent_gene)

                    action = dist.sample()
                    action_log_prob = dist.log_prob(action)

                    action_to_env = action.cpu()

                    if exists(self.discrete_to_continuous):
                        action_to_env = self.discrete_to_continuous[action_to_env]

                    elif self.continuous_actions and exists(self.continuous_actions_clamp):
                        # environment clamping for now, before incorporating squashed gaussian etc

                        clamp_min, clamp_max = self.continuous_actions_clamp
                        action.clamp_(clamp_min, clamp_max)

                    env_step_out = env.step(action_to_env.tolist())

                    if len(env_step_out) >= 4:
                        next_state, reward, terminated, truncated, *_ = env_step_out
                    elif len(env_step_out) == 3:
                        next_state, reward, terminated = env_step_out
                        truncated = False
                    else:
                        raise RuntimeError('invalid number of returns from environment .step')

                    next_state = from_numpy(next_state).to(device)

                    reward = float(reward)

                    if has_curiosity_reward:
                        reward += state_pred_entropy.item() * self.curiosity_reward_weight

                    cumulative_rewards += reward

                    prev_action = action
                    prev_reward = tensor(reward).to(device) # from the xval paper, we know pre-norm transformers can handle scaled tokens https://arxiv.org/abs/2310.02989

                    memory = create_memory(state, action, action_log_prob, tensor(reward), tensor(terminated), value)

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
                gene_ids.append(gene_id)
                one_update_reward_dropouts.append(reward_dropout)
                one_update_action_dropouts.append(prev_action_dropout)

                # add list[Memory] to all episode memories list[list[Memory]]

                memories.append(one_episode_memories)

            # updating of the agent

            self.accelerator.wait_for_everyone()

            if agent.evolutionary:
                fitnesses = self.accelerator.reduce(fitnesses)

            agent.learn(
                memories,
                tensor(episode_lens, device = device),
                tensor(gene_ids, device = device),
                stack(one_update_reward_dropouts),
                stack(one_update_action_dropouts),
                fitnesses
            )

            memories.clear()
            episode_lens.clear()
            gene_ids.clear()
            one_update_reward_dropouts.clear()
            one_update_action_dropouts.clear()

            if divisible_by(learning_update, self.save_every):
                self.agent.save()

        self.agent.save()

        print(f'training complete')
