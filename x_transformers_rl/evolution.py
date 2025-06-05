import torch
from torch import nn
from torch import cat, tensor
from torch.nn import Module
import torch.nn.functional as F

import einx
from einops.layers.torch import Rearrange
from einops import rearrange, repeat

from x_transformers_rl.distributed import (
    is_distributed,
    maybe_sync_seed
)

# functions

def divisible_by(num, den):
    return (num % den) == 0

def l2norm(t):
    return F.normalize(t, dim = -1)

# latent gene pool

# proposed by Wang et al. evolutionary policy optimization (EPO)
# https://arxiv.org/abs/2503.19037

class LatentGenePool(Module):
    def __init__(
        self,
        dim,
        num_genes_per_island,
        num_selected,
        tournament_size,
        num_elites = 1,             # exempt from genetic mutation and migration
        mutation_prob = 0.25,
        mutation_std_dev = 0.1,
        num_islands = 1,
        migrate_genes_every = 10,   # every number of evolution step to do a migration between islands, if using multi-islands for increasing diversity
        num_frac_migrate = 0.1,     # migrate 10 percent of the bottom population
        fitness_var_threshold = 0.  # the threshold of variance of the fitness scores until which breeding of genes can occur
    ):
        super().__init__()
        assert num_islands >= 1
        assert num_genes_per_island > 2

        self.num_islands = num_islands

        num_genes = num_genes_per_island * num_islands
        self.num_genes = num_genes
        self.num_genes_per_island = num_genes_per_island

        assert 2 <= num_selected < num_genes_per_island, f'must select at least 2 genes for mating'

        self.num_selected = num_selected
        self.num_children = num_genes_per_island - num_selected
        self.tournament_size = tournament_size

        self.dim_gene = dim
        self.genes = nn.Parameter(l2norm(torch.randn(num_genes, dim)))

        # islands

        self.split_islands = Rearrange('(i g) ... -> i g ...', i = num_islands)
        self.merge_islands = Rearrange('i g ... -> (i g) ...')

        self.num_elites = num_elites # todo - redo with affinity maturation algorithm from artificial immune system field

        # mutation related

        self.mutation_prob = mutation_prob
        self.mutation_std_dev = mutation_std_dev

        # control for when to take a genetic algorithm step

        self.fitness_var_threshold = fitness_var_threshold

        # migration related

        assert 0. <= num_frac_migrate <= 1.

        self.num_frac_migrate = num_frac_migrate
        self.migrate_genes_every = migrate_genes_every

        self.register_buffer('step', tensor(0))

    def __getitem__(self, idx):
        return l2norm(self.genes[idx])

    @torch.inference_mode()
    def evolve_(
        self,
        fitnesses,
        temperature = 1.5
    ):
        device, num_selected = fitnesses.device, self.num_selected
        assert fitnesses.ndim == 1 and fitnesses.shape[0] == self.num_genes

        fitness_var = fitnesses.var().item()

        if fitness_var <= self.fitness_var_threshold:
            return

        if is_distributed():
            seed = maybe_sync_seed(device)
            torch.manual_seed(seed)

        # split out the islands

        genes = self.genes
        num_islands = self.num_islands
        has_elites = self.num_elites > 0

        fitnesses = self.split_islands(fitnesses)
        genes = self.split_islands(genes)

        # local competition within each island

        sorted_fitness, sorted_gene_ids = fitnesses.sort(dim = -1, descending = True)

        selected_gene_ids = sorted_gene_ids[:, :num_selected]

        selected_gene_ids_for_gather = repeat(selected_gene_ids, '... -> ... d', d = self.dim_gene)

        selected_genes = genes.gather(1, selected_gene_ids_for_gather)

        # tournament

        num_children = self.num_children

        batch_randperm = torch.randn((num_islands, num_children, num_selected), device = device).argsort(dim = -1)
        tourn_ids = batch_randperm[..., :self.tournament_size]

        sorted_fitness = repeat(sorted_fitness, '... -> ... d', d = tourn_ids.shape[-1])

        tourn_fitness_ids = sorted_fitness.gather(1, tourn_ids)

        parent_ids = tourn_fitness_ids.topk(2, dim = -1).indices

        parent_ids = rearrange(parent_ids, 'i g parents -> i (g parents)')

        parent_ids = repeat(parent_ids, '... -> ... d', d = self.dim_gene)

        parents = selected_genes.gather(1, parent_ids)
        parents = rearrange(parents, 'i (g parents) d -> parents i g d', parents = 2)

        # cross over

        parent1, parent2 = parents

        children = parent1.lerp(parent2, (torch.randn_like(parent1) / temperature).sigmoid())

        # maybe migration

        if (
            divisible_by(self.step.item() + 1, self.migrate_genes_every) and
            self.num_islands > 1 and
            self.num_frac_migrate > 0.
        ):

            if has_elites:
                elites, selected_genes = selected_genes[:, :1], selected_genes[:, 1:]

            num_can_migrate = selected_genes.shape[1]

            num_migrate = max(1, num_can_migrate * self.num_frac_migrate)

            # fixed migration pattern - what i observe to work best, for now
            # todo - option to make it randomly selected with a mask

            selected_genes, migrants = selected_genes[:, -num_migrate:], selected_genes[:, :-num_migrate]

            migrants = torch.roll(migrants, 1, dims = (1,))

            selected_genes = cat((selected_genes, migrants), dim = 1)

            if has_elites:
                selected_genes = cat((elites, selected_genes), dim = 1)

        # concat children

        genes = torch.cat((selected_genes, children), dim = 1)

        # mutate

        if self.mutation_std_dev > 0:

            if has_elites:
                elites, genes = genes[:, :1], genes[:, 1:]

            noise = torch.randn_like(genes) * self.mutation_std_dev

            if self.mutation_prob < 1.:
                should_mutate = torch.rand(genes.shape[0], device = device) <= self.mutation_prob

                noise = einx.where('p, p ...,', should_mutate, noise, 0.)

            genes.add_(noise)

            if has_elites:
                genes = torch.cat((elites, genes), dim = 1)

        genes = self.merge_islands(genes)

        self.genes.copy_(l2norm(genes))

        self.step.add_(1)

        return selected_gene_ids # return the selected gene ids
