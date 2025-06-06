## x-transformers-rl (wip)

Implementation of a transformer for reinforcement learning using `x-transformers`

## Install

```bash
$ pip install x-transformers-rl
```

## Usage

```python
import numpy as np

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
    world_model = dict(
        attn_dim_head = 16,
        heads = 4,
        depth = 1,
    )
)

learner(sim, 100)
```

## Example

### Lunar Lander

```bash
$ pip install -r requirements.txt
```

Then

```python
$ python train_lander.py
```

## Citation

```bibtex
@inproceedings{Wang2025EvolutionaryPO,
    title = {Evolutionary Policy Optimization},
    author = {Jianren Wang and Yifan Su and Abhinav Gupta and Deepak Pathak},
    year  = {2025},
    url   = {https://api.semanticscholar.org/CorpusID:277313729}
}
```

```bibtex
@article{Schulman2017ProximalPO,
    title   = {Proximal Policy Optimization Algorithms},
    author  = {John Schulman and Filip Wolski and Prafulla Dhariwal and Alec Radford and Oleg Klimov},
    journal = {ArXiv},
    year    = {2017},
    volume  = {abs/1707.06347},
    url     = {https://api.semanticscholar.org/CorpusID:28695052}
}
```

```bibtex
@article{Farebrother2024StopRT,
    title   = {Stop Regressing: Training Value Functions via Classification for Scalable Deep RL},
    author  = {Jesse Farebrother and Jordi Orbay and Quan Ho Vuong and Adrien Ali Taiga and Yevgen Chebotar and Ted Xiao and Alex Irpan and Sergey Levine and Pablo Samuel Castro and Aleksandra Faust and Aviral Kumar and Rishabh Agarwal},
    journal = {ArXiv},
    year   = {2024},
    volume = {abs/2403.03950},
    url    = {https://api.semanticscholar.org/CorpusID:268253088}
}
```

```bibtex
@article{Lee2025HypersphericalNF,
    title   = {Hyperspherical Normalization for Scalable Deep Reinforcement Learning},
    author  = {Hojoon Lee and Youngdo Lee and Takuma Seno and Donghu Kim and Peter Stone and Jaegul Choo},
    journal = {ArXiv},
    year    = {2025},
    volume  = {abs/2502.15280},
    url     = {https://api.semanticscholar.org/CorpusID:276558261}
}
```
