import flax.linen as nn
import jax
import jax.numpy as jnp

class Predictor(nn.Module):
    branch_layers: int
    trunk_layers: int
    hidden_dim: int
    output_dim: int

    def setup(self):
        self.branch_net = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.tanh,
            *[layer for _ in range(self.branch_layers - 1)
              for layer in (nn.Dense(self.hidden_dim), nn.tanh)],
            nn.Dense(self.output_dim)
        ])
        # trunk_net is not used in this Predictor setup
        self.trunk_net = None

    def __call__(self, a):  
        return self.branch_net(a)  # shape: (batch, output_dim)


class Corrector(nn.Module):
    branch_layers: int
    trunk_layers: int
    hidden_dim: int
    output_dim: int

    def setup(self):
        self.branch_net = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.tanh,
            *[layer for _ in range(self.branch_layers - 1)
              for layer in (nn.Dense(self.hidden_dim), nn.tanh)],
            nn.Dense(self.output_dim)
        ])
        self.trunk_net = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.tanh,
            *[layer for _ in range(self.trunk_layers - 1)
              for layer in (nn.Dense(self.hidden_dim), nn.tanh)],
            nn.Dense(self.output_dim)
        ])

    def __call__(self, x, approx):  # x: (batch, grid, 2), approx: (batch, output_dim)
        trunk_out = jax.vmap(self.trunk_net)(x)          # (batch, grid, out)
        branch_out = self.branch_net(approx)             # (batch, out)
        return jnp.sum(trunk_out * branch_out[:, None, :], axis=-1)  # (batch, grid)


class PredictorCorrector(nn.Module):
    predictor: Predictor
    corrector: Corrector

    def __call__(self, x, a):
        approx = self.predictor(a)           # <- FIXED: predictor takes only `a`
        corrected = self.corrector(x, approx)
        return corrected

+