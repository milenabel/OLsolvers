# predictor_corrector.py
import flax.linen as nn
import jax
import jax.numpy as jnp


class PredictorDeepONet(nn.Module):
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

    def __call__(self, x, a):
        # x: (batch, grid, 2) – spatial locations
        # a: (batch, m) – forcing function (FEM)
        trunk_out = jax.vmap(self.trunk_net)(x)              # (batch, grid, out)
        branch_out = jax.vmap(self.branch_net)(a)            # (batch, out)
        combined = jnp.sum(trunk_out * branch_out[:, None, :], axis=-1)  # (batch, grid)
        return combined


class CorrectorDeepONet(nn.Module):
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

    def __call__(self, x, approx):
        # x: (batch, grid, 2) – spatial locations
        # approx: (batch, grid) – predictor outputs
        trunk_out = jax.vmap(self.trunk_net)(x)                # (batch, grid, out)
        branch_out = self.branch_net(approx)                   # (batch, out)
        combined = jnp.sum(trunk_out * branch_out[:, None, :], axis=-1)  # (batch, grid)
        return combined


class PredictorCorrectorDeepONet(nn.Module):
    predictor: PredictorDeepONet
    corrector: CorrectorDeepONet

    def __call__(self, x, a):
        # x: (batch, grid, 2)
        # a: (batch, m)
        approx = self.predictor(x, a)                          # → (batch, grid)
        corrected = self.corrector(x, approx)                 # → (batch, grid)
        return corrected
