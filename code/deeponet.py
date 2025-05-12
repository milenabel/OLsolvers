import jax
import jax.numpy as jnp
import flax.linen as nn

class DeepONet(nn.Module):
    trunk_layers: int
    branch_layers: int
    hidden_dim: int
    output_dim: int

    def setup(self):
        self.trunk_net = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.tanh,
            *[layer for _ in range(self.trunk_layers - 1) for layer in (nn.Dense(self.hidden_dim), nn.tanh)],
            nn.Dense(self.output_dim)
        ])
        self.branch_net = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.tanh,
            *[layer for _ in range(self.branch_layers - 1) for layer in (nn.Dense(self.hidden_dim), nn.tanh)],
            nn.Dense(self.output_dim)
        ])

    def __call__(self, x, a):
        """
        Supports both:
          - Batched input: x shape (B, G, d), a shape (B, m)
          - Scalar input: x shape (1, 1, d), a shape (1, m)
        """
        if x.ndim == 3:
            # Batched mode
            trunk_out = jax.vmap(self.trunk_net)(x)  # shape (B, G, output_dim)
            branch_out = jax.vmap(self.branch_net)(a)  # shape (B, output_dim)
            return jnp.sum(trunk_out * branch_out[:, None, :], axis=-1)  # shape (B, G)

        elif x.ndim == 2:
            # Single sample mode: x shape (1, d), a shape (1, m)
            trunk_out = self.trunk_net(x)  # shape (1, output_dim)
            branch_out = self.branch_net(a)  # shape (1, output_dim)
            return jnp.sum(trunk_out * branch_out, axis=-1)  # shape (1,)

        elif x.ndim == 1:
            # Truly scalar evaluation: x shape (d,), a shape (m,)
            trunk_out = self.trunk_net(x)  # shape (output_dim,)
            branch_out = self.branch_net(a)  # shape (output_dim,)
            return jnp.sum(trunk_out * branch_out)  # scalar

        else:
            raise ValueError(f"Unsupported x shape: {x.shape}")
