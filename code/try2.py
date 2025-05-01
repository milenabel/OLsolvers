import jax
import jax.numpy as jnp
from jax import grad

# === Mock DeepONet-style model ===
class MockDeepONet:
    def apply(self, params, x, a):
        # x shape: (1, 1, 2), a shape: (1, input_dim)
        x_val = x[0, 0, 0]
        y_val = x[0, 0, 1]
        # Return a vector like DeepONet: (1, 1, grid_size)
        u0 = jnp.sin(jnp.pi * x_val) + jnp.cos(jnp.pi * y_val)
        u1 = jnp.sin(jnp.pi * x_val * y_val)
        return jnp.array([[[u0, u1]]])  # shape (1, 1, 2)

# === Corrected function ===
def compute_fourth_derivatives(model, params, x, a, idx=0):
    def apply_fn(xi, ai):
        xi_batched = xi[None, None, :]  # shape: (1, 1, 2)
        ai_batched = ai[None, :]        # shape: (1, input_dim)
        output = model.apply(params, xi_batched, ai_batched).squeeze()  # shape: (grid_size,)
        return output[idx]  # scalar output

    def fourth_deriv_scalar(xi, ai):
        def u_x(x_val):
            return apply_fn(jnp.array([x_val, xi[1]]), ai)

        def u_y(y_val):
            return apply_fn(jnp.array([xi[0], y_val]), ai)

        d4x = grad(grad(grad(grad(u_x))))(xi[0])
        d4y = grad(grad(grad(grad(u_y))))(xi[1])
        return d4x, d4y

    return jax.vmap(fourth_deriv_scalar)(x, a)


# === Run test ===
mock_model = MockDeepONet()
mock_params = None

x_test = jnp.array([[0.1, 0.2], [0.5, 0.5], [0.9, 0.8]])  # shape (3, 2)
a_test = jnp.zeros((3, 1))  # dummy branch input

d4x_vals, d4y_vals = compute_fourth_derivatives(mock_model, mock_params, x_test, a_test, idx=0)

# Compare against known analytical derivatives
true_d4x = (jnp.pi**4) * jnp.sin(jnp.pi * x_test[:, 0])  # for u0 = sin(pi x)
true_d4y = (jnp.pi**4) * jnp.cos(jnp.pi * x_test[:, 1])  # for u0 = cos(pi y)

print("Computed d4x:", d4x_vals)
print("True d4x:    ", true_d4x)
print("Computed d4y:", d4y_vals)
print("True d4y:    ", true_d4y)
