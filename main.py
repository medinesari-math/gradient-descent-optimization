import numpy as np
import matplotlib.pyplot as plt

from functions import (
    quadratic, grad_quadratic,
    himmelblau, grad_himmelblau,
    rosenbrock, grad_rosenbrock,
)
from optimizer import gradient_descent, stochastic_gradient_descent


# =========================================================
# Helper function to make contour plots + optimization paths
# =========================================================
def plot_optimization(f, grad_f, x0, step_size, num_iters, noise_std,
                      x_range, y_range, title):

    # Run optimizers
    xs_gd = np.array(
        gradient_descent(grad_f, x0, step_size, num_iters)
    )

    xs_sgd = np.array(
        stochastic_gradient_descent(grad_f, x0, step_size, num_iters, noise_std)
    )

    # Grid for contours
    x_vals = np.linspace(x_range[0], x_range[1], 400)
    y_vals = np.linspace(y_range[0], y_range[1], 400)
    X, Y = np.meshgrid(x_vals, y_vals)

    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))

    # Plot
    plt.figure(figsize=(7, 6))
    plt.contour(X, Y, Z, levels=30)

    plt.plot(xs_gd[:, 0], xs_gd[:, 1],
             linewidth=2.5, label="Gradient Descent")

    plt.plot(xs_sgd[:, 0], xs_sgd[:, 1],
             linestyle="--", linewidth=2, label="Stochastic Gradient Descent")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ===============================
# Plot 1: Quadratic Function
# ===============================
plot_optimization(
    f=quadratic,
    grad_f=grad_quadratic,
    x0=np.array([5.0, -3.0]),
    step_size=0.1,
    num_iters=25,
    noise_std=0.2,
    x_range=(-6, 6),
    y_range=(-6, 6),
    title="Quadratic Function (Convex): GD vs SGD"
)



# ===============================
# Plot 2: Himmelblau Function
# ===============================
plot_optimization(
    f=himmelblau,
    grad_f=grad_himmelblau,
    x0=np.array([0.0, 0.0]),
    step_size=0.01,
    num_iters=400,
    noise_std=0.1,
    x_range=(-6, 6),
    y_range=(-6, 6),
    title="Himmelblau’s Function (Non-Convex): GD vs SGD"
)

# ==========================================
# Plot 3: Himmelblau with Clearly Visible SGD Noise
# ==========================================

# Parameters
x0 = np.array([0.0, 0.0])
step_size = 0.01
num_iters = 80          # fewer steps → randomness visible
noise_std = 0.6         # strong noise

# Run Gradient Descent
xs_gd = gradient_descent(
    grad_himmelblau,
    x0,
    step_size,
    num_iters
)
xs_gd = np.array(xs_gd)

# Run Stochastic Gradient Descent
xs_sgd = stochastic_gradient_descent(
    grad_himmelblau,
    x0,
    step_size,
    num_iters,
    noise_std
)
xs_sgd = np.array(xs_sgd)

# Create contour grid
x_vals = np.linspace(-6, 6, 300)
y_vals = np.linspace(-6, 6, 300)
X, Y = np.meshgrid(x_vals, y_vals)

# Compute Himmelblau values (SAFE way)
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = himmelblau(np.array([X[i, j], Y[i, j]]))

# Plot
plt.figure(figsize=(7, 6))

plt.contour(X, Y, Z, levels=15)

# Deterministic GD (blue line)
plt.plot(
    xs_gd[:, 0],
    xs_gd[:, 1],
    linewidth=3,
    color="tab:blue",
    label="Gradient Descent (Deterministic)"
)

# Stochastic GD (orange scattered points)
plt.scatter(
    xs_sgd[:, 0],
    xs_sgd[:, 1],
    s=35,
    alpha=0.85,
    color="tab:orange",
    label="Stochastic Gradient Descent (High Noise)"
)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Himmelblau Function: Effect of Stochastic Noise")
plt.legend()
plt.tight_layout()
plt.show()
