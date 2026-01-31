import numpy as np


def gradient_descent(grad_f, x0, step_size, num_iters):
    """
    Basic deterministic gradient descent algorithm.
    """
    x = x0.copy()
    xs = [x.copy()]

    for _ in range(num_iters):
        grad = grad_f(x)
        x = x - step_size * grad
        xs.append(x.copy())

    return xs


def stochastic_gradient_descent(grad_f, x0, step_size, num_iters, noise_std):
    """
    Stochastic gradient descent with additive Gaussian noise.
    """
    x = x0.copy()
    xs = [x.copy()]

    for _ in range(num_iters):
        grad = grad_f(x)
        noise = noise_std * np.sqrt(step_size) * np.random.randn(*x.shape)
        x = x - step_size * grad + noise
        xs.append(x.copy())

    return xs

