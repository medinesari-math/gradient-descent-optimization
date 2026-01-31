import numpy as np
import numpy as py

def quadratic(x):
    """
    simple quadratic function:
    f(x) = x^T x
    Minimum is at X = 0
    """

    return np.dot(x, x)


def grad_quadratic(x):

    """
    gradient of f(x) = x^T x
    ∇f(x) = 2x
    """
    return 2 * x

def rosenbrock(x):
    """
    Rosenbrock function. (ref)
    Global minimum at (1, 1).
    """
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def grad_rosenbrock(x):
    """
    Gradient of the Rosenbrock function.
    """
    dfdx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    dfdy = 200 * (x[1] - x[0]**2)
    return np.array([dfdx, dfdy])

def himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2


def grad_himmelblau(x):
    dx = 4*x[0]*(x[0]**2 + x[1] - 11) + 2*(x[0] + x[1]**2 - 7)
    dy = 2*(x[0]**2 + x[1] - 11) + 4*x[1]*(x[0] + x[1]**2 - 7)
    return np.array([dx, dy])
