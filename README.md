# Deterministic and Stochastic Gradient Methods  
**Convergence, Noise, and Optimization Dynamics**

This project presents a mathematical and computational study of deterministic and stochastic Gradient Descent methods. The goal is to analyze convergence behavior and understand how noise influences optimization dynamics in convex and non-convex settings.

---

## Project Overview

The project compares:

- **Gradient Descent (GD):** Uses the exact gradient  
- **Stochastic Gradient Descent (SGD):** Uses noisy gradient estimates  

Experiments investigate how these methods behave when minimizing different objective functions and how stochasticity affects convergence paths.

---

## Mathematical Setting

We study iterative optimization methods of the form:

**Gradient Descent (GD)**  
`x_{k+1} = x_k − α ∇f(x_k)`

**Stochastic Gradient Descent (SGD)**  
`x_{k+1} = x_k − α ∇f(x_k) + noise`

---

## Test Functions

- **Quadratic Function (Convex)** — illustrates stable convergence  
- **Himmelblau Function (Non-Convex)** — demonstrates multi-minima behavior and the exploratory role of noise  

---

## Key Observations

Deterministic Gradient Descent follows smooth trajectories toward a local minimum determined by initialization. Stochastic Gradient Descent exhibits oscillatory behavior due to random perturbations. Moderate noise enables exploration of the function landscape and can help escape local traps in non-convex settings.

---

## Code Structure - Python Implementation

- `functions.py` — Objective functions and gradients  
- `optimizer.py` — GD and SGD implementations  
- `main.py` — Experiment execution and visualization  

## Topics Involved

Numerical optimization, gradient-based algorithms, stochastic processes, and mathematical modeling
