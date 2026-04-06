# KoopmanModeDecomposition.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://uchytil.github.io/KoopmanModeDecomposition.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://uchytil.github.io/KoopmanModeDecomposition.jl/dev/)
[![Build Status](https://github.com/uchytil/KoopmanModeDecomposition.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/uchytil/KoopmanModeDecomposition.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/uchytil/KoopmanModeDecomposition.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/uchytil/KoopmanModeDecomposition.jl)

**KoopmanModeDecomposition.jl** is a lightweight framework for Extended Dynamic Mode Decomposition (EDMD) style data-driven approximations of the Koopman Operator.


## Quick Start
KMD.jl allows you to define liftings using a simple composition pipeline, fit an approximation of the Koopman operator, and step it forward in time.

```julia
using KMD

# 1. Generate some dummy data (3 variables, 100 timesteps)
X = rand(3, 100)

# 2. Define the lifting function
lifting = [
    Identity();                                  # The raw states
    Delay(1:3) ∘ Identity(1:2);                  # Block delays (τ=1,2,3) on vars 1 & 2
    Monomials(2, drop_constant=true)             # 2nd-degree polynomials
]

# 3. Fit the model using a Truncated SVD solver
solver = TruncatedSVD(atol=1e-4)
model = fit(solver, lifting, X)

# 4. Predict the future
# Note: Because our max delay is 3, the model requires at least 4 historical snapshots to start
X = X[:, end-3:end]
Y = predict(model, X, 50)   # Predict 50 steps forward

```

## Features
- Function composition using the `∘` and `vcat` operators
- Time-delay embedding support via single delays `Delay(1)`, block delays `Delay(1:5)`, and arbitrary delay combinations `Delay([2,8])`.
- Support for multiple regression solvers. So far `PseudoInverse()` and `TruncatedSVD()` are available.