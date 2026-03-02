# Tensor Network Approach to Moiré Excitons

Code and data accompanying the manuscript: **(arxiv link)**. This repository provides a reference implementation of tensor network methods applied to moiré excitons.

This repository aims to:
- Provide core algorithms used in the paper
- Include example workflows showing basic usage
- Offer minimal reproducible examples at small system sizes

---

## Repository structure

### `Main_Modules`

Core implementations of the methods presented in the manuscript:
- `Hamiltonians.jl`: Used to fetch the 2D single-particle Hamiltonian on a square lattice as a matrix product operator (MPO)
- `ExcitonKPM.jl`: Kernel Polynomial Algorithms and routines for exciton observables
- `exciton_builders.jl`: functions to build the exciton Hamiltonian MPO
- `extra_util.jl`: extra utilities for tensor manipulations, e.g., interleaved ordering.
- `kin_builders.jl`: helper functions to build kinetic operators in MPO format for single-particle Hamiltonians

### `Examples notebook`

Examples (same as in the manuscript) illustrating methodology at tractable system sizes:
- `Examples_ExcitonTN.ipynb`: Step-by-step notebook showing how to build Hamiltonians, compute observables, and visualize results

---

## Installation

The code is written in **Julia**. The required packages are shown below.

### Required packages

Run the following in the Julia REPL before using the repository:
```julia
using Pkg

Install all dependencies via:

```julia
using Pkg

Pkg.add([
    "ITensors",
    "ITensorMPS",
    "Quantics",
    "QuanticsTCI",
    "TensorCrossInterpolation",
    "TCIITensorConversion",
    "FFTW",
    "ProgressMeter",
    "Plots",
    "LaTeXStrings"
])
