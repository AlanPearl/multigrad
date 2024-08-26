---
title: 'MultiGrad: A framework for optimizing MPI-distributed Jax models'
tags:
  - Python
  - Jax
  - MPI
authors:
  - name: Alan N. Pearl
    orcid: 0000-0001-9820-9619
    affiliation: 1
  - name: Gillian D. Beltz-Mohrmann
    orcid: 0000-0002-4392-8920
    affiliation: 1
  - name: Andrew P. Hearin
    orcid: 0000-0003-2219-6852
    affiliation: 1
affiliations:
 - name: HEP Division, Argonne National Laboratory, 9700 South Cass Avenue, Lemont, IL 60439, USA
   index: 1
date: 8 August 2024
bibliography: paper.bib
---

# Summary

`multigrad` is a Python package which greatly facilitates the implementation of data-parallelized, differentiable models using the Jax [@jax2018github] framework.
Leveraging MPI (Message Passing Interface), `multigrad` efficiently sums and propagates gradients of custom-defined summary statistics across processors and computing nodes,
making it a valuable tool for high-performance computing. Its simple yet flexible design makes it applicable to a wide variety of
problems requiring large-data scalable solutions that would benefit from gradient-based optimization techniques.


# Statement of Need

In and beyond the field of cosmology, parameterized models can describe complex systems, provided that the
parameters have been tuned adequately to fit the model to observational data. Fitting capabilities can be increased dramatically
by gradient-based techniques, particularly in high-dimensional parameter spaces. Existing gradient descent tools in Jax do not
inherently support data-parallelism with MPI, creating a speed and memory bottleneck for such computations.

`multigrad` addresses this need by providing an easy-to-use interface for implementing data-parallelized models. It handles
the MPI reductions as well as the mathematical complexities involved in propagating chain rules required to compute the gradient
of the loss, which is a function of parallelized summary statistics, which are in turn functions of the model parameters. At the
same time, it is very flexible in that it allows users to define their own functions to compute their summary statistics and loss.
As a result, this package can enable scalability through parallelization to the optimization routine of nearly any big-data model.

# Method

`multigrad` allows the user to implement a loss term, which is a function of summary statistics, which are functions of parameters,
$L(\vec{y}(\vec{x}))$ where the summary statistics are summed over multiple MPI-linked processes: $\vec{y} = \sum_i\vec{y}_{(i)}$ where $i$ is
the index of each process. In this section, we will derive the gradient of the loss $\vec{\nabla} L$ with respect to the parameters
and as a sum of terms that each process can compute independently.

We will begin from the definition of the multivariate chain rule,

$$ \frac{\partial L}{\partial x_j} = \sum\limits_{k} \frac{\partial L}{\partial y_k} \frac{\partial y_k}{\partial x_j} $$

where $\partial y_k$ = $\sum_i\partial y_{k (i)}$. By pulling out the MPI summation over $i$,

$$ \frac{\partial L}{\partial x_j} = \sum\limits_{i} \sum\limits_{k} \frac{\partial L}{\partial y_k} \frac{\partial y_{k (i)}}{\partial x_j} $$

and by rewriting this as vector-matrix multiplication,

$$ \vec{\nabla_x} L = \sum\limits_{i} (\vec{\nabla_y} L)^T J_{(i)} $$

we can clearly identify that each process has to perform a vector-Jacobian product (VJP), where $J_{(i)}$ is the Jacobian matrix such
that $J_{kj (i)} = \frac{\partial y_{k (i)}}{\partial x_j}$. Fortunately, this is a computation that Jax can perform very efficiently,
without the need to explicitly calculate the full Jacobian matrix by making use of the
[`jax.vjp`](https://jax.readthedocs.io/en/latest/_autosummary/jax.vjp.html) feature, saving us orders of magnitude of time and memory
requirements.



# Science Use Case

`multigrad` was developed to aid in parameter optimization for high-dimensional differentiable models applied to large datasets. It has enabled the scaling to cosmological volumes of a differentiable forward modeling pipeline which predicts galaxy properties based on a simulated dark matter density field (Diffmah: @Hearin:2021; Diffstar: @Alarcon:2023; DSPS: @Hearin:2023). Ongoing research is currently utilizing `multigrad` to optimize the parameters of this pipeline to reproduce observed galaxy properties (e.g. Beltz-Mohrmann et al. in prep.). More broadly, `multigrad` has useful applications for any scientific research that focuses on fitting high-dimensional models to large datasets and would benefit from computing parameter gradients in parallel.

# Acknowledgements



# References
