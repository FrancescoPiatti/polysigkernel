<h1 align='center'>Polynomial-based schems for Signature Kernels</h1>

Polysigkernel is a [JAX](https://github.com/google/jax) library for [signature kernels](https://arxiv.org/abs/2502.08470).

This package is inspired by the [sigkerax](https://github.com/crispitagorico/sigkerax) package by Cristopher Salvi.

## Installation

```bash
pip install git+https://github.com/FrancescoPiatti/polysigkernel.git
```

Requires Python 3.12+, JAX 0.4.23+.


## Quick example

```python
import jax
from polysigkernel import SigKernel

key1, key2 = jax.random.split(jax.random.PRNGKey(0), num=2)

batch_X, batch_Y, length_X, length_Y, channels = 20, 50, 100, 200, 10
X = 1e-1 * jax.random.normal(key1, shape=(batch_X, length_X, channels)).cumsum(axis=1)
Y = 1e-1 * jax.random.normal(key2, shape=(batch_Y, length_Y, channels)).cumsum(axis=1)

signature_kernel = SigKernel(order=5, static_kernel="linear")
signature_kernel_matrix = signature_kernel.kernel_matrix(X, Y)
```
