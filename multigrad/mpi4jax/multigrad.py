"""mpiexec -n 10 python -m multigrad
"""
import math

import pandas as pd
import jax
from jax import numpy as jnp

import mpi4jax
from mpi4py import MPI
COMM = MPI.COMM_WORLD


# NOTE: This assumes the entire data is initially loaded in memory
# TODO: A separate function will be needed for loading data from file(s)
# jax.jit
def distribute_data(data):
    rank, nranks = COMM.Get_rank(), COMM.Get_size()
    fullsize = len(data)
    chunksize = math.ceil(fullsize / nranks)
    start = chunksize * rank
    stop = start + chunksize
    return data[start:stop]


# jax.jit
def reduce_sum(partial_value):
    ans, token = mpi4jax.allreduce(partial_value, op=MPI.SUM, comm=COMM)
    return ans


# partial(jax.jit, static_argnums=0)
def simple_grad_descent(data_dict, loss_and_grad_func, guess,
                        learning_rate=0.01, nsteps=100):
    rank = COMM.Get_rank()

    # Create our mpi4jax token with a dummy broadcast
    def loopfunc(state, _x):
        grad, params = state

        # Evaluate the loss and gradient at given parameters
        loss, grad = loss_and_grad_func(data_dict, params)
        # NOTE: We have to sum grad over all processes, probably due to a bug in mpi4jax
        grad = reduce_sum(grad)
        y = (loss, params)

        # Calculate the next parameters to evaluate
        if not rank:
            # In this case, we could just do this on all processes - but let's
            # try to figure out how to only do it on rank 0 then broadcast it
            params = params - learning_rate * grad
        params = mpi4jax.bcast(params, root=0, comm=COMM)[0]
        state = grad, params
        return state, y

    initstate = (0.0, guess)
    iterations = jax.lax.scan(
        loopfunc, initstate, jnp.arange(nsteps), nsteps)[1]

    loss, params = iterations
    return pd.DataFrame(dict(loss=loss, params=params))


if __name__ == "__main__":
    # Maybe somehow perform gradient descent if directly executed?
    pass
