"""
"""
import math
from dataclasses import dataclass
from typing import Any, Hashable, NamedTuple, Optional, Union

import jax
import numpy as np
from jax import numpy as jnp

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    N_RANKS = COMM.Get_size()
except ImportError:
    MPI, COMM = None, None
    RANK = 0
    N_RANKS = 1

try:
    if RANK:
        raise ImportError("Only show progress bar on RANK=0 task")
    import tqdm

    def trange(n, desc=None):
        return tqdm.trange(n, desc=desc)
except ImportError:
    tqdm = None

    def trange(n, *_args, **_kwargs):
        return range(n)


class GradDescentResult(NamedTuple):
    loss: jnp.ndarray
    params: jnp.ndarray
    aux: Union[jnp.ndarray, list]


def distribute_data(data):
    fullsize = len(data)
    chunksize = math.ceil(fullsize / N_RANKS)
    start = chunksize * RANK
    stop = start + chunksize
    return data[start:stop]


def subrank_by_node():
    comm = MPI.COMM_WORLD
    # print("My rank is {0} of {1}".format(RANK, N_RANKS))

    node_name = MPI.Get_processor_name()
    # print("My rank is {0} and node number is {1}".format(rank, node_number))

    nodelist = comm.allgather(node_name)
    unique_nodelist = sorted(list(set(nodelist)))
    node_number = unique_nodelist.index(node_name)
    intra_node_id = len([i for i in nodelist[:RANK] if i == node_name])
    comm.Barrier()

    rankinfo = (RANK, intra_node_id, node_number)
    infolist = comm.allgather(rankinfo)
    sorted_infolist = sorted(infolist, key=lambda x: x[1])
    sorted_infolist = sorted(sorted_infolist, key=lambda x: x[2])
    pat = "Rank={0}, subrank={1}, node={2}"
    if RANK == 0:
        for item in sorted_infolist:
            print(pat.format(*item))

    sub_comm = comm.Split(color=node_number)
    sub_rank, sub_nranks = sub_comm.Get_rank(), sub_comm.Get_size()
    # print("My subrank is {0} of {1}".format(sub_rank, sub_nranks))

    sub_comm.Free()  # Sometimes this cleanup helps prevent memory leaks
    return sub_rank, sub_nranks, node_number, len(unique_nodelist)


def reduce_sum(value, root=None):
    """Returns the sum of `value` across all MPI processes

    Parameters
    ----------
    value : np.ndarray | float
        value input by each MPI process to be summed
    root : int, optional
        rank of the process to receive and sum the values,
        by default None (broadcast result to all ranks)

    Returns
    -------
    np.ndarray | float
        Sum of values given by each process
    """
    if COMM is None:
        return value
    return_to_scalar = not hasattr(value, "__len__")
    value = np.asarray(value)
    if root is None:
        # All-to-all sum
        total = np.empty_like(value)
        COMM.Allreduce(value, total, op=MPI.SUM)
    else:
        # All-to-root sum
        total = np.empty_like(value)
        COMM.Reduce(value, total, op=MPI.SUM, root=root)

    if return_to_scalar:
        total = total.tolist()
    return total


def simple_grad_descent(
    loss_func,
    guess,
    nsteps,
    learning_rate,
    loss_and_grad_func=None,
    grad_loss_func=None,
    has_aux=False,
    **kwargs,
):
    if loss_and_grad_func is None:
        if grad_loss_func is None:
            loss_and_grad_func = jax.value_and_grad(
                loss_func, has_aux=has_aux, **kwargs)
        else:

            def loss_and_grad_func(params):
                return (loss_func(params), grad_loss_func(params))

    # Create our mpi4jax token with a dummy broadcast
    def loopfunc(state, _x):
        grad, params = state
        params = jnp.asarray(params)

        # Evaluate the loss and gradient at given parameters
        (loss, grad), aux = loss_and_grad_func(params), None
        if has_aux:
            (loss, aux), grad = loss, grad
        y = (loss, params, aux)

        # Calculate the next parameters to evaluate (no need to broadcast this)
        params = params - learning_rate * grad
        # params = broadcast(params, root=0)
        state = grad, params
        return state, y

    # The below is equivalent to lax.scan without jitting
    # ===================================================
    initstate = (0.0, guess)
    loss, params, aux = [], [], []
    for x in trange(nsteps, desc="Simple Gradient Descent Progress"):
        initstate, y = loopfunc(initstate, x)
        loss.append(y[0])
        params.append(y[1])
        aux.append(y[2])
    loss = jnp.array(loss)
    params = jnp.array(params)
    try:
        aux = jnp.array(aux)
    except TypeError:
        pass
    ##################################

    return GradDescentResult(loss=loss, params=params, aux=aux)


# Prototypee for a PyTree (i.e., JAX-compatible object) that can
# calculate a model prediction and loss, distributed over MPI
@jax.tree_util.register_pytree_node_class
@dataclass
class MultiDiffOnePointModel:
    dynamic_data: Any = None
    static_data: tuple[Hashable, ...] = ()
    loss_func_has_aux: bool = False
    sumstats_func_has_aux: bool = False
    root: Optional[int] = None

    def calc_partial_sumstats_from_params(self, params):
        """Custom method to map parameters to summary statistics"""
        raise NotImplementedError(
            "Subclass must implement `calc_partial_sumstats_func_from_params`"
        )

    def calc_loss_from_sumstats(self, sumstats, sumstats_aux=None):
        """Custom method to map summary statistics to loss"""
        raise NotImplementedError(
            "Subclass must implement `calc_loss_func_from_sumstats`"
        )

    # NOTE: Never jit this method because it uses mpi4py
    def run_grad_descent(self, guess: jnp.ndarray,
                         nsteps=100, learning_rate=1e-3):
        return simple_grad_descent(
            self.calc_loss_from_params,
            guess=guess,
            nsteps=nsteps,
            learning_rate=learning_rate,
            loss_and_grad_func=self.calc_loss_and_grad_from_params,
            has_aux=self.loss_func_has_aux,
        )

    # NOTE: Never jit this method because it uses mpi4py
    def run_adam(self, guess: jnp.ndarray,
                 nsteps=100, epsilon=1e-3):
        from .adam import run_adam
        final_params = run_adam(
            lambda x, _: self.calc_loss_and_grad_from_params(x),
            params=guess, data=None,
            n_steps=nsteps, epsilon=epsilon
        )
        return jnp.asarray(COMM.bcast(final_params, root=0))

    # NOTE: Never jit this method because it uses mpi4py
    def run_bfgs(self, guess: jnp.ndarray, maxsteps=100):
        import scipy.optimize

        pbar = trange(maxsteps, desc="BFGS Gradient Descent Progress")

        def callback(*_args, **_kwargs):
            if hasattr(pbar, "update"):
                pbar.update()

        return scipy.optimize.minimize(
            self.calc_loss_and_grad_from_params, x0=guess, callback=callback,
            method="L-BFGS-B", jac=True, options=dict(maxiter=maxsteps))

    # No need to jit __post_init__, since it is only called once
    # TODO: might want to completely get rid of _jac_sumstats_from_params
    # since we don't use it due to its large memory usage
    def __post_init__(self):
        # Create auto-diff functions needed for gradient descent
        self._jac_sumstats_from_params = jax.jit(
            jax.jacobian(
                self.calc_partial_sumstats_from_params,
                has_aux=self.sumstats_func_has_aux,
            )
        )
        self._grad_loss_from_sumstats = jax.jit(
            jax.grad(self.calc_loss_from_sumstats,
                     has_aux=self.loss_func_has_aux)
        )

    # sumstats functions
    # NOTE: Never jit this method because it uses mpi4py (when total=True)
    def calc_sumstats_from_params(
        self, params: jnp.ndarray, total=True
    ) -> Union[jnp.ndarray, tuple[jnp.ndarray, Any]]:
        result, aux = self.calc_partial_sumstats_from_params(params), None
        if self.sumstats_func_has_aux:
            result, aux = result
        if total:
            result = jnp.asarray(reduce_sum(result, root=self.root))
        result = (result, aux) if self.sumstats_func_has_aux else result
        return result

    # loss functions
    # NOTE: Never jit this method because it uses mpi4py
    def calc_loss_from_params(
        self, params: jnp.ndarray
    ) -> Union[jnp.ndarray, tuple[jnp.ndarray, Any]]:
        sumstats = self.calc_sumstats_from_params(params)
        if not self.sumstats_func_has_aux:
            sumstats = (sumstats,)
        return self.calc_loss_from_sumstats(*sumstats)

    @jax.jit
    def calc_dloss_dsumstats(
        self, sumstats: jnp.ndarray, sumstats_aux=None
    ) -> Union[jnp.ndarray, tuple[jnp.ndarray, Any]]:
        sumstats = jnp.asarray(sumstats)
        args = (sumstats, sumstats_aux) if self.sumstats_func_has_aux else (
            sumstats,)
        return self._grad_loss_from_sumstats(*args)

    # NOTE: Never jit this method because it uses mpi4py
    def calc_loss_and_grad_from_params(
        self, params: jnp.ndarray
    ) -> tuple[Union[jnp.ndarray, tuple], jnp.ndarray]:
        params = jnp.asarray(params)

        # Calculate sumstats AND save VJP func to perform chain rule later
        vjp_results = jax.vjp(
            self.calc_partial_sumstats_from_params, params,
            has_aux=self.sumstats_func_has_aux)
        sumstats, vjp_func = vjp_results[:2]
        sumstats = jnp.asarray(reduce_sum(sumstats, root=None))
        args = (sumstats, *vjp_results[2:])

        # Calculate dloss_dsumstats for chain rule. Should be inexpensive
        dloss_dsumstats = self.calc_dloss_dsumstats(*args)
        if self.loss_func_has_aux:
            dloss_dsumstats = dloss_dsumstats[0]

        # Use VJP for the chain rule dL/dp[i] = sum(dL/ds[j] * ds[j]/dp[i])
        dloss_dparams = jnp.asarray(reduce_sum(
            vjp_func(dloss_dsumstats)[0], root=self.root))

        # Return ((loss, *aux_if_any), dloss_dparams)
        return self.calc_loss_from_sumstats(*args), dloss_dparams

    # Don't use the following two methods - they are too memory intensive
    # ===================================================================
    # NOTE: Never jit this method because it uses mpi4py (when total=True)
    def calc_dsumstats_dparams(
        self, params: jnp.ndarray, total=True
    ) -> Union[jnp.ndarray, tuple[jnp.ndarray, Any]]:
        params = jnp.asarray(params)
        result, aux = self._jac_sumstats_from_params(params), None
        if self.sumstats_func_has_aux:
            result, aux = result
        if total:
            result = jnp.asarray(reduce_sum(result, root=self.root))
        if self.sumstats_func_has_aux:
            result = (result, aux)
        return result

    # NOTE: Never jit this method because it uses mpi4py
    def calc_dloss_dparams(
        self, params: jnp.ndarray
    ) -> Union[jnp.ndarray, tuple[jnp.ndarray, Any]]:
        params = jnp.asarray(params)
        sumstats = self.calc_sumstats_from_params(params)
        dsumstats_dparams = self.calc_dsumstats_dparams(params)
        if self.sumstats_func_has_aux:
            dsumstats_dparams = dsumstats_dparams[0]
        else:
            sumstats = (sumstats,)

        dloss_dsumstats, aux = self.calc_dloss_dsumstats(*sumstats), None
        if self.loss_func_has_aux:
            dloss_dsumstats, aux = dloss_dsumstats

        # Chain rule
        result = jnp.sum(dloss_dsumstats[:, None] * dsumstats_dparams, axis=0)
        if self.loss_func_has_aux:
            result = result, aux
        return result

    # JAX compatibility functions below
    # =================================
    def tree_flatten(self) -> tuple[tuple, dict]:
        children = (self.dynamic_data,)  # arrays / dynamic values
        aux_data = dict(  # static values
            static_data=self.static_data,
            loss_func_has_aux=self.loss_func_has_aux,
            sumstats_func_has_aux=self.sumstats_func_has_aux,
            root=self.root,
        )
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data: dict, children: tuple):
        return cls(*children, **aux_data)
