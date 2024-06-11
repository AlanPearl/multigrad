"""
"""
from scipy.stats import qmc
import math
from dataclasses import dataclass
from typing import Any, NamedTuple, Union

import jax
import numpy as np
from jax import numpy as jnp

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    N_RANKS = COMM.Get_size()
    Comm = MPI.Comm
    Intracomm = MPI.Intracomm
except ImportError:
    MPI = COMM = None
    Comm = Intracomm = type(None)
    RANK = 0
    N_RANKS = 1

try:
    if RANK:
        raise ImportError("Only show progress bar on RANK=0 task")
    from tqdm import auto as tqdm
except ImportError:
    tqdm = None


class GradDescentResult(NamedTuple):
    loss: jnp.ndarray
    params: jnp.ndarray
    aux: Union[jnp.ndarray, list]


def trange_no_tqdm(n, desc=None):
    return range(n)


def trange_with_tqdm(n, desc=None):
    return tqdm.trange(n, desc=desc)


trange = trange_no_tqdm if tqdm is None else trange_with_tqdm


def latin_hypercube_sampler(xmin, xmax, n_dim, num_evaluations,
                            seed=None, optimization=None):
    xmin = np.zeros(n_dim) + xmin
    xmax = np.zeros(n_dim) + xmax
    sampler = qmc.LatinHypercube(n_dim, seed=seed, optimization=optimization)
    unit_hypercube = sampler.random(num_evaluations)
    return qmc.scale(unit_hypercube, xmin, xmax)


def sort_all_by_ultimate_top_dump(ultimate_dump,
                                  arrays_to_sort=[],
                                  arrays_to_sort_and_reindex=[]):
    ultimate_top_dump = find_ultimate_top_indices(ultimate_dump)
    argsort = np.argsort(ultimate_top_dump)
    argsort2 = np.argsort(argsort)

    sorted_arrays = [np.asarray(x)[argsort] for x in arrays_to_sort]
    reindexed_arrays = [sort_and_reindex(x, argsort, argsort2)
                        for x in arrays_to_sort_and_reindex]

    return sorted_arrays, reindexed_arrays


def find_ultimate_top_indices(indices):
    indices = np.array(indices)
    recursion_count = 0
    max_recursion = 50
    while np.any(indices != indices[indices]):
        recursion_count += 1
        if recursion_count > max_recursion:
            raise RecursionError(
                f"Host search hasn't finished after {max_recursion} steps")
        indices = indices[indices]
    return indices


def sort_and_reindex(indices, argsort=None, argsort2=None):
    indices = np.asarray(indices)
    argsort = np.argsort(indices) if argsort is None else argsort
    argsort2 = np.argsort(argsort) if argsort2 is None else argsort2
    return argsort2[indices][argsort]


def scatter_nd(array, axis=0, comm=COMM, root=0):
    """Scatter n-dimensional array from root to all ranks"""
    ans: np.ndarray = np.array([])
    if comm.rank == root:
        splits = np.array_split(array, comm.size, axis=axis)
        for i in range(comm.size):
            if i == root:
                ans = splits[i]
            else:
                comm.send(splits[i], dest=i)
    else:
        ans = comm.recv(source=root)
    return ans


def split_subcomms_by_node(comm=COMM):
    """
    Split comm into sub-comms (grouped by nodes)

    Parameters
    ----------
    comm : MPI.Comm, optional
        Specify a sub-communicator to split into sub-sub-communicators

    Returns
    -------
    subcomm: MPI.Comm
        The sub-comm that now controls this process
    num_groups: int
        The number of groups of subcomms (= number of nodes)
    group_rank: int
        The rank of this group (0 <= subcomm_rank < num_subcomms)
    """
    assert MPI is not None, "Cannot split subcomms without mpi4py"
    node_name = MPI.Get_processor_name()

    nodelist = comm.allgather(node_name)
    unique_nodelist = sorted(list(set(nodelist)))
    node_number = unique_nodelist.index(node_name)
    intra_node_id = len([i for i in nodelist[:RANK] if i == node_name])
    comm.Barrier()

    rankinfo = (RANK, intra_node_id, node_number)
    infolist = comm.allgather(rankinfo)
    sorted_infolist = sorted(infolist, key=lambda x: x[1])
    sorted_infolist = sorted(sorted_infolist, key=lambda x: x[2])

    subcomm = comm.Split(color=node_number)
    subcomm.Set_name(f"{comm.name}.{node_number}".replace(
        "MPI_COMM_WORLD.", ""))

    # subcomm.Free()  # Sometimes this cleanup helps prevent memory leaks
    return subcomm, len(unique_nodelist), node_number


def split_subcomms(num_groups=None, ranks_per_group=None,
                   comm=COMM):
    """
    Split comm into sub-comms (not grouped by nodes)

    Parameters
    ----------
    num_groups : int, optional
        Specify the number of evenly divided groups of subcomms
    ranks_per_group : list[int], optional
        Specify the number of ranks given to each sub-comm
    comm : MPI.Comm, optional
        Specify a sub-communicator to split into sub-sub-communicators

    Returns
    -------
    subcomm: MPI.Comm
        The sub-comm that now controls this process
    num_groups: int
        The number of groups of subcomms (same as input if not None)
    group_rank: int
        The rank of this group (0 <= subcomm_rank < num_subcomms)
    """
    assert comm is not None, "Cannot split subcomms without mpi4py"
    main_msg = "Specify either num_subcomms OR ranks_per_subcomm"
    sumrps_msg = "The sum of ranks_per_subcomm must equal comm.size"
    nsub_msg = "Cannot create more subcomms than there are ranks"
    if num_groups is not None:
        assert ranks_per_group is None, main_msg
        assert (comm.size >= num_groups), nsub_msg
        num_groups = int(num_groups)
        subnames = (np.ones(math.ceil(comm.size / num_groups))[None, :]
                    * np.arange(num_groups)[:, None])[:comm.size]
        subnames = subnames.ravel().astype(int)
    else:
        assert ranks_per_group is not None, main_msg
        assert sum(ranks_per_group) == comm.size, sumrps_msg
        num_groups = len(ranks_per_group)
        subnames = np.repeat(np.arange(num_groups), ranks_per_group)

    subname = str(np.array_split(subnames, comm.size)[comm.rank][0])

    nodelist = comm.allgather(subname)
    unique_nodelist = sorted(list(set(nodelist)))
    node_number = unique_nodelist.index(subname)
    intra_node_id = len([i for i in nodelist[:comm.rank] if i == subname])
    comm.Barrier()

    rankinfo = (comm.rank, intra_node_id, node_number)
    infolist = comm.allgather(rankinfo)
    sorted_infolist = sorted(infolist, key=lambda x: x[1])
    sorted_infolist = sorted(sorted_infolist, key=lambda x: x[2])

    sub_comm = comm.Split(color=node_number)
    sub_comm.Set_name(f"{comm.name}.{subname}".replace(
        "MPI_COMM_WORLD.", ""))

    # sub_comm.Free()  # Sometimes this cleanup helps prevent memory leaks
    return sub_comm, num_groups, int(subname)


def reduce_sum(value, root=None, comm=COMM):
    """Returns the sum of `value` across all MPI processes

    Parameters
    ----------
    value : np.ndarray | float
        value input by each MPI process to be summed
    root : int, optional
        rank of the process to receive and sum the values,
        by default None (broadcast result to all ranks)
    comm : MPI.Intracomm (default = MPI.COMM_WORLD)
        option to pass a sub-communicator in case the operation
        is not performed by all MPI ranks

    Returns
    -------
    np.ndarray | float
        Sum of values given by each process
    """
    if comm is None:
        return value
    return_to_scalar = not hasattr(value, "__len__")
    value = np.asarray(value)
    if root is None:
        # All-to-all sum
        total = np.empty_like(value)
        comm.Allreduce(value, total, op=MPI.SUM)
    else:
        # All-to-root sum
        total = np.empty_like(value)
        comm.Reduce(value, total, op=MPI.SUM, root=root)

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
            def explicit_loss_and_grad_func(params):
                return (loss_func(params), grad_loss_func(params))
            loss_and_grad_func = explicit_loss_and_grad_func

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
    if has_aux:
        try:
            aux = jnp.array(aux)
        except TypeError:
            pass
    ##################################

    return GradDescentResult(loss=loss, params=params, aux=aux)


@dataclass
class MultiDiffOnePointModel:
    """
    ALlows differentiable one-point calculations to be performed on separate
    MPI ranks, and automatically sums over each rank controlled by the comm
    """
    dynamic_data: Any = None
    comm: Any = None
    loss_func_has_aux: bool = False
    sumstats_func_has_aux: bool = False

    def calc_partial_sumstats_from_params(self, params, randkey=None):
        """Custom method to map parameters to summary statistics"""
        raise NotImplementedError(
            "Subclass must implement `calc_partial_sumstats_func_from_params`"
        )

    def calc_loss_from_sumstats(self, sumstats, sumstats_aux=None,
                                randkey=None):
        """Custom method to map summary statistics to loss"""
        raise NotImplementedError(
            "Subclass must implement `calc_loss_func_from_sumstats`"
        )

    # NOTE: Never jit this method because it uses mpi4py
    def run_simple_grad_descent(self: Any, guess,
                                nsteps=100, learning_rate=1e-3):
        return simple_grad_descent(
            None,
            guess=guess,
            nsteps=nsteps,
            learning_rate=learning_rate,
            loss_and_grad_func=self.calc_loss_and_grad_from_params,
            has_aux=False,
        )

    # NOTE: Never jit this method because it uses mpi4py
    def run_adam(self: Any, guess,
                 nsteps=100, epsilon=1e-3, randkey=None, _comm=None):
        from .adam import run_adam
        guess = jnp.asarray(guess)
        final_params = run_adam(
            lambda x, _, **kw: self.calc_loss_and_grad_from_params(x, **kw),
            params=guess, data=None,
            n_steps=nsteps, epsilon=epsilon, randkey=randkey
        )

        comm = self.comm if _comm is None else _comm
        return jnp.asarray(comm.bcast(final_params, root=0))

    # NOTE: Never jit this method because it uses mpi4py
    def run_bfgs(self: Any, guess, maxsteps=100, randkey=None):
        import scipy.optimize

        pbar = trange(maxsteps, desc="BFGS Gradient Descent Progress")

        def callback(*_args, **_kwargs):
            if hasattr(pbar, "update"):
                pbar.update()  # type: ignore

        return scipy.optimize.minimize(
            self.calc_loss_and_grad_from_params, x0=guess, callback=callback,
            method="L-BFGS-B", jac=True, options=dict(maxiter=maxsteps),
            args=(randkey,))

    def run_lhs_param_scan(self, xmins, xmaxs, n_dim,
                           num_evaluations, seed=None):
        params = latin_hypercube_sampler(xmins, xmaxs, n_dim,
                                         num_evaluations, seed=seed)
        sumstats = [self.calc_sumstats_from_params(x) for x in params]
        losses = [self.calc_loss_from_sumstats(x) for x in sumstats]
        return params, np.array(sumstats), np.array(losses)

    def __post_init__(self):
        if self.comm is None:
            self.comm = COMM
        # Create auto-diff functions needed for gradient descent
        self._grad_loss_from_sumstats = jax.grad(
            self.calc_loss_from_sumstats,
            has_aux=self.loss_func_has_aux)

    # sumstats functions
    # NOTE: Never jit this method because it uses mpi4py (when total=True)
    def calc_sumstats_from_params(
            self, params, total=True, randkey=None):
        kwargs = {} if randkey is None else {"randkey": randkey}
        result, aux = self.calc_partial_sumstats_from_params(
            params, **kwargs), None
        if self.sumstats_func_has_aux:
            result, aux = result
        if total:
            result = jnp.asarray(reduce_sum(result, comm=self.comm))
        result = (result, aux) if self.sumstats_func_has_aux else result
        return result

    # loss functions
    # NOTE: Never jit this method because it uses mpi4py
    def calc_loss_from_params(
            self, params, randkey=None):
        kwargs = {} if randkey is None else {"randkey": randkey}
        sumstats = self.calc_sumstats_from_params(params, **kwargs)
        if not self.sumstats_func_has_aux:
            sumstats = (sumstats,)
        return self.calc_loss_from_sumstats(*sumstats, **kwargs)

    def calc_dloss_dsumstats(
            self, sumstats, sumstats_aux=None, randkey=None):
        kwargs = {} if randkey is None else {"randkey": randkey}
        sumstats = jnp.asarray(sumstats)
        args = (sumstats, sumstats_aux) if self.sumstats_func_has_aux else (
            sumstats,)
        return self._grad_loss_from_sumstats(*args, **kwargs)

    # NOTE: Never jit this method because it uses mpi4py
    def calc_dloss_dparams(self, params, randkey=None):
        return self._vjp(params, randkey=randkey, include_loss=False)

    # NOTE: Never jit this method because it uses mpi4py
    def calc_loss_and_grad_from_params(self, params, randkey=None):
        """
        `calc_loss_and_grad_from_params(x)` returns the equivalent of
        `(calc_loss_from_params(x), calc_dloss_dparams(x))` but it is
        significantly cheaper than calling them separately
        """
        return self._vjp(params, randkey=randkey, include_loss=True)

    # NOTE: Never jit this method because it uses mpi4py
    def _vjp(
        self, params, randkey=None, include_loss=True
    ):
        kwargs = {} if randkey is None else {"randkey": randkey}
        params = jnp.asarray(params)

        def sumstats_func(params):
            return self.calc_partial_sumstats_from_params(params, **kwargs)

        # Calculate sumstats AND save VJP func to perform chain rule later
        vjp_results = jax.vjp(
            sumstats_func, params,
            has_aux=self.sumstats_func_has_aux)  # type: ignore
        sumstats, vjp_func = vjp_results[:2]
        sumstats = jnp.asarray(reduce_sum(sumstats, comm=self.comm))
        args = (sumstats, *vjp_results[2:])

        # Calculate dloss_dsumstats for chain rule. Should be inexpensive
        dloss_dsumstats = self.calc_dloss_dsumstats(*args, **kwargs)
        if self.loss_func_has_aux:
            dloss_dsumstats = dloss_dsumstats[0]

        # Use VJP for the chain rule dL/dp[i] = sum(dL/ds[j] * ds[j]/dp[i])
        dloss_dparams = jnp.asarray(reduce_sum(
            vjp_func(dloss_dsumstats)[0], comm=self.comm))

        if include_loss:
            # Return (loss_and_aux, dloss_dparams)
            return self.calc_loss_from_sumstats(*args, **kwargs), dloss_dparams
        else:
            return dloss_dparams

    def __hash__(self):
        return hash((self.comm.name, self.calc_loss_from_sumstats))

    def __eq__(self, other):
        return isinstance(other, MultiDiffGroup) and self is other


@dataclass
class MultiDiffGroup:
    """
    Allows different MultiDiffOnePointModels to simultaneously perform their
    calc_loss_and_grad_from_params method. The results are summed.
    """
    models: Union[tuple[MultiDiffOnePointModel, ...], MultiDiffOnePointModel]
    main_comm: Any = None

    def __post_init__(self):
        if self.main_comm is None:
            self.main_comm = COMM
        if isinstance(self.models, MultiDiffOnePointModel):
            self.models = (self.models,)
        assert isinstance(self.models[0], MultiDiffOnePointModel)

    # NOTE: Never jit this method because it uses mpi4py
    def calc_loss_and_grad_from_params(self, params):
        loss, grad = [], []
        for model in self.models:
            res = model.calc_loss_and_grad_from_params(params)
            loss.append(res[0]*0 if model.comm.rank else res[0])
            grad.append(res[1]*0 if model.comm.rank else res[1])
        loss = jnp.concatenate(jnp.array(self.main_comm.allgather(loss)))
        grad = jnp.concatenate(jnp.array(self.main_comm.allgather(grad)))
        return loss.sum(), grad.sum(axis=0)

    # NOTE: Never jit this method because it uses mpi4py
    def run_simple_grad_descent(self, guess,
                                nsteps=100, learning_rate=1e-3):
        return MultiDiffOnePointModel.run_simple_grad_descent(
            self, guess, nsteps, learning_rate)

    # NOTE: Never jit this method because it uses mpi4py
    def run_bfgs(self, guess, maxsteps=100):
        return MultiDiffOnePointModel.run_bfgs(self, guess, maxsteps)

    # NOTE: Never jit this method because it uses mpi4py
    def run_adam(self, guess,
                 nsteps=100, epsilon=1e-3, randkey=None):
        return MultiDiffOnePointModel.run_adam(
            self, guess, nsteps, epsilon, randkey, self.main_comm)

    def __hash__(self):
        if isinstance(self.models, MultiDiffOnePointModel):
            self.models = (self.models,)
        return hash((self.main_comm.name, self.models[0]))

    def __eq__(self, other):
        return isinstance(other, MultiDiffGroup) and self is other
