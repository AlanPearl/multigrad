"""
Modified version of Matt's code taken from:
https://github.com/ArgonneCPAC/diff-ghmod-tools/blob/main/diff_ghmod_tools/adam.py  # noqa
"""
try:
    from tqdm import auto as tqdm
except ImportError:
    tqdm = None

import jax.random
import jax.numpy as jnp
from jax.example_libraries import optimizers as jax_opt

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    N_RANKS = COMM.Get_size()
except ImportError:
    COMM = None
    RANK = 0
    N_RANKS = 1


def trange_no_tqdm(n, desc=None):
    return range(n)


def trange_with_tqdm(n, desc="Adam Gradient Descent Progress"):
    return tqdm.trange(n, desc=desc)


adam_trange = trange_no_tqdm if tqdm is None else trange_with_tqdm


def _master_wrapper(params, logloss_and_grad_fn, data, randkey=None):
    if COMM is not None:
        COMM.bcast("compute", root=0)
        COMM.bcast(params, root=0)

    kwargs = {}
    if randkey is not None:
        kwargs["randkey"] = randkey
    loss, grad = logloss_and_grad_fn(params, data, **kwargs)

    return loss, grad


def _adam_optimizer(params, fn, fn_data, n_steps, epsilon, randkey=None):
    kwargs = {}
    if randkey is not None:
        kwargs["randkey"] = randkey

    opt_init, opt_update, get_params = jax_opt.adam(epsilon)
    opt_state = opt_init(params)

    for step in adam_trange(n_steps):
        if randkey is not None:
            randkey = gen_new_key(randkey)
            kwargs["randkey"] = randkey
        _, grad = fn(params, *fn_data, **kwargs)
        opt_state = opt_update(step, grad, opt_state)

    return get_params(opt_state)


def run_adam(logloss_and_grad_fn, params, data, n_steps=100, epsilon=1e-3,
             randkey=None):
    """Run the adam optimizer on a loss function with a custom gradient.

    Parameters
    ----------
    logloss_and_grad_fn : callable
        Function with signature logloss_and_grad_fn(params, data) that returns
        a 2-tuple of the loss and the gradient of the loss.
    params : array-like
        The starting parameters.
    data : anything
        The data passed to logloss_and_grad_fn
    n_steps : int
        The number of steps to take.
    epsilon : float
        The adam learning rate.
    randkey : int | PRNG Key
        If given, a new PRNG Key will be generated at each iteration and be
        passed to `logloss_and_grad_fn` under the "randkey" kwarg

    Returns
    -------
    opt : array-like
        The optimal parameters.
    """
    kwargs = {}
    if randkey is not None:
        randkey = init_randkey(randkey)
        kwargs["randkey"] = randkey

    if RANK == 0:
        fn = _master_wrapper
        fn_data = (logloss_and_grad_fn, data)

        params = _adam_optimizer(params, fn, fn_data, n_steps, epsilon,
                                 randkey=randkey)

        if COMM is not None:
            COMM.bcast("exit", root=0)
    else:
        # never get here if COMM is none
        while True:
            task = COMM.bcast(None, root=0)

            if task == "compute":
                params = COMM.bcast(None, root=0)
                # mpi bcast params
                if randkey is not None:
                    randkey = gen_new_key(randkey)
                    kwargs["randkey"] = randkey
                logloss_and_grad_fn(params, data, **kwargs)
            elif task == "exit":
                break
            else:
                raise ValueError("task %s not recognized!" % task)

        params = None

    return params


def init_randkey(randkey):
    """Check that randkey is a PRNG key or create one from an int"""
    if isinstance(randkey, int):
        randkey = jax.random.key(randkey)
    else:
        msg = f"Invalid {type(randkey)=}: Must be int or PRNG Key"
        assert hasattr(randkey, "dtype"), msg
        assert jnp.issubdtype(randkey.dtype, jax.dtypes.prng_key), msg

    return randkey


@jax.jit
def gen_new_key(randkey):
    """Split PRNG key to generate a new one"""
    return jax.random.split(randkey, 1)[0]
