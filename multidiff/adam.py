"""
Copied directly from Matt's code here:
https://github.com/ArgonneCPAC/diff-ghmod-tools/blob/main/diff_ghmod_tools/adam.py  # noqa
"""
try:
    import tqdm

    def adam_trange(n, desc="Adam Gradient Descent Progress"):
        return tqdm.trange(n, desc=desc)
except ImportError:
    tqdm = None
    adam_trange = range
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


def _master_wrapper(params, logloss_and_grad_fn, data):
    if COMM is not None:
        COMM.bcast("compute", root=0)
        COMM.bcast(params, root=0)

    loss, grad = logloss_and_grad_fn(params, data)

    return loss, grad


def _adam_optimizer(params, fn, fn_data, n_steps, epsilon):
    opt_init, opt_update, get_params = jax_opt.adam(epsilon)
    opt_state = opt_init(params)

    for step in adam_trange(n_steps):
        _, grad = fn(params, *fn_data)
        opt_state = opt_update(step, grad, opt_state)

    return get_params(opt_state)


def run_adam(logloss_and_grad_fn, params, data, n_steps=100, epsilon=1e-3):
    """Run the adam optimizer on a loss function.

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

    Returns
    -------
    opt : array-like
        The optimal parameters.
    """

    if RANK == 0:
        fn = _master_wrapper
        fn_data = (logloss_and_grad_fn, data)

        params = _adam_optimizer(params, fn, fn_data, n_steps, epsilon)

        if COMM is not None:
            COMM.bcast("exit", root=0)
    else:
        # never get here if COMM is none
        while True:
            task = COMM.bcast(None, root=0)

            if task == "compute":
                params = COMM.bcast(None, root=0)
                # mpi bcast params
                logloss_and_grad_fn(params, data)
            elif task == "exit":
                break
            else:
                raise ValueError("task %s not recognized!" % task)

        params = None

    return params
