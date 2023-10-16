"""mpiexec -n 10 python parallel_sum_mpi4py_demo.py
"""
import numpy as np
from mpi4py import MPI

if __name__ == "__main__":
    COMM = MPI.COMM_WORLD
    rank, nranks = COMM.Get_rank(), COMM.Get_size()

    arr = np.zeros(5) + rank
    arr_tot = np.zeros_like(arr)

    COMM.Reduce(arr, arr_tot, MPI.SUM, root=0)
    print("rank = {0}, arr_tot={1}".format(rank, arr_tot))
