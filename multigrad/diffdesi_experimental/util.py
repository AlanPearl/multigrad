import numpy as np


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
