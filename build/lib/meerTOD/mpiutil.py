"""
This module is adapted from https://github.com/radiocosmology/caput/blob/master/caput/mpiutil.py
"""
import numpy as np
import logging

# Add near the top with other CPU affinity code
import os

rank = 0
size = 1
_comm = None
world = None
rank0 = True

logger = logging.getLogger(__name__)

# import mpi4py.rc
# mpi4py.rc.initialize = False  # Disables auto-initialization of MPI

# Add MPI initialization control
_mpi_initialized = False

def init_mpi():
    """Initialize MPI once at module import"""
    global _mpi_initialized
    if not _mpi_initialized:
        from mpi4py import MPI
        if not MPI.Is_initialized():
            MPI.Init()
        _mpi_initialized = True

# Initialize MPI when module is imported
init_mpi()

from mpi4py import MPI
_comm = MPI.COMM_WORLD
world = _comm
rank = _comm.Get_rank()
size = _comm.Get_size()

if _comm is not None and size > 1:
    logger.debug("Starting MPI rank=%i [size=%i]", rank, size)

rank0 = rank == 0


def partition_list(full_list, i, n, method="con"):
    """
    Partition a list into `n` pieces. Return the `i`th partition.
    """

    def _partition(N, n, i):
        # If partiion `N` numbers into `n` pieces,
        # return the start and stop of the `i` th piece
        base = N // n
        rem = N % n
        num_lst = rem * [base + 1] + (n - rem) * [base]
        cum_num_lst = np.cumsum([0] + num_lst)

        return cum_num_lst[i], cum_num_lst[i + 1]

    N = len(full_list)
    start, stop = _partition(N, n, i)

    if method == "con":
        return full_list[start:stop]
    elif method == "alt":
        return full_list[i::n]
    elif method == "rand":
        choices = np.random.permutation(N)[start:stop]
        return [full_list[i] for i in choices]
    else:
        raise ValueError("Unknown partition method %s" % method)

def partition_list_mpi(full_list, method="con", comm=_comm):
    """
    Return the partition of a list specific to the current MPI process.
    """
    if comm is not None:
        rank = comm.rank
        size = comm.size

    return partition_list(full_list, rank, size, method=method)

def parallel_map_gather(func, glist, multi_inputs=False, root=None, method="con", comm=_comm):
    """
    Apply a parallel map using MPI.
    Should be called collectively on the same list. All ranks return the full
    set of results.
    Parameters
    ----------
    func : function
        Function to apply.
    glist : list
        List of map over. Must be globally defined.
    root : None or Integer
        Which process should gather the results, all processes will gather the results if None.
    method: str
        How to split `glist` to each process, can be 'con': continuously, 'alt': alternatively, 'rand': randomly. Default is 'con'.
    comm : MPI communicator
        MPI communicator that array is distributed over. Default is the gobal _comm.
    Returns
    -------
    results : list
        Global list of results.
    """

    # Synchronize
    barrier(comm=comm)

    # If we're only on a single node, then just perform without MPI
    if comm is None or comm.size == 1:
        if multi_inputs:
            return [func(*item) for item in glist]
        else:
            return [func(item) for item in glist]

    # Pair up each list item with its position.
    zlist = list(enumerate(glist))

    # Partition list based on MPI rank
    llist = partition_list_mpi(zlist, method=method, comm=comm)

    # Operate on sublist
    if multi_inputs:
        flist = [(ind, func(*item)) for ind, item in llist]
    else:
        flist = [(ind, func(item)) for ind, item in llist]

    barrier(comm=comm)

    rlist = None
    if root is None:
        # Gather all results onto all ranks
        rlist = comm.allgather(flist)
    else:
        # Gather all results onto the specified rank
        rlist = comm.gather(flist, root=root)

    if rlist is not None:
        # Flatten the list of results
        flatlist = [item for sublist in rlist for item in sublist]

        # Sort into original order
        sortlist = sorted(flatlist, key=(lambda item: item[0]))

        # Synchronize
        barrier(comm=comm)

        # Extract the return values into a list
        return [item for ind, item in sortlist]
    else:
        return None

def parallel_jobs_no_gather_no_return(func, glist, method="con", comm=_comm):
    """
    Apply a parallel map using MPI.
    Should be called collectively on the same list. All ranks return the full
    set of results.
    Parameters
    ----------
    func : function
        Function to apply.
    glist : zipped list
        List of map over. Must be globally defined.
    root : None or Integer
        Which process should gather the results, all processes will gather the results if None.
    method: str
        How to split `glist` to each process, can be 'con': continuously, 'alt': alternatively, 'rand': randomly. Default is 'con'.
    comm : MPI communicator
        MPI communicator that array is distributed over. Default is the gobal _comm.
    Returns
    -------
    results : list
        Global list of results.
    """

    # Synchronize
    barrier(comm=comm)

    # If we're only on a single node, then just perform without MPI
    if comm is None or comm.size == 1:
        return [func(item) for item in glist]


    # Partition list based on MPI rank
    llist = partition_list_mpi(glist, method=method, comm=comm)

    # Operate on sublist
    for zipped_item in llist:
        func(zip(*zipped_item))

    # Synchronize
    barrier(comm=comm)
    return None

def barrier(comm=_comm):
    """
    Synchronize all MPI processes.
    """
    if comm is not None and comm.size > 1:
        comm.Barrier()