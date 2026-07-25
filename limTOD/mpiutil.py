"""
This module is adapted from https://github.com/radiocosmology/caput/blob/master/caput/mpiutil.py
"""

import logging
import os
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    from mpi4py.MPI import Comm

rank: int = 0
size: int = 1
_comm: Optional["Comm"] = None
# `world` is accessed unguarded (e.g. `mpiutil.world.bcast(...)`) inside
# `size > 1` blocks that mypy cannot connect to non-None-ness, so it is
# typed Any rather than Optional["Comm"].
world: Any = None
rank0: bool = True

logger = logging.getLogger(__name__)

# import mpi4py.rc
# mpi4py.rc.initialize = False  # Disables auto-initialization of MPI

# Add MPI initialization control
_mpi_initialized = False


def init_mpi() -> None:
    """Initialize MPI once at module import"""
    global _mpi_initialized
    if not _mpi_initialized:
        from mpi4py import MPI

        if not MPI.Is_initialized():
            MPI.Init()
        _mpi_initialized = True


# Environment variables set by common MPI launchers (mpirun/mpiexec/srun),
# used to detect the "launched under MPI but mpi4py missing" misconfiguration.
_MPI_LAUNCHER_SIZE_VARS = (
    "OMPI_COMM_WORLD_SIZE",  # Open MPI
    "PMI_SIZE",              # MPICH / Intel MPI (Hydra)
    "MV2_COMM_WORLD_SIZE",   # MVAPICH2
    "SLURM_NTASKS",          # Slurm srun
)


def _detect_mpi_launcher() -> Optional[str]:
    """Return "VAR=value" if an MPI launcher with >1 tasks is detected, else None."""
    for var in _MPI_LAUNCHER_SIZE_VARS:
        value = os.environ.get(var)
        if value is None:
            continue
        try:
            n_tasks = int(value)
        except ValueError:
            continue
        if n_tasks > 1:
            return f"{var}={value}"
    return None


# mpi4py is an OPTIONAL dependency (install with: pip install "limTOD[mpi]").
# Without it, every function in this module degrades to serial mode
# (rank=0, size=1, world=None) — the same fallback the upstream caput
# mpiutil this file is adapted from provides. All consumers already guard
# on `size == 1` / `comm is None`, so serial behavior is unchanged.
try:
    # Initialize MPI when module is imported
    init_mpi()
    from mpi4py import MPI

    _comm = MPI.COMM_WORLD
    world = _comm
    rank = _comm.Get_rank()
    size = _comm.Get_size()

    if _comm is not None and size > 1:
        logger.debug("Starting MPI rank=%i [size=%i]", rank, size)
except ImportError:
    # Guard against the silent-duplication trap: under `mpirun -n N` without
    # mpi4py, every process would believe it is rank 0 of 1 and run the FULL
    # workload — N-fold duplicated compute, and rank-0-gated file writes
    # would collide. Fail loudly instead (escape hatch: LIMTOD_FORCE_SERIAL=1).
    _launcher = _detect_mpi_launcher()
    if _launcher is not None and os.environ.get("LIMTOD_FORCE_SERIAL") != "1":
        raise RuntimeError(
            f"An MPI launcher is detected ({_launcher}) but mpi4py is not "
            "installed, so every process would silently run the whole "
            "workload in serial mode. Install the MPI extra "
            '(pip install "limTOD[mpi]") or, if running N independent serial '
            "copies is intentional, set LIMTOD_FORCE_SERIAL=1."
        ) from None
    logger.debug("mpi4py not found — running in serial mode (size=1)")

rank0 = rank == 0


def partition_list(
    full_list: Sequence[Any], i: int, n: int, method: str = "con"
) -> Sequence[Any]:
    """
    Partition a list into `n` pieces. Return the `i`th partition.
    """

    def _partition(N: int, n: int, i: int) -> Tuple[int, int]:
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


def partition_list_mpi(
    full_list: Sequence[Any], method: str = "con", comm: Optional["Comm"] = _comm
) -> Sequence[Any]:
    """
    Return the partition of a list specific to the current MPI process.
    """
    # Distinct local names: assigning to `rank`/`size` here would shadow the
    # module-level serial defaults and leave them unbound when comm is None.
    if comm is not None:
        proc_rank, proc_size = comm.rank, comm.size
    else:
        proc_rank, proc_size = rank, size

    return partition_list(full_list, proc_rank, proc_size, method=method)


def parallel_map_gather(
    func: Callable[..., Any],
    glist: Sequence[Any],
    multi_inputs: bool = False,
    root: Optional[int] = None,
    method: str = "con",
    comm: Optional["Comm"] = _comm,
) -> Optional[List[Any]]:
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


def parallel_jobs_no_gather_no_return(
    func: Callable[..., Any],
    glist: Sequence[Any],
    method: str = "con",
    comm: Optional["Comm"] = _comm,
) -> Optional[List[Any]]:
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


def barrier(comm: Optional["Comm"] = _comm) -> None:
    """
    Synchronize all MPI processes.
    """
    if comm is not None and comm.size > 1:
        comm.Barrier()
