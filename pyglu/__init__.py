"""
pyglu — Python bindings for the GLU GPU-accelerated sparse LU solver.

API mirrors scipy.sparse.linalg.splu / spsolve:

    lu = pyglu.splu(A)          # factorize (returns GLUFactorization)
    x  = lu.solve(b)            # solve Ax = b

    x  = pyglu.spsolve(A, b)    # factorize and solve in one call

A can be any scipy sparse matrix (any format) or a
(data, indices, indptr, shape) tuple in CSC format.

Note: GLU's REAL type is double precision. The Python interface accepts and
returns float64 arrays.
"""

import numpy as np

try:
    import scipy.sparse as _sp
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

from ._pyglu import GLUFactorization


def _to_csc_arrays(A):
    """Return (data_f64, indices_i32, indptr_i32, n) from a sparse matrix."""
    if _HAS_SCIPY and _sp.issparse(A):
        A = A.tocsc()
        A.sum_duplicates()
        A.sort_indices()
        n = A.shape[0]
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"Matrix must be square, got shape {A.shape}")
        data    = np.ascontiguousarray(A.data,    dtype=np.float64)
        indices = np.ascontiguousarray(A.indices, dtype=np.int32)
        indptr  = np.ascontiguousarray(A.indptr,  dtype=np.int32)
    elif isinstance(A, tuple) and len(A) == 4:
        data, indices, indptr, shape = A
        n = shape[0]
        if shape[0] != shape[1]:
            raise ValueError(f"Matrix must be square, got shape {shape}")
        data    = np.ascontiguousarray(data,    dtype=np.float64)
        indices = np.ascontiguousarray(indices, dtype=np.int32)
        indptr  = np.ascontiguousarray(indptr,  dtype=np.int32)
    else:
        raise TypeError(
            "A must be a scipy sparse matrix or a "
            "(data, indices, indptr, shape) tuple in CSC format"
        )

    if len(indptr) != n + 1:
        raise ValueError(
            f"indptr length {len(indptr)} inconsistent with n={n}"
        )
    if n == 0:
        raise ValueError("Matrix dimension must be > 0")

    return data, indices, indptr, n


def splu(A, perturb=False, update_strategy="right-looking", scheduling="level", *,
         sf_workers=0, symbolic_backend="auto", symbolic_batch_cols=128,
         symbolic_block_size=256, symbolic_verify=False, symbolic_no_fallback=False,
         symbolic_sf_workers=0, symbolic_pool_entries=0, symbolic_max_pool_entries=0,
         numeric_sm_padding=False):
    """Compute the sparse LU factorization of A on the GPU.

    Parameters
    ----------
    A : scipy.sparse matrix (any format) or (data, indices, indptr, shape) tuple
        Square n-by-n sparse matrix.
    perturb : bool, optional
        Enable diagonal perturbation (GESP) to handle near-singular pivots.
    update_strategy : {"right-looking", "left-looking"}, optional
        Numeric update strategy. Default "right-looking". The aliases
        "right"/"rl" and "left"/"ll" are also accepted.
    scheduling : {"level", "synchronization-free"}, optional
        Default "level". Synchronization-free scheduling requires left-looking
        updates. Independent of the symbolic backend.
    sf_workers : int, optional
        Numeric SF worker limit; 0 selects automatically. Requires SF scheduling.
    numeric_sm_padding : bool, optional
        Experimental LL-SF grid with one block per SM and idle padding workers.
        Default False. Pads activity counters and may change execution time.
    symbolic_backend : {"auto", "cpu", "gpu", "gpu-sf"}, optional
        Default "auto" selects GPU-SF when n >= 20000, max column nnz / n >=
        0.1, max column nnz / mean column nnz >= 64, and
        sum(m_j * max(0, m_j - (n-j))) >= 1e9 after preprocessing; otherwise CPU.
        "gpu" uses graph search; "gpu-sf" merges published L patterns with
        bitmap workers and incremental checkpoints. Selection scans column
        pointers in O(n) time and O(1) extra space.
    symbolic_batch_cols : int, optional
        Positive graph GPU workspace slot limit. Default 128.
    symbolic_block_size : {64, 128, 256}, optional
        Symbolic GPU threads per block. Default 256.
    symbolic_verify : bool, optional
        Compare all three GPU CSC arrays against CPU fill-in. Default False.
    symbolic_no_fallback : bool, optional
        Raise on GPU symbolic failure instead of using CPU. Default False.
        Verification and input errors always raise. With "auto", GPU options
        apply only when GPU-SF is selected; selecting CPU is not a fallback.
    symbolic_sf_workers : int, optional
        Symbolic SF worker limit, separate from sf_workers; 0 = automatic.
    symbolic_pool_entries, symbolic_max_pool_entries : int, optional
        Initial/capped sparse SF arena capacities in row entries; 0 = automatic.
        Nonzero worker/pool options require symbolic_backend="auto" or "gpu-sf".

    Returns
    -------
    factorization : GLUFactorization
        Object with ``.solve(b)``, ``.n``, the actual ``.symbolic_backend``
        (including CPU fallback), and timing/counter text in ``.diagnostics``.

    Raises
    ------
    ValueError
        If the matrix is not square or arrays are inconsistent.
    RuntimeError
        If preprocessing or GPU factorization fails.

    Examples
    --------
    >>> import scipy.sparse as sp, numpy as np, pyglu
    >>> A = sp.eye(4, format='csc') * 2.0
    >>> lu = pyglu.splu(A)
    >>> x = lu.solve(np.ones(4))
    """
    data, indices, indptr, n = _to_csc_arrays(A)
    return GLUFactorization(
        data, indices, indptr, n, perturb, update_strategy, scheduling,
        sf_workers=sf_workers, symbolic_backend=symbolic_backend,
        symbolic_batch_cols=symbolic_batch_cols, symbolic_block_size=symbolic_block_size,
        symbolic_verify=symbolic_verify, symbolic_no_fallback=symbolic_no_fallback,
        symbolic_sf_workers=symbolic_sf_workers, symbolic_pool_entries=symbolic_pool_entries,
        symbolic_max_pool_entries=symbolic_max_pool_entries,
        numeric_sm_padding=numeric_sm_padding)


def spsolve(A, b, perturb=False, update_strategy="right-looking", scheduling="level", *,
            sf_workers=0, symbolic_backend="auto", symbolic_batch_cols=128,
            symbolic_block_size=256, symbolic_verify=False, symbolic_no_fallback=False,
            symbolic_sf_workers=0, symbolic_pool_entries=0, symbolic_max_pool_entries=0,
            numeric_sm_padding=False):
    """Solve the sparse linear system Ax = b on the GPU.

    Parameters
    ----------
    A : scipy.sparse matrix or (data, indices, indptr, shape) tuple
        Square n-by-n coefficient matrix.
    b : array_like, shape (n,)
        Right-hand side vector.
    perturb : bool, optional
        Enable diagonal perturbation. Default False.
    update_strategy : {"right-looking", "left-looking"}, optional
        Numeric update strategy. Default "right-looking".
    scheduling : {"level", "synchronization-free"}, optional
        Numeric scheduling. Default "level"; synchronization-free requires LL.
    sf_workers, symbolic_backend, symbolic_batch_cols, symbolic_block_size,
    symbolic_verify, symbolic_no_fallback, symbolic_sf_workers,
    symbolic_pool_entries, symbolic_max_pool_entries, numeric_sm_padding : optional
        Same keyword-only options and defaults as :func:`splu`.

    Returns
    -------
    x : numpy.ndarray, float64, shape (n,)

    Examples
    --------
    >>> import scipy.sparse as sp, numpy as np, pyglu
    >>> A = sp.eye(4, format='csc') * 2.0
    >>> x = pyglu.spsolve(A, np.ones(4))
    """
    b = np.asarray(b, dtype=np.float64)
    return splu(A, perturb=perturb, update_strategy=update_strategy,
                scheduling=scheduling, sf_workers=sf_workers,
                symbolic_backend=symbolic_backend, symbolic_batch_cols=symbolic_batch_cols,
                symbolic_block_size=symbolic_block_size, symbolic_verify=symbolic_verify,
                symbolic_no_fallback=symbolic_no_fallback, symbolic_sf_workers=symbolic_sf_workers,
                symbolic_pool_entries=symbolic_pool_entries,
                symbolic_max_pool_entries=symbolic_max_pool_entries,
                numeric_sm_padding=numeric_sm_padding).solve(b)


__all__ = ["splu", "spsolve", "GLUFactorization"]
