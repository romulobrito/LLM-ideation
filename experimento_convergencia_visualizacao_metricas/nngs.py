import numpy as np
from typing import Sequence, Tuple, List, Optional
from dataclasses import dataclass

try:
    from sklearn.neighbors import NearestNeighbors
except Exception:
    NearestNeighbors = None  # type: ignore


@dataclass
class NNGSResult:
    k_values: List[int]
    nngs_values: List[float]
    baseline_values: List[float]
    per_point_jaccard: Optional[np.ndarray] = None  # shape (n,) for the last k computed


def _check_inputs(X: np.ndarray, Y: np.ndarray) -> None:
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays (n x d).")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of rows (paired samples).")
    if X.shape[0] < 3:
        raise ValueError("Need at least 3 points.")


def _neighbors_indices(M: np.ndarray, k: int, metric: str = "cosine") -> np.ndarray:
    """Return indices of k nearest neighbors for each row (excluding the point itself).
    Uses sklearn NearestNeighbors if available; otherwise falls back to a numpy
    cosine distance implementation for small datasets.
    """
    n = M.shape[0]
    if k >= n:
        raise ValueError("k must be < n.")
    if NearestNeighbors is not None:
        nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)  # +1 include self, will drop later
        nn.fit(M)
        dists, inds = nn.kneighbors(M, n_neighbors=k + 1, return_distance=True)
        # drop the first neighbor (self index)
        return inds[:, 1:]
    # Fallback: cosine distance via dot products (only for metric='cosine')
    if metric != "cosine":
        raise RuntimeError("sklearn is not available; only metric='cosine' is supported in numpy fallback.")
    # Normalize rows to unit norm
    M_norm = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    sims = M_norm @ M_norm.T  # cosine similarity
    # set diagonal to -inf to avoid self
    np.fill_diagonal(sims, -np.inf)
    # argpartition to get top-k indices (descending similarities)
    inds = np.argpartition(-sims, kth=np.arange(k), axis=1)[:, :k]
    # reorder top-k by actual similarity (descending)
    row_indices = np.arange(n)[:, None]
    top_sims = sims[row_indices, inds]
    order = np.argsort(-top_sims, axis=1)
    inds_sorted = inds[row_indices, order]
    return inds_sorted


def _jaccard_overlap(ids_A: np.ndarray, ids_B: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-row Jaccard |A∩B| / |A∪B| given two (n x k) integer arrays of neighbor indices.
    Returns (inter_sizes, union_sizes, jaccard_per_row).
    """
    n, k = ids_A.shape
    inter = np.empty((n,), dtype=np.int32)
    union = np.empty((n,), dtype=np.int32)
    for i in range(n):
        a = ids_A[i]
        b = ids_B[i]
        a_s = np.sort(a)
        b_s = np.sort(b)
        ia = ib = 0
        cnt_inter = 0
        while ia < k and ib < k:
            if a_s[ia] == b_s[ib]:
                cnt_inter += 1
                ia += 1
                ib += 1
            elif a_s[ia] < b_s[ib]:
                ia += 1
            else:
                ib += 1
        inter[i] = cnt_inter
        union[i] = 2 * k - cnt_inter
    jacc = inter / union
    return inter, union, jacc


def nngs_score(X: np.ndarray, Y: np.ndarray, k: int, metric: str = "cosine") -> Tuple[float, np.ndarray]:
    """Compute NNGS(X,Y,k) and return (mean_jaccard, per_point_jaccard)."""
    _check_inputs(X, Y)
    if not (1 <= k < X.shape[0]):
        raise ValueError("k must satisfy 1 <= k < n.")
    neigh_X = _neighbors_indices(X, k=k, metric=metric)
    neigh_Y = _neighbors_indices(Y, k=k, metric=metric)
    _, _, jacc = _jaccard_overlap(neigh_X, neigh_Y)
    return float(np.mean(jacc).item()), jacc


def nngs_curve(X: np.ndarray, Y: np.ndarray, k_values: Sequence[int], metric: str = "cosine") -> NNGSResult:
    _check_inputs(X, Y)
    n = X.shape[0]
    k_vals = sorted({int(k) for k in k_values if 1 <= int(k) < n})
    if len(k_vals) == 0:
        raise ValueError("Provide at least one valid k in [1, n-1].")
    vals: List[float] = []
    baselines: List[float] = []
    last_per_point: Optional[np.ndarray] = None
    for k in k_vals:
        mean_j, per = nngs_score(X, Y, k=k, metric=metric)
        vals.append(mean_j)
        # analytic random baseline H(k) = k / (2*(n-1) - k)
        H = k / (2 * (n - 1) - k)
        baselines.append(H)
        last_per_point = per
    return NNGSResult(k_values=k_vals, nngs_values=vals, baseline_values=baselines, per_point_jaccard=last_per_point)


def load_array(path: str) -> np.ndarray:
    """Load array from .npy or .csv (comma-separated)."""
    path = str(path)
    if path.lower().endswith(".npy"):
        return np.load(path)
    # assume CSV
    import pandas as pd
    return pd.read_csv(path, header=None).to_numpy(dtype=float)
