"""Fast, vectorized Non-negative Matrix Factorization (NMF)."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import os
from time import perf_counter
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from scipy import sparse
    from scipy.sparse.linalg import svds
except Exception:  # scipy is optional
    sparse = None
    svds = None

try:
    from threadpoolctl import threadpool_limits
except Exception:  # optional dependency
    threadpool_limits = None

EPS = 1e-10
ArrayLike = Union[np.ndarray, "sparse.spmatrix"]


@dataclass
class NMFResult:
    W: np.ndarray
    H: np.ndarray
    reconstruction_err: float
    n_iter: int
    history: List[float]
    elapsed_sec: float


@dataclass
class KSelectionResult:
    best_k: int
    best_score: float
    scores: Dict[int, float]
    metric: str


@dataclass
class GeneModuleResult:
    modules: Dict[int, List[str]]
    score_matrix: np.ndarray


class FastNMF:
    """Efficient NMF using multiplicative updates with practical speed knobs."""

    def __init__(
        self,
        n_components: int,
        max_iter: int = 300,
        tol: float = 1e-4,
        init: Literal["nndsvd", "random"] = "nndsvd",
        random_state: Optional[int] = None,
        l2_reg: float = 0.0,
        batch_size: Optional[int] = None,
        check_every: int = 5,
        patience: int = 3,
        dtype: Union[np.dtype, str] = np.float64,
        n_threads: Optional[int] = None,
        max_memory_mb: Optional[float] = None,
        verbose: bool = False,
    ) -> None:
        if n_components <= 0:
            raise ValueError("n_components must be > 0")
        if max_iter <= 0:
            raise ValueError("max_iter must be > 0")
        if check_every <= 0:
            raise ValueError("check_every must be > 0")
        if patience <= 0:
            raise ValueError("patience must be > 0")
        if init not in {"nndsvd", "random"}:
            raise ValueError("init must be 'nndsvd' or 'random'")
        if n_threads is not None and n_threads <= 0:
            raise ValueError("n_threads must be > 0 when provided")
        if max_memory_mb is not None and max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be > 0 when provided")

        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = random_state
        self.l2_reg = l2_reg
        self.batch_size = batch_size
        self.check_every = check_every
        self.patience = patience
        self.dtype = np.dtype(dtype)
        self.n_threads = n_threads
        self.max_memory_mb = max_memory_mb
        self.verbose = verbose

        self.W_: Optional[np.ndarray] = None
        self.H_: Optional[np.ndarray] = None
        self.history_: List[float] = []

    def fit_transform(self, X: ArrayLike) -> np.ndarray:
        result = self.factorize(X)
        self.W_, self.H_, self.history_ = result.W, result.H, result.history
        return self.W_

    def fit(self, X: ArrayLike) -> "FastNMF":
        self.fit_transform(X)
        return self

    def transform(self, X: ArrayLike, n_iter: int = 100) -> np.ndarray:
        if self.H_ is None:
            raise RuntimeError("Model is not fitted.")
        X = _check_nonnegative(X, self.dtype)
        m = X.shape[0]
        rng = np.random.default_rng(self.random_state)
        W = rng.random((m, self.n_components), dtype=self.dtype) + self.dtype.type(0.1)
        H = self.H_
        HHt = H @ H.T
        for _ in range(n_iter):
            numer = _matmul(X, H.T)
            denom = W @ HHt + self.l2_reg * W + EPS
            W *= numer / denom
        return W

    def inverse_transform(self, W: np.ndarray) -> np.ndarray:
        if self.H_ is None:
            raise RuntimeError("Model is not fitted.")
        return W @ self.H_

    def factorize(self, X: ArrayLike) -> NMFResult:
        X = _check_nonnegative(X, self.dtype)
        _, n = X.shape
        start = perf_counter()
        W, H = self._initialize(X)

        best = np.inf
        stale = 0
        history: List[float] = []
        eff_batch_size = self._resolve_batch_size(X)

        with _thread_ctx(self.n_threads):
            for it in range(1, self.max_iter + 1):
                wtw = W.T @ W
                if eff_batch_size and eff_batch_size < n:
                    for j0 in range(0, n, eff_batch_size):
                        j1 = min(j0 + eff_batch_size, n)
                        xb = X[:, j0:j1]
                        numer = _matmul(W.T, xb)
                        denom = wtw @ H[:, j0:j1] + self.l2_reg * H[:, j0:j1] + EPS
                        H[:, j0:j1] *= numer / denom
                else:
                    numer = _matmul(W.T, X)
                    denom = wtw @ H + self.l2_reg * H + EPS
                    H *= numer / denom

                hht = H @ H.T
                numer = _matmul(X, H.T)
                denom = W @ hht + self.l2_reg * W + EPS
                W *= numer / denom

                if it % self.check_every != 0 and it != self.max_iter:
                    continue

                err = _fro_error(X, W, H)
                history.append(err)
                if self.verbose:
                    print(f"iter={it:4d} err={err:.6e}")
                rel_improve = (best - err) / max(best, EPS)
                if rel_improve < self.tol:
                    stale += 1
                else:
                    stale = 0
                    best = err
                if stale >= self.patience:
                    break

        elapsed = perf_counter() - start
        final_err = history[-1] if history else _fro_error(X, W, H)
        return NMFResult(W=W, H=H, reconstruction_err=final_err, n_iter=it, history=history, elapsed_sec=elapsed)

    def select_k(
        self,
        X: ArrayLike,
        k_values: Sequence[int],
        metric: Literal["reconstruction", "bic"] = "reconstruction",
    ) -> KSelectionResult:
        X = _check_nonnegative(X, self.dtype)
        ks = sorted(set(int(k) for k in k_values))
        if not ks:
            raise ValueError("k_values must not be empty")

        m, n = X.shape
        scores: Dict[int, float] = {}
        best_k, best_score = ks[0], np.inf

        for k in ks:
            if k <= 0 or k > min(m, n):
                raise ValueError(f"Invalid k={k}; must be in [1, {min(m, n)}]")
            model = FastNMF(
                n_components=k,
                max_iter=self.max_iter,
                tol=self.tol,
                init=self.init,
                random_state=self.random_state,
                l2_reg=self.l2_reg,
                batch_size=self.batch_size,
                check_every=self.check_every,
                patience=self.patience,
                dtype=self.dtype,
                n_threads=self.n_threads,
                max_memory_mb=self.max_memory_mb,
                verbose=False,
            )
            res = model.factorize(X)
            err = res.reconstruction_err
            if metric == "reconstruction":
                score = err
            elif metric == "bic":
                params = k * (m + n)
                sigma2 = max((err**2) / max(m * n, 1), EPS)
                ll_approx = -0.5 * m * n * np.log(sigma2)
                score = -2.0 * ll_approx + params * np.log(max(m * n, 2))
            else:
                raise ValueError("metric must be 'reconstruction' or 'bic'")
            scores[k] = float(score)
            if score < best_score:
                best_k, best_score = k, float(score)

        return KSelectionResult(best_k=best_k, best_score=best_score, scores=scores, metric=metric)

    def fit_best_k(
        self,
        X: ArrayLike,
        k_values: Sequence[int],
        metric: Literal["reconstruction", "bic"] = "bic",
    ) -> Tuple[KSelectionResult, NMFResult]:
        sel = self.select_k(X, k_values, metric=metric)
        best = FastNMF(
            n_components=sel.best_k,
            max_iter=self.max_iter,
            tol=self.tol,
            init=self.init,
            random_state=self.random_state,
            l2_reg=self.l2_reg,
            batch_size=self.batch_size,
            check_every=self.check_every,
            patience=self.patience,
            dtype=self.dtype,
            n_threads=self.n_threads,
            max_memory_mb=self.max_memory_mb,
            verbose=self.verbose,
        )
        res = best.factorize(X)
        self.n_components = sel.best_k
        self.W_, self.H_, self.history_ = res.W, res.H, res.history
        return sel, res

    def extract_gene_modules_3ca(
        self,
        gene_names: Sequence[str],
        top_n: int = 50,
        min_specificity: float = 1.5,
    ) -> GeneModuleResult:
        if self.H_ is None:
            raise RuntimeError("Model is not fitted.")
        return extract_gene_modules_3ca(self.H_, gene_names, top_n, min_specificity)

    def _resolve_batch_size(self, X: ArrayLike) -> Optional[int]:
        _, n = X.shape
        if self.max_memory_mb is None:
            return self.batch_size
        itemsize = np.dtype(self.dtype).itemsize
        m = X.shape[0]
        k = self.n_components
        bytes_per_col = (m + 3 * k) * itemsize
        budget = int(self.max_memory_mb * 1024 * 1024)
        auto_batch = max(1, budget // max(bytes_per_col, 1))
        auto_batch = min(auto_batch, n)
        return auto_batch if self.batch_size is None else min(self.batch_size, auto_batch)

    def _initialize(self, X: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        m, n = X.shape
        k = self.n_components
        if k > min(m, n):
            raise ValueError("n_components must be <= min(n_samples, n_features)")
        rng = np.random.default_rng(self.random_state)
        if self.init == "random":
            avg = np.sqrt(max(_mean_value(X), EPS) / max(k, 1))
            return avg * rng.random((m, k), dtype=self.dtype) + EPS, avg * rng.random((k, n), dtype=self.dtype) + EPS

        U, S, VT = _topk_svd(X, k)
        W = np.zeros((m, k), dtype=self.dtype)
        H = np.zeros((k, n), dtype=self.dtype)
        W[:, 0] = np.sqrt(S[0]) * np.maximum(U[:, 0], 0)
        H[0, :] = np.sqrt(S[0]) * np.maximum(VT[0, :], 0)
        for j in range(1, k):
            uj, vj = U[:, j], VT[j, :]
            u_pos, u_neg = np.maximum(uj, 0), np.maximum(-uj, 0)
            v_pos, v_neg = np.maximum(vj, 0), np.maximum(-vj, 0)
            upn, unn = np.linalg.norm(u_pos), np.linalg.norm(u_neg)
            vpn, vnn = np.linalg.norm(v_pos), np.linalg.norm(v_neg)
            n_pos, n_neg = upn * vpn, unn * vnn
            if n_pos >= n_neg:
                W[:, j] = np.sqrt(S[j] * n_pos) * (u_pos / (upn + EPS))
                H[j, :] = np.sqrt(S[j] * n_pos) * (v_pos / (vpn + EPS))
            else:
                W[:, j] = np.sqrt(S[j] * n_neg) * (u_neg / (unn + EPS))
                H[j, :] = np.sqrt(S[j] * n_neg) * (v_neg / (vnn + EPS))
        W[W <= 0] = EPS
        H[H <= 0] = EPS
        return W, H


def extract_gene_modules_3ca(
    H: np.ndarray,
    gene_names: Sequence[str],
    top_n: int = 50,
    min_specificity: float = 1.5,
) -> GeneModuleResult:
    """3CA-style module extraction from factor loadings.

    Uses per-gene dominant-factor specificity ratio and keeps top genes per module.
    """
    H = np.asarray(H, dtype=np.float64)
    if H.ndim != 2:
        raise ValueError("H must be 2D: [k, n_genes]")
    k, n_genes = H.shape
    if len(gene_names) != n_genes:
        raise ValueError("gene_names length must equal H.shape[1]")

    denom = np.partition(H, kth=max(k - 2, 0), axis=0)[max(k - 2, 0), :] + EPS if k > 1 else np.ones(n_genes)
    dominant = np.argmax(H, axis=0)
    maxv = H[dominant, np.arange(n_genes)]
    specificity = maxv / denom

    modules: Dict[int, List[str]] = {}
    score = H.copy()
    for c in range(k):
        idx = np.where((dominant == c) & (specificity >= min_specificity))[0]
        if idx.size == 0:
            modules[c] = []
            continue
        order = idx[np.argsort(H[c, idx])[::-1]]
        keep = order[:top_n]
        modules[c] = [str(gene_names[i]) for i in keep]
    return GeneModuleResult(modules=modules, score_matrix=score)


def _topk_svd(X: ArrayLike, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sparse is not None and sparse.issparse(X) and svds is not None:
        U, S, VT = svds(X, k=k)
        order = np.argsort(S)[::-1]
        return U[:, order], S[order], VT[order, :]
    U, S, VT = np.linalg.svd(_to_dense(X), full_matrices=False)
    return U[:, :k], S[:k], VT[:k, :]


def _check_nonnegative(X: ArrayLike, dtype: np.dtype) -> ArrayLike:
    if sparse is not None and sparse.issparse(X):
        X = X.astype(dtype)
        if X.data.size and X.data.min() < 0:
            raise ValueError("X must be non-negative")
        return X
    X = np.asarray(X, dtype=dtype)
    if np.min(X) < 0:
        raise ValueError("X must be non-negative")
    return X


def _to_dense(X: ArrayLike) -> np.ndarray:
    if sparse is not None and sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def _matmul(A: ArrayLike, B: np.ndarray) -> np.ndarray:
    return A @ B


def _mean_value(X: ArrayLike) -> float:
    if sparse is not None and sparse.issparse(X):
        return float(X.sum()) / (X.shape[0] * X.shape[1])
    return float(np.mean(X))


def _fro_error(X: ArrayLike, W: np.ndarray, H: np.ndarray) -> float:
    if sparse is not None and sparse.issparse(X):
        x_f2 = float(X.multiply(X).sum())
        xtw = X.T @ W
        cross = float(np.sum(H * xtw.T))
        wh_f2 = float(np.sum((W @ H) ** 2))
        return float(np.sqrt(max(x_f2 - 2.0 * cross + wh_f2, 0.0)))
    return float(np.linalg.norm(X - W @ H, ord="fro"))


def _thread_ctx(n_threads: Optional[int]):
    if n_threads is None:
        return nullcontext()
    if threadpool_limits is not None:
        return threadpool_limits(limits=n_threads)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n_threads)
    return nullcontext()


def benchmark(seed: int = 42) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    m, n, k = 1200, 600, 20
    X = rng.random((m, k)) @ rng.random((k, n))
    model = FastNMF(n_components=k, max_iter=100, tol=1e-5, init="nndsvd", check_every=4, random_state=seed)
    res = model.factorize(X)
    return {"iter": float(res.n_iter), "recon_err": res.reconstruction_err, "elapsed_sec": res.elapsed_sec}


def benchmark_k_selection(seed: int = 42) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    m, n, k_true = 1200, 600, 24
    X = rng.random((m, k_true)) @ rng.random((k_true, n)) + 0.01 * rng.random((m, n))
    base = FastNMF(n_components=8, max_iter=80, tol=1e-4, init="nndsvd", check_every=4, random_state=seed)
    sel = base.select_k(X, k_values=range(12, 37, 6), metric="bic")
    return {"best_k": float(sel.best_k), "best_score": sel.best_score}


if __name__ == "__main__":
    print({"benchmark": benchmark(), "k_selection": benchmark_k_selection()})
