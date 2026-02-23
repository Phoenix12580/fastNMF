from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import nnls
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

EPS = 1e-10
VERSION = "0.2.1"
ArrayLike = Union[np.ndarray, sparse.spmatrix]


@dataclass
class NMFFit:
    W: np.ndarray
    H: np.ndarray
    loss_curve: List[float]
    n_iter: int
    runtime_sec: float
    seed: int
    backend: str
    sparse_preserved_flag: bool


@dataclass
class Program:
    program_id: str
    sample_id: str
    rank: int
    idx: int
    top_genes: List[str]
    weights: Dict[str, float]


@dataclass
class RobustProgram:
    robust_id: str
    sample_id: str
    member_program_ids: List[str]
    support_distinct_k: int
    consensus_top50: List[str]


@dataclass
class MetaProgram:
    mp_id: str
    consensus_top50: List[str]
    member_program_ids: List[str]
    support_programs: int
    support_tumors: int
    median_intra_jaccard: float
    score: float


def parse_ranks(ranks: Union[str, Sequence[int], range, int]) -> List[int]:
    if isinstance(ranks, int):
        return [ranks]
    if isinstance(ranks, range):
        return list(ranks)
    if isinstance(ranks, str):
        x = ranks.strip()
        if ":" in x:
            p = [int(v) for v in x.split(":")]
            if len(p) == 2:
                return list(range(p[0], p[1] + 1))
            if len(p) == 3:
                return list(range(p[0], p[2] + (1 if p[1] > 0 else -1), p[1]))
            raise ValueError("ranks must be start:end or start:step:end")
        if "," in x:
            return [int(v.strip()) for v in x.split(",") if v.strip()]
        return [int(x)]
    out = [int(v) for v in ranks]
    if not out:
        raise ValueError("ranks cannot be empty")
    return out


def _to_csr(x: ArrayLike) -> csr_matrix:
    if sparse.issparse(x):
        return x.tocsr(copy=False)
    return sparse.csr_matrix(np.asarray(x))


def _hash_sparse_matrix(x: ArrayLike) -> str:
    csr = _to_csr(x)
    h = hashlib.sha256()
    h.update(str(csr.shape).encode())
    h.update(csr.indptr.tobytes())
    h.update(csr.indices.tobytes())
    h.update(csr.data.tobytes())
    return h.hexdigest()


def adapt_input_anndata(adata: Any, prefer: Sequence[str] = ("counts", "raw", "X")) -> Tuple[ArrayLike, str, bool]:
    for key in prefer:
        if key == "counts" and hasattr(adata, "layers") and "counts" in adata.layers:
            x = adata.layers["counts"]
            return (x if sparse.issparse(x) else sparse.csr_matrix(x)), "layers[counts]", sparse.issparse(x)
        if key == "raw" and getattr(adata, "raw", None) is not None:
            x = adata.raw.X
            return (x if sparse.issparse(x) else sparse.csr_matrix(x)), "raw.X", sparse.issparse(x)
        if key == "X":
            x = adata.X
            return (x if sparse.issparse(x) else sparse.csr_matrix(x)), "X", sparse.issparse(x)
    raise ValueError("No matrix source found in AnnData")


def _loss_trace(X: csr_matrix, W: np.ndarray, H: np.ndarray, x_norm_sq: float) -> float:
    t1 = float(np.sum(W * (X @ H.T)))
    wt_w = W.T @ W
    hh_t = H @ H.T
    t2 = float(np.sum(wt_w * hh_t))
    val = x_norm_sq - 2.0 * t1 + t2
    return max(0.0, val)


def _update_mu(X: csr_matrix, W: np.ndarray, H: np.ndarray, l1_h: float) -> Tuple[np.ndarray, np.ndarray]:
    H *= (W.T @ X) / (W.T @ W @ H + l1_h + EPS)
    W *= (X @ H.T) / (W @ (H @ H.T) + EPS)
    np.maximum(H, EPS, out=H)
    np.maximum(W, EPS, out=W)
    return W, H


def _update_projgrad(X: csr_matrix, W: np.ndarray, H: np.ndarray, l1_h: float) -> Tuple[np.ndarray, np.ndarray]:
    grad_h = W.T @ (W @ H) - (W.T @ X)
    H -= 0.01 * grad_h
    H -= l1_h
    np.maximum(H, EPS, out=H)
    grad_w = (W @ (H @ H.T)) - (X @ H.T)
    W -= 0.01 * grad_w
    np.maximum(W, EPS, out=W)
    return W, H


def nmf_fit(
    x: ArrayLike,
    k: int,
    nrun: int = 10,
    method: str = "mu",
    max_iter: int = 500,
    tol: float = 1e-4,
    seed: int = 1,
    warm_start: bool = True,
    early_stop_patience: int = 10,
    l1_h: float = 0.0,
    return_loss: bool = True,
    float32: bool = False,
    init_wh: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    blas_threads: Optional[int] = None,
) -> NMFFit:
    if method not in {"mu", "projgrad"}:
        raise ValueError("method must be one of {'mu','projgrad'}")
    X = _to_csr(x)
    dtype = np.float32 if float32 else np.float64
    m, n = X.shape
    x_norm_sq = float(np.dot(X.data, X.data))
    sparse_flag = sparse.issparse(x)

    best_fit: Optional[NMFFit] = None
    best_loss = np.inf

    with threadpool_limits(limits=blas_threads):
        for run_idx in range(nrun):
            rng = np.random.default_rng(seed + run_idx)
            if init_wh is not None and warm_start:
                W, H = init_wh
                W = np.maximum(W[:, : min(k, W.shape[1])].astype(dtype, copy=True), EPS)
                H = np.maximum(H[: min(k, H.shape[0]), :].astype(dtype, copy=True), EPS)
                if W.shape[1] < k:
                    W = np.pad(W, ((0, 0), (0, k - W.shape[1])), mode="edge")
                if H.shape[0] < k:
                    H = np.pad(H, ((0, k - H.shape[0]), (0, 0)), mode="edge")
            else:
                W = rng.random((m, k), dtype=dtype) + EPS
                H = rng.random((k, n), dtype=dtype) + EPS

            start = perf_counter()
            loss_curve: List[float] = []
            stale = 0
            prev = np.inf
            for it in range(1, max_iter + 1):
                if method == "mu":
                    W, H = _update_mu(X, W, H, l1_h)
                else:
                    W, H = _update_projgrad(X, W, H, l1_h)

                cur = _loss_trace(X, W, H, x_norm_sq)
                if return_loss:
                    loss_curve.append(cur)
                rel = (prev - cur) / max(prev, EPS)
                stale = stale + 1 if rel < tol else 0
                prev = cur
                if stale >= early_stop_patience:
                    break

            fit = NMFFit(
                W=W,
                H=H,
                loss_curve=loss_curve,
                n_iter=it,
                runtime_sec=perf_counter() - start,
                seed=seed + run_idx,
                backend=method,
                sparse_preserved_flag=sparse_flag,
            )
            final_loss = loss_curve[-1] if loss_curve else np.inf
            if final_loss < best_loss:
                best_loss = final_loss
                best_fit = fit

    assert best_fit is not None
    return best_fit


def fit_global_hvg(samples: Dict[str, Tuple[ArrayLike, List[str]]], top_genes: int = 7000) -> List[str]:
    if not samples:
        raise ValueError("samples cannot be empty")
    common = None
    for _, (_, genes) in samples.items():
        s = set(genes)
        common = s if common is None else (common & s)
    common = sorted(common or [])
    if not common:
        raise ValueError("No common genes across samples")

    var_sum = np.zeros(len(common), dtype=float)
    idx_ref = {g: i for i, g in enumerate(common)}
    for _, (x, genes) in samples.items():
        csr = _to_csr(x)
        gidx = [genes.index(g) for g in common]
        sub = csr[gidx, :]
        mean = np.asarray(sub.mean(axis=1)).ravel()
        sq_mean = np.asarray(sub.multiply(sub).mean(axis=1)).ravel()
        var_sum += (sq_mean - mean * mean)
    order = np.argsort(-var_sum)[: min(top_genes, len(common))]
    return [common[i] for i in order]


def _select_genes_by_reference(x: ArrayLike, gene_names: List[str], reference_genes: List[str]) -> Tuple[csr_matrix, List[str]]:
    csr = _to_csr(x)
    idx_map = {g: i for i, g in enumerate(gene_names)}
    rows = []
    for g in reference_genes:
        if g in idx_map:
            rows.append(csr[idx_map[g], :])
        else:
            rows.append(sparse.csr_matrix((1, csr.shape[1])))
    return sparse.vstack(rows, format="csr"), list(reference_genes)


def _top_variable_genes_per_sample(x: ArrayLike, gene_names: List[str], top_genes: int) -> Tuple[csr_matrix, List[str]]:
    csr = _to_csr(x)
    mean = np.asarray(csr.mean(axis=1)).ravel()
    sq_mean = np.asarray(csr.multiply(csr).mean(axis=1)).ravel()
    var = sq_mean - mean * mean
    idx = np.argsort(-var)[: min(top_genes, csr.shape[0])]
    return csr[idx, :], [gene_names[i] for i in idx]


def _programs_from_fit(fit: NMFFit, rank: int, sample_id: str, genes: List[str], topN: int) -> List[Program]:
    out: List[Program] = []
    for j in range(rank):
        w = fit.W[:, j]
        idx = np.argsort(-w)[:topN]
        top = [genes[i] for i in idx]
        out.append(
            Program(
                program_id=f"{sample_id}|K{rank}_P{j+1}",
                sample_id=sample_id,
                rank=rank,
                idx=j + 1,
                top_genes=top,
                weights={genes[i]: float(w[i]) for i in idx},
            )
        )
    return out


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(1, len(sa | sb))


def _candidate_pairs(sets: List[Set[str]]) -> Set[Tuple[int, int]]:
    inv: Dict[str, List[int]] = {}
    for i, s in enumerate(sets):
        for g in s:
            inv.setdefault(g, []).append(i)
    pairs: Set[Tuple[int, int]] = set()
    for ids in inv.values():
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                if a > b:
                    a, b = b, a
                pairs.add((a, b))
    return pairs


def _components(adj: List[List[int]]) -> List[List[int]]:
    seen = [False] * len(adj)
    out: List[List[int]] = []
    for i in range(len(adj)):
        if seen[i]:
            continue
        st = [i]
        seen[i] = True
        comp = []
        while st:
            cur = st.pop()
            comp.append(cur)
            for nb in adj[cur]:
                if not seen[nb]:
                    seen[nb] = True
                    st.append(nb)
        out.append(comp)
    return out


def build_robust_programs(
    programs: List[Program],
    sample_intra_min_jaccard: float = 0.35,
    require_distinct_k_support: int = 2,
    min_cluster_size_programs: int = 2,
) -> List[RobustProgram]:
    sets = [set(p.top_genes) for p in programs]
    adj = [[] for _ in programs]
    for i, j in _candidate_pairs(sets):
        if _jaccard(sets[i], sets[j]) >= sample_intra_min_jaccard:
            adj[i].append(j)
            adj[j].append(i)
    out: List[RobustProgram] = []
    for comp in _components(adj):
        if len(comp) < min_cluster_size_programs:
            continue
        ks = {programs[i].rank for i in comp}
        if len(ks) < require_distinct_k_support:
            continue
        freq: Dict[str, int] = {}
        wsum: Dict[str, float] = {}
        for i in comp:
            for g in programs[i].top_genes:
                freq[g] = freq.get(g, 0) + 1
                wsum[g] = wsum.get(g, 0.0) + programs[i].weights.get(g, 0.0)
        consensus = sorted(freq.keys(), key=lambda g: (-freq[g], -wsum[g], g))[:50]
        out.append(
            RobustProgram(
                robust_id=f"{programs[comp[0]].sample_id}_RP{len(out)+1}",
                sample_id=programs[comp[0]].sample_id,
                member_program_ids=[programs[i].program_id for i in comp],
                support_distinct_k=len(ks),
                consensus_top50=consensus,
            )
        )
    return out


def build_cohort_mps(
    robust_programs_by_sample: Dict[str, List[RobustProgram]],
    inter_min_jaccard_for_edge: float = 0.30,
    mp_min_programs: int = 20,
    mp_min_tumors: int = 12,
    mp_median_intra_jaccard: float = 0.30,
    redundancy_jaccard: float = 0.50,
) -> Tuple[List[MetaProgram], List[MetaProgram]]:
    all_rp = [rp for v in robust_programs_by_sample.values() for rp in v]
    if len(robust_programs_by_sample) < 2 or len(all_rp) == 0:
        return [], []

    sets = [set(rp.consensus_top50) for rp in all_rp]
    adj = [[] for _ in all_rp]
    for i, j in _candidate_pairs(sets):
        if _jaccard(sets[i], sets[j]) >= inter_min_jaccard_for_edge:
            adj[i].append(j)
            adj[j].append(i)

    mps: List[MetaProgram] = []
    for comp in _components(adj):
        members = [all_rp[i] for i in comp]
        tumors = {m.sample_id for m in members}
        if len(members) < mp_min_programs or len(tumors) < mp_min_tumors:
            continue
        pairs = []
        for a in range(len(comp)):
            for b in range(a + 1, len(comp)):
                pairs.append(_jaccard(sets[comp[a]], sets[comp[b]]))
        med = float(np.median(pairs)) if pairs else 1.0
        if med < mp_median_intra_jaccard:
            continue
        freq: Dict[str, int] = {}
        for m in members:
            for g in m.consensus_top50:
                freq[g] = freq.get(g, 0) + 1
        cons = sorted(freq, key=lambda g: (-freq[g], g))[:50]
        score = len(tumors) * med * math.log1p(len(members))
        mps.append(
            MetaProgram(
                mp_id=f"MP{len(mps)+1}",
                consensus_top50=cons,
                member_program_ids=[pid for m in members for pid in m.member_program_ids],
                support_programs=len(members),
                support_tumors=len(tumors),
                median_intra_jaccard=med,
                score=score,
            )
        )

    best: List[MetaProgram] = []
    for mp in sorted(mps, key=lambda z: z.score, reverse=True):
        if any(_jaccard(mp.consensus_top50, b.consensus_top50) >= redundancy_jaccard for b in best):
            continue
        best.append(mp)
    return mps, best


def run_nmf_ranks(
    obj_or_matrix: ArrayLike,
    ranks: Union[str, Sequence[int], range, int] = range(4, 10),
    top_genes: int = 7000,
    topN: int = 50,
    sample_id: str = "sample1",
    gene_names: Optional[List[str]] = None,
    hvg_mode: str = "global_fixed",
    reference_genes: Optional[List[str]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    version: str = VERSION,
    **nmf_kwargs: Any,
) -> Dict[str, Any]:
    if hvg_mode not in {"global_fixed", "per_sample"}:
        raise ValueError("hvg_mode must be global_fixed or per_sample")
    if gene_names is None:
        raise ValueError("gene_names is required")

    X = _to_csr(obj_or_matrix)
    if hvg_mode == "global_fixed":
        if reference_genes is None:
            raise ValueError("global_fixed requires reference_genes")
        Xs, genes = _select_genes_by_reference(X, gene_names, reference_genes)
    else:
        Xs, genes = _top_variable_genes_per_sample(X, gene_names, top_genes)

    rr = parse_ranks(ranks)
    payload = {
        "sample_id": sample_id,
        "ranks": rr,
        "top_genes": top_genes,
        "topN": topN,
        "hvg_mode": hvg_mode,
        "seed": nmf_kwargs.get("seed", 1),
        "version": version,
        "input_hash": _hash_sparse_matrix(X),
    }
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    if cache_dir:
        cpath = Path(cache_dir) / f"{sample_id}_{key}.json"
        if cpath.exists():
            data = json.loads(cpath.read_text())
            return data

    fits: Dict[int, NMFFit] = {}
    warm = None
    programs: List[Program] = []
    for k in rr:
        fit = nmf_fit(Xs, k=k, init_wh=warm, **nmf_kwargs)
        fits[k] = fit
        warm = (fit.W, fit.H)
        programs.extend(_programs_from_fit(fit, k, sample_id, genes, topN))

    robust = build_robust_programs(programs)
    basis = np.concatenate([fits[k].W for k in rr], axis=1)
    program_names = [f"K{k}_P{i+1}" for k in rr for i in range(k)]

    out = {
        "sample_id": sample_id,
        "genes_nmf_w_basis": basis.tolist(),
        "genes": genes,
        "program_names": program_names,
        "robust_programs": [asdict(r) for r in robust],
        "params": {
            **payload,
            "sample_intra_min_jaccard": 0.35,
            "require_distinct_k_support": 2,
            "min_cluster_size_programs": 2,
            "inter_min_jaccard_for_edge": 0.30,
            "mp_min_programs": 20,
            "mp_min_tumors": 12,
            "mp_median_intra_jaccard": 0.30,
            "cache_source": "computed",
        },
    }
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        cpath = Path(cache_dir) / f"{sample_id}_{key}.json"
        cpath.write_text(json.dumps(out))
    return out


def compute_plot_data_jaccard(
    items: Sequence[Union[Program, RobustProgram, MetaProgram, Dict[str, Any]]],
    entity: str = "program",
    topN: int = 50,
    full_matrix: bool = False,
    max_entities_for_full: int = 2000,
) -> Dict[str, Any]:
    ids = []
    sets = []
    for i, item in enumerate(items):
        if isinstance(item, Program):
            ids.append(item.program_id)
            sets.append(set(item.top_genes[:topN]))
        elif isinstance(item, RobustProgram):
            ids.append(item.robust_id)
            sets.append(set(item.consensus_top50[:topN]))
        elif isinstance(item, MetaProgram):
            ids.append(item.mp_id)
            sets.append(set(item.consensus_top50[:topN]))
        else:
            ids.append(str(item.get("id", f"item_{i}")))
            sets.append(set(item.get("consensus_top50", item.get("top_genes", []))[:topN]))

    edges = []
    for i, j in _candidate_pairs(sets):
        jv = _jaccard(sets[i], sets[j])
        if jv > 0:
            edges.append((i, j, jv, 1 - jv))
    edge_df = pd.DataFrame(edges, columns=["i", "j", "jaccard", "distance"])

    data: Dict[str, Any] = {
        "entity": entity,
        "ids": ids,
        "edges": edge_df,
        "order": list(range(len(ids))),
        "annotations": {},
        "full_matrix": None,
    }

    if full_matrix:
        if len(ids) > max_entities_for_full:
            raise ValueError("too many entities for full heatmap")
        mat = np.eye(len(ids), dtype=float)
        for i, j, jv, _ in edges:
            mat[i, j] = jv
            mat[j, i] = jv
        data["full_matrix"] = mat
    return data


def save_plot_data_jaccard(plot_data: Dict[str, Any], out_dir: Union[str, Path], prefix: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_data["edges"].to_csv(out / f"{prefix}_edges.csv", index=False)
    meta = {"entity": plot_data["entity"], "ids": plot_data["ids"], "order": plot_data["order"]}
    (out / f"{prefix}_metadata.json").write_text(json.dumps(meta, indent=2))


def cli_run() -> None:
    p = argparse.ArgumentParser(prog="nmf-mp")
    sp = p.add_subparsers(dest="cmd", required=True)
    run = sp.add_parser("run")
    run.add_argument("--input", required=True)
    run.add_argument("--format", required=True, choices=["h5ad", "npy", "npz"])
    run.add_argument("--layer", default="counts")
    run.add_argument("--ranks", default="4:9")
    run.add_argument("--top-genes", type=int, default=7000)
    run.add_argument("--topN", type=int, default=50)
    run.add_argument("--hvg-mode", default="global_fixed", choices=["global_fixed", "per_sample"])
    run.add_argument("--reference-genes", default=None)
    run.add_argument("--out", required=True)
    args = p.parse_args()

    if args.format == "h5ad":
        import anndata as ad

        adata = ad.read_h5ad(args.input)
        X, _, _ = adapt_input_anndata(adata)
        genes = list(map(str, adata.var_names))
    elif args.format == "npy":
        X = np.load(args.input)
        genes = [f"gene_{i}" for i in range(X.shape[0])]
    else:
        X = sparse.load_npz(args.input)
        genes = [f"gene_{i}" for i in range(X.shape[0])]

    ref = None
    if args.reference_genes:
        ref = [x.strip() for x in Path(args.reference_genes).read_text().splitlines() if x.strip()]

    out = run_nmf_ranks(
        X,
        ranks=args.ranks,
        top_genes=args.top_genes,
        topN=args.topN,
        gene_names=genes,
        hvg_mode=args.hvg_mode,
        reference_genes=ref,
    )
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "result.json").write_text(json.dumps(out))


if __name__ == "__main__":
    cli_run()
