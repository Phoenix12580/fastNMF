from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import pandas as pd

import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

EPS = 1e-10
VERSION = "0.2.0"
ArrayLike = Union[np.ndarray, sparse.spmatrix]


@dataclass
class NMFConfig:
    k: int
    nrun: int = 10
    method: str = "hals"
    max_iter: int = 500
    tol: float = 1e-4
    seed: int = 1
    warm_start: bool = True
    early_stop_patience: int = 10
    l1_h: float = 0.0
    float32: bool = False
    return_loss: bool = True


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


@dataclass
class RunNMFResult:
    fits: Dict[int, NMFFit]
    genes_nmf_w_basis: np.ndarray
    genes: List[str]
    program_names: List[str]
    robust_programs: List[RobustProgram]
    mps: List[MetaProgram]
    best_mps: List[MetaProgram]
    params: Dict[str, Any]


def parse_ranks(ranks: Union[str, Sequence[int], range, int]) -> List[int]:
    if isinstance(ranks, int):
        return [ranks]
    if isinstance(ranks, range):
        return list(ranks)
    if isinstance(ranks, str):
        ranks = ranks.strip()
        if ":" in ranks:
            parts = [int(x) for x in ranks.split(":")]
            if len(parts) == 2:
                start, end = parts
                step = 1
            elif len(parts) == 3:
                start, step, end = parts
            else:
                raise ValueError("ranks string must be 'start:end' or 'start:step:end'")
            return list(range(start, end + (1 if step > 0 else -1), step))
        if "," in ranks:
            return [int(x.strip()) for x in ranks.split(",") if x.strip()]
        return [int(ranks)]
    out = [int(x) for x in ranks]
    if not out:
        raise ValueError("ranks cannot be empty")
    return out


def _to_csr(x: ArrayLike) -> sparse.csr_matrix:
    if sparse.issparse(x):
        return x.tocsr(copy=False)
    return sparse.csr_matrix(np.asarray(x))


def _hash_sparse_matrix(x: ArrayLike) -> str:
    csr = _to_csr(x)
    h = hashlib.sha256()
    h.update(str(csr.shape).encode())
    h.update(csr.indices.tobytes())
    h.update(csr.indptr.tobytes())
    h.update(csr.data.tobytes())
    return h.hexdigest()


def adapt_input_anndata(adata: Any, prefer: Sequence[str] = ("counts", "raw", "X")) -> Tuple[ArrayLike, str, bool]:
    for key in prefer:
        if key == "counts" and hasattr(adata, "layers") and "counts" in adata.layers:
            m = adata.layers["counts"]
            return (m if sparse.issparse(m) else sparse.csr_matrix(m)), "layers[counts]", sparse.issparse(m)
        if key == "raw" and getattr(adata, "raw", None) is not None:
            m = adata.raw.X
            return (m if sparse.issparse(m) else sparse.csr_matrix(m)), "raw.X", sparse.issparse(m)
        if key == "X":
            m = adata.X
            return (m if sparse.issparse(m) else sparse.csr_matrix(m)), "X", sparse.issparse(m)
    raise ValueError("No matrix source found in AnnData")


def _set_blas_threads(n_threads: Optional[int]) -> None:
    if n_threads is None:
        return
    value = str(max(1, int(n_threads)))
    os.environ["OMP_NUM_THREADS"] = value
    os.environ["OPENBLAS_NUM_THREADS"] = value
    os.environ["MKL_NUM_THREADS"] = value


def nmf_fit(
    x: ArrayLike,
    k: int,
    nrun: int = 10,
    method: str = "hals",
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
    if method not in {"hals", "anls"}:
        raise ValueError("method must be one of {'hals','anls'}")
    _set_blas_threads(blas_threads)
    X = _to_csr(x)
    dtype = np.float32 if float32 else np.float64
    m, n = X.shape
    sparse_flag = sparse.issparse(x)

    best_fit: Optional[NMFFit] = None
    best_loss = np.inf
    for run_idx in range(nrun):
        rng = np.random.default_rng(seed + run_idx)
        if init_wh is not None and warm_start:
            W0, H0 = init_wh
            W = np.maximum(W0[:, :k].astype(dtype, copy=True), EPS)
            H = np.maximum(H0[:k, :].astype(dtype, copy=True), EPS)
            if W.shape[1] < k:
                W = np.pad(W, ((0, 0), (0, k - W.shape[1])), mode="edge")
            if H.shape[0] < k:
                H = np.pad(H, ((0, k - H.shape[0]), (0, 0)), mode="edge")
        else:
            W = rng.random((m, k), dtype=dtype) + EPS
            H = rng.random((k, n), dtype=dtype) + EPS

        loss_curve: List[float] = []
        start = perf_counter()
        stale = 0
        prev = np.inf
        for it in range(1, max_iter + 1):
            if method == "hals":
                wt_x = (W.T @ X).astype(dtype, copy=False)
                wt_w_h = (W.T @ W @ H).astype(dtype, copy=False)
                H *= wt_x / (wt_w_h + l1_h + EPS)
                x_ht = (X @ H.T).astype(dtype, copy=False)
                w_h_ht = (W @ (H @ H.T)).astype(dtype, copy=False)
                W *= x_ht / (w_h_ht + EPS)
                np.maximum(W, EPS, out=W)
                np.maximum(H, EPS, out=H)
            else:
                resid = X - W @ H
                H = np.maximum(H + 0.01 * (W.T @ resid) - l1_h, EPS)
                resid = X - W @ H
                W = np.maximum(W + 0.01 * (resid @ H.T), EPS)

            resid = X - W @ H
            loss = float(np.sqrt(resid.multiply(resid).sum()) if sparse.issparse(resid) else np.linalg.norm(resid, ord="fro"))
            if return_loss:
                loss_curve.append(loss)
            rel = (prev - loss) / max(prev, EPS)
            stale = stale + 1 if rel < tol else 0
            prev = loss
            if stale >= early_stop_patience:
                break

        fit = NMFFit(W=W, H=H, loss_curve=loss_curve, n_iter=it, runtime_sec=perf_counter() - start, seed=seed + run_idx, backend=method, sparse_preserved_flag=sparse_flag)
        if loss_curve and loss_curve[-1] < best_loss:
            best_loss = loss_curve[-1]
            best_fit = fit
    assert best_fit is not None
    return best_fit


def _top_variable_genes(X: ArrayLike, top_genes: int, gene_names: Optional[List[str]] = None) -> Tuple[sparse.csr_matrix, List[str], np.ndarray]:
    csr = _to_csr(X)
    mean = np.asarray(csr.mean(axis=1)).ravel()
    sq_mean = np.asarray(csr.multiply(csr).mean(axis=1)).ravel()
    var = sq_mean - mean * mean
    idx = np.argsort(-var)[:top_genes]
    genes = gene_names if gene_names is not None else [f"gene_{i}" for i in range(csr.shape[0])]
    return csr[idx, :], [genes[i] for i in idx], idx


def _programs_from_fit(fit: NMFFit, rank: int, sample_id: str, genes: List[str], topN: int) -> List[Program]:
    progs: List[Program] = []
    for j in range(rank):
        w = fit.W[:, j]
        ord_idx = np.argsort(-w)[:topN]
        tg = [genes[i] for i in ord_idx]
        progs.append(Program(program_id=f"{sample_id}|K{rank}_P{j + 1}", sample_id=sample_id, rank=rank, idx=j + 1, top_genes=tg, weights={genes[i]: float(w[i]) for i in ord_idx}))
    return progs


def _jaccard_sets(a: Set[str], b: Set[str]) -> float:
    return len(a & b) / max(1, len(a | b))


def _candidate_pairs_from_inverted_index(gene_sets: List[Set[str]]) -> Set[Tuple[int, int]]:
    inv: Dict[str, List[int]] = {}
    for i, gs in enumerate(gene_sets):
        for g in gs:
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


def _connected_components(adj: List[List[int]]) -> List[List[int]]:
    seen = [False] * len(adj)
    out: List[List[int]] = []
    for i in range(len(adj)):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in adj[cur]:
                if not seen[nb]:
                    seen[nb] = True
                    stack.append(nb)
        out.append(comp)
    return out


def build_robust_programs(
    programs: List[Program],
    sample_intra_min_jaccard: float = 0.35,
    require_distinct_k_support: int = 2,
    min_cluster_size_programs: int = 2,
) -> List[RobustProgram]:
    gene_sets = [set(p.top_genes) for p in programs]
    cand_pairs = _candidate_pairs_from_inverted_index(gene_sets)
    adj = [[] for _ in programs]
    for i, j in cand_pairs:
        if _jaccard_sets(gene_sets[i], gene_sets[j]) >= sample_intra_min_jaccard:
            adj[i].append(j)
            adj[j].append(i)

    robust: List[RobustProgram] = []
    for comp in _connected_components(adj):
        if len(comp) < min_cluster_size_programs:
            continue
        ks = {programs[c].rank for c in comp}
        if len(ks) < require_distinct_k_support:
            continue
        freq: Dict[str, int] = {}
        wsum: Dict[str, float] = {}
        for c in comp:
            for g in programs[c].top_genes:
                freq[g] = freq.get(g, 0) + 1
                wsum[g] = wsum.get(g, 0.0) + programs[c].weights.get(g, 0.0)
        top = sorted(freq.keys(), key=lambda g: (-freq[g], -wsum[g], g))[:50]
        robust.append(RobustProgram(robust_id=f"{programs[comp[0]].sample_id}_RP{len(robust) + 1}", sample_id=programs[comp[0]].sample_id, member_program_ids=[programs[c].program_id for c in comp], support_distinct_k=len(ks), consensus_top50=top))
    return robust


def build_meta_programs(
    robust_programs: List[RobustProgram],
    inter_min_jaccard_for_edge: float = 0.30,
    mp_min_programs: int = 20,
    mp_min_tumors: int = 12,
    mp_median_intra_jaccard: float = 0.30,
) -> List[MetaProgram]:
    if not robust_programs:
        return []
    gene_sets = [set(rp.consensus_top50) for rp in robust_programs]
    cand_pairs = _candidate_pairs_from_inverted_index(gene_sets)
    adj = [[] for _ in robust_programs]
    for i, j in cand_pairs:
        if _jaccard_sets(gene_sets[i], gene_sets[j]) >= inter_min_jaccard_for_edge:
            adj[i].append(j)
            adj[j].append(i)

    out: List[MetaProgram] = []
    for comp in _connected_components(adj):
        members = [robust_programs[c] for c in comp]
        tumors = {m.sample_id for m in members}
        if len(members) < mp_min_programs or len(tumors) < mp_min_tumors:
            continue
        pairs = []
        for a in range(len(comp)):
            for b in range(a + 1, len(comp)):
                pairs.append(_jaccard_sets(gene_sets[comp[a]], gene_sets[comp[b]]))
        median_j = float(np.median(pairs)) if pairs else 1.0
        if median_j < mp_median_intra_jaccard:
            continue
        freq: Dict[str, int] = {}
        for m in members:
            for g in m.consensus_top50:
                freq[g] = freq.get(g, 0) + 1
        top = sorted(freq, key=lambda g: (-freq[g], g))[:50]
        score = len(tumors) * median_j * math.log1p(len(members))
        out.append(MetaProgram(mp_id=f"MP{len(out) + 1}", consensus_top50=top, member_program_ids=[pid for m in members for pid in m.member_program_ids], support_programs=len(members), support_tumors=len(tumors), median_intra_jaccard=median_j, score=score))
    return out


def select_best_mps(
    mps: List[MetaProgram],
    redundancy_jaccard: float = 0.50,
    mp_min_programs: int = 20,
    mp_min_tumors: int = 12,
    mp_median_intra_jaccard: float = 0.30,
) -> List[MetaProgram]:
    candidates = [m for m in mps if m.support_programs >= mp_min_programs and m.support_tumors >= mp_min_tumors and m.median_intra_jaccard >= mp_median_intra_jaccard]
    sorted_mps = sorted(candidates, key=lambda m: m.score, reverse=True)
    kept: List[MetaProgram] = []
    kept_sets: List[Set[str]] = []
    for mp in sorted_mps:
        mp_set = set(mp.consensus_top50)
        if any(_jaccard_sets(mp_set, ks) >= redundancy_jaccard for ks in kept_sets):
            continue
        kept.append(mp)
        kept_sets.append(mp_set)
    return kept


def run_nmf_ranks(
    obj_or_matrix: ArrayLike,
    ranks: Union[str, Sequence[int], range, int] = range(4, 10),
    top_genes: int = 7000,
    topN: int = 50,
    sample_id: str = "sample1",
    gene_names: Optional[List[str]] = None,
    hvg_mode: str = "global_fixed",
    hvg_method: str = "variance",
    reference_genes: Optional[List[str]] = None,
    export_space: str = "reference",
    cache_dir: Optional[Union[str, Path]] = None,
    version: str = VERSION,
    plot_mode: str = "save_both",
    out_dir: Optional[Union[str, Path]] = None,
    **nmf_kwargs: Any,
) -> RunNMFResult:
    if hvg_method != "variance":
        raise ValueError("Currently only hvg_method='variance' is supported")
    if hvg_mode not in {"global_fixed", "per_sample"}:
        raise ValueError("hvg_mode must be global_fixed or per_sample")
    if plot_mode not in {"none", "save_data", "save_plots", "save_both"}:
        raise ValueError("plot_mode must be one of none/save_data/save_plots/save_both")
    X = _to_csr(obj_or_matrix)
    ranks_list = parse_ranks(ranks)

    if hvg_mode == "global_fixed":
        Xsel, genes, _ = _top_variable_genes(X, top_genes=top_genes, gene_names=gene_names)
    else:
        Xsel, genes, _ = _top_variable_genes(X, top_genes=top_genes, gene_names=gene_names)
        if export_space == "reference" and reference_genes is None:
            raise ValueError("reference_genes is required for hvg_mode='per_sample' when export_space='reference'")

    cache_payload = {
        "ranks": ranks_list,
        "sample_id": sample_id,
        "top_genes": top_genes,
        "topN": topN,
        "hvg_mode": hvg_mode,
        "seed": nmf_kwargs.get("seed", 1),
        "version": version,
        "input_hash": _hash_sparse_matrix(X),
    }
    cache_key = hashlib.sha256(json.dumps(cache_payload, sort_keys=True).encode()).hexdigest()
    if cache_dir:
        cache_path = Path(cache_dir) / f"{sample_id}_{cache_key}.npz"
        if cache_path.exists():
            arr = np.load(cache_path, allow_pickle=True)
            genes_nmf_w_basis = arr["genes_nmf_w_basis"]
            genes = arr["genes"].tolist()
            program_names = arr["program_names"].tolist()
            params = json.loads(arr["params_json"].item())
            return RunNMFResult(fits={}, genes_nmf_w_basis=genes_nmf_w_basis, genes=genes, program_names=program_names, robust_programs=[], mps=[], best_mps=[], params=params)

    fits: Dict[int, NMFFit] = {}
    all_programs: List[Program] = []
    warm: Optional[Tuple[np.ndarray, np.ndarray]] = None
    for rank in ranks_list:
        fit = nmf_fit(Xsel, k=rank, init_wh=warm, **nmf_kwargs)
        fits[rank] = fit
        warm = (fit.W, fit.H)
        all_programs.extend(_programs_from_fit(fit, rank=rank, sample_id=sample_id, genes=genes, topN=topN))

    genes_nmf_w_basis = np.concatenate([fits[r].W for r in ranks_list], axis=1)
    program_names = [f"K{r}_P{i + 1}" for r in ranks_list for i in range(r)]

    robust = build_robust_programs(all_programs)
    mps = build_meta_programs(robust)
    best = select_best_mps(mps)

    if hvg_mode == "per_sample" and export_space == "reference" and reference_genes is not None:
        ref_index = {g: i for i, g in enumerate(reference_genes)}
        out = np.zeros((len(reference_genes), genes_nmf_w_basis.shape[1]), dtype=genes_nmf_w_basis.dtype)
        for i, g in enumerate(genes):
            if g in ref_index:
                out[ref_index[g], :] = genes_nmf_w_basis[i, :]
        genes_nmf_w_basis = out
        genes = list(reference_genes)

    params = {
        "ranks": ranks_list,
        "top_genes": top_genes,
        "topN": topN,
        "hvg_mode": hvg_mode,
        "hvg_method": hvg_method,
        "export_space": export_space,
        "sample_intra_min_jaccard": 0.35,
        "require_distinct_k_support": 2,
        "min_cluster_size_programs": 2,
        "inter_min_jaccard_for_edge": 0.30,
        "mp_min_programs": 20,
        "mp_min_tumors": 12,
        "mp_median_intra_jaccard": 0.30,
        "seed": nmf_kwargs.get("seed", 1),
        "version": version,
        "input_hash": cache_payload["input_hash"],
    }

    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            Path(cache_dir) / f"{sample_id}_{cache_key}.npz",
            genes_nmf_w_basis=genes_nmf_w_basis,
            genes=np.array(genes, dtype=object),
            program_names=np.array(program_names, dtype=object),
            params_json=np.array(json.dumps(params), dtype=object),
        )

    if plot_mode != "none":
        base_out = Path(out_dir) if out_dir is not None else Path("out")
        plot_data_dir = base_out / "plot_data"
        plots_dir = base_out / "plots"
        prog_plot_data = compute_plot_data_jaccard(all_programs, entity="program", topN=topN)
        mp_plot_data = compute_plot_data_jaccard(best if best else mps, entity="mp", topN=topN)
        if plot_mode in {"save_data", "save_both"}:
            save_plot_data_jaccard(prog_plot_data, plot_data_dir, "program_jaccard")
            save_plot_data_jaccard(mp_plot_data, plot_data_dir, "mp_jaccard")
            mpdist_data = compute_plot_data_mp_distribution({sample_id: (Xsel, genes)}, best if best else mps)
            save_plot_data_mp_distribution(mpdist_data, plot_data_dir)
        if plot_mode in {"save_plots", "save_both"}:
            plot_jaccard(prog_plot_data, plots_dir, "program_jaccard", style="3ca")
            plot_jaccard(mp_plot_data, plots_dir, "mp_jaccard", style="3ca")
            mpdist_data = compute_plot_data_mp_distribution({sample_id: (Xsel, genes)}, best if best else mps)
            plot_mp_distribution(mpdist_data, plots_dir, style="3ca")

    return RunNMFResult(fits=fits, genes_nmf_w_basis=genes_nmf_w_basis, genes=genes, program_names=program_names, robust_programs=robust, mps=mps, best_mps=best, params=params)



def _to_dense(x: ArrayLike) -> np.ndarray:
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def compute_plot_data_jaccard(
    items: Sequence[Union[Program, RobustProgram, MetaProgram, Dict[str, Any]]],
    entity: str = "program",
    topN: int = 50,
) -> Dict[str, Any]:
    if entity not in {"program", "mp"}:
        raise ValueError("entity must be 'program' or 'mp'")

    ids: List[str] = []
    gene_sets: List[Set[str]] = []
    for item in items:
        if isinstance(item, Program):
            ids.append(item.program_id)
            gene_sets.append(set(item.top_genes[:topN]))
        elif isinstance(item, (RobustProgram, MetaProgram)):
            ids.append(getattr(item, "robust_id", getattr(item, "mp_id", "item")))
            gene_sets.append(set(item.consensus_top50[:topN]))
        else:
            ids.append(str(item.get("id", f"item_{len(ids)}")))
            genes = item.get("top_genes", item.get("consensus_top50", []))
            gene_sets.append(set(genes[:topN]))

    n = len(gene_sets)
    jac = np.eye(n, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            val = _jaccard_sets(gene_sets[i], gene_sets[j])
            jac[i, j] = val
            jac[j, i] = val
    dist = 1.0 - jac

    if n > 1:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform

        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method="average")
        order = leaves_list(Z).tolist()
        clustering = {"method": "average", "linkage": Z.tolist()}
    else:
        order = [0] if n == 1 else []
        clustering = {"method": "average", "linkage": []}

    ordered_ids = [ids[i] for i in order]
    jac_df = pd.DataFrame(jac, index=ids, columns=ids)
    melted_df = jac_df.reset_index(names="row").melt(id_vars="row", var_name="col", value_name="jaccard")
    return {
        "entity": entity,
        "ids": ids,
        "order": order,
        "ordered_ids": ordered_ids,
        "jaccard_matrix": jac,
        "distance_matrix": dist,
        "clustering": clustering,
        "annotations": {},
        "melted_df": melted_df,
    }


def save_plot_data_jaccard(plot_data: Dict[str, Any], out_dir: Union[str, Path], prefix: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ids = plot_data["ids"]
    pd.DataFrame(plot_data["jaccard_matrix"], index=ids, columns=ids).to_csv(out / f"{prefix}_jaccard_matrix.csv")
    pd.DataFrame(plot_data["distance_matrix"], index=ids, columns=ids).to_csv(out / f"{prefix}_distance_matrix.csv")
    plot_data["melted_df"].to_csv(out / f"{prefix}_melted_df.csv", index=False)
    meta = {
        "entity": plot_data["entity"],
        "ids": ids,
        "order": plot_data["order"],
        "ordered_ids": plot_data["ordered_ids"],
        "clustering": plot_data["clustering"],
    }
    (out / f"{prefix}_metadata.json").write_text(json.dumps(meta, indent=2))


def plot_jaccard(plot_data: Dict[str, Any], out_dir: Union[str, Path], prefix: str, style: str = "3ca") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ord_idx = plot_data["order"]
    mat = plot_data["jaccard_matrix"]
    if ord_idx:
        mat = mat[np.ix_(ord_idx, ord_idx)]
        labels = [plot_data["ids"][i] for i in ord_idx]
    else:
        labels = plot_data["ids"]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title(f"{plot_data['entity']} Jaccard ({style})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out / f"{prefix}.png", dpi=200)
    fig.savefig(out / f"{prefix}.pdf")
    plt.close(fig)


def compute_plot_data_mp_distribution(
    expr_list_per_sample: Dict[str, Tuple[ArrayLike, List[str]]],
    mp_list: Sequence[MetaProgram],
    min_genes: int = 25,
    min_score: float = 1.0,
    min_cells: float = 0.05,
) -> Dict[str, Any]:
    mp_ids = [mp.mp_id for mp in mp_list]
    mp_genes = {mp.mp_id: set(mp.consensus_top50) for mp in mp_list}
    cell_scores: Dict[str, pd.DataFrame] = {}
    cell_assignments: Dict[str, List[str]] = {}
    mp_freq_by_sample: Dict[str, Dict[str, float]] = {}

    kept_global = set(mp_ids)
    for sid, (expr, genes) in expr_list_per_sample.items():
        X = _to_dense(expr)
        Xlog = np.log2((X / 10.0) + 1.0)
        gene_idx = {g: i for i, g in enumerate(genes)}
        scores = np.zeros((len(mp_ids), Xlog.shape[1]), dtype=float)
        for m_i, mp_id in enumerate(mp_ids):
            idx = [gene_idx[g] for g in mp_genes[mp_id] if g in gene_idx]
            if len(idx) >= min_genes:
                scores[m_i, :] = Xlog[idx, :].mean(axis=0)
        score_df = pd.DataFrame(scores.T, columns=mp_ids)
        max_idx = np.argmax(scores, axis=0) if len(mp_ids) else np.array([], dtype=int)
        max_vals = np.max(scores, axis=0) if len(mp_ids) else np.array([], dtype=float)
        assigns = [mp_ids[i] if v >= min_score else "unassigned" for i, v in zip(max_idx, max_vals)]

        freq = pd.Series(assigns).value_counts(normalize=True).to_dict()
        filtered_sample = {k: v for k, v in freq.items() if k == "unassigned" or v >= min_cells}
        kept_sample = {k for k in filtered_sample if k != "unassigned"}
        kept_global = kept_global & kept_sample if kept_global else kept_sample

        cell_scores[sid] = score_df
        cell_assignments[sid] = assigns
        mp_freq_by_sample[sid] = filtered_sample

    global_counts: Dict[str, float] = {k: 0.0 for k in mp_ids}
    total_cells = 0
    for sid, assigns in cell_assignments.items():
        total_cells += len(assigns)
        s = pd.Series(assigns).value_counts()
        for k, v in s.items():
            if k in global_counts:
                global_counts[k] += float(v)
    mp_freq_global = {k: (v / max(total_cells, 1)) for k, v in global_counts.items() if (v / max(total_cells, 1)) >= min_cells}

    final_mps = [m for m in mp_ids if m in kept_global and m in mp_freq_global]
    heatmap_matrix_by_sample: Dict[str, np.ndarray] = {}
    heatmap_orders: Dict[str, List[int]] = {}
    for sid, score_df in cell_scores.items():
        assigns = np.array(cell_assignments[sid])
        order = []
        for mp in final_mps:
            idx = np.where(assigns == mp)[0]
            if len(idx):
                ord_idx = idx[np.argsort(-score_df.iloc[idx][mp].to_numpy())]
                order.extend(ord_idx.tolist())
        unassigned_idx = np.where(assigns == "unassigned")[0].tolist()
        order.extend(unassigned_idx)
        if final_mps:
            mat = score_df[final_mps].to_numpy().T
            mat = mat[:, order] if order else mat
            mat = mat - mat.mean(axis=1, keepdims=True)
        else:
            mat = np.zeros((0, len(assigns)))
        heatmap_matrix_by_sample[sid] = mat
        heatmap_orders[sid] = order

    return {
        "cell_scores": cell_scores,
        "cell_assignments": cell_assignments,
        "mp_freq_global": mp_freq_global,
        "mp_freq_by_sample": mp_freq_by_sample,
        "heatmap_matrix_by_sample": heatmap_matrix_by_sample,
        "orders": heatmap_orders,
        "plot_params": {"min_genes": min_genes, "min_score": min_score, "min_cells": min_cells, "color_range": [-2, 2]},
        "mp_order": final_mps,
    }


def save_plot_data_mp_distribution(plot_data: Dict[str, Any], out_dir: Union[str, Path], prefix: str = "mp_distribution") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for sid, df in plot_data["cell_scores"].items():
        df.to_csv(out / f"{prefix}_{sid}_cell_scores.csv", index=False)
        pd.DataFrame({"assignment": plot_data["cell_assignments"][sid]}).to_csv(out / f"{prefix}_{sid}_assignments.csv", index=False)
        pd.DataFrame(plot_data["heatmap_matrix_by_sample"][sid]).to_csv(out / f"{prefix}_{sid}_heatmap.csv", index=False)
    meta = {
        "mp_freq_global": plot_data["mp_freq_global"],
        "mp_freq_by_sample": plot_data["mp_freq_by_sample"],
        "orders": plot_data["orders"],
        "plot_params": plot_data["plot_params"],
        "mp_order": plot_data["mp_order"],
    }
    (out / f"{prefix}_metadata.json").write_text(json.dumps(meta, indent=2))


def plot_mp_distribution(plot_data: Dict[str, Any], out_dir: Union[str, Path], prefix: str = "mp_distribution", style: str = "3ca") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Global pie
    labels = list(plot_data["mp_freq_global"].keys())
    vals = [plot_data["mp_freq_global"][k] for k in labels]
    if labels and sum(vals) > 0:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(vals, labels=labels, autopct="%1.1f%%")
        ax.set_title(f"MP Global Distribution ({style})")
        fig.tight_layout()
        fig.savefig(out / f"{prefix}_pie.png", dpi=200)
        fig.savefig(out / f"{prefix}_pie.pdf")
        plt.close(fig)

    for sid, freq in plot_data["mp_freq_by_sample"].items():
        items = [(k, v) for k, v in freq.items() if k != "unassigned"]
        if items:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar([x[0] for x in items], [x[1] for x in items])
            ax.set_xticklabels([x[0] for x in items], rotation=90, fontsize=7)
            ax.set_title(f"{sid} MP Frequency ({style})")
            fig.tight_layout()
            fig.savefig(out / f"{prefix}_{sid}_bar.png", dpi=200)
            fig.savefig(out / f"{prefix}_{sid}_bar.pdf")
            plt.close(fig)

        mat = plot_data["heatmap_matrix_by_sample"][sid]
        if mat.size > 0:
            fig, ax = plt.subplots(figsize=(7, 3))
            im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=plot_data["plot_params"]["color_range"][0], vmax=plot_data["plot_params"]["color_range"][1])
            ax.set_yticks(range(len(plot_data["mp_order"])))
            ax.set_yticklabels(plot_data["mp_order"], fontsize=7)
            ax.set_title(f"{sid} MP Heatmap ({style})")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(out / f"{prefix}_{sid}_heatmap.png", dpi=200)
            fig.savefig(out / f"{prefix}_{sid}_heatmap.pdf")
            plt.close(fig)


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
    run.add_argument("--cache-dir", default=None)
    run.add_argument("--plot-mode", default="save_both", choices=["none", "save_data", "save_plots", "save_both"])
    run.add_argument("--out", required=True)
    args = p.parse_args()

    if args.format == "h5ad":
        import anndata as ad

        adata = ad.read_h5ad(args.input)
        X, _, _ = adapt_input_anndata(adata)
        genes = list(getattr(adata, "var_names", [f"gene_{i}" for i in range(X.shape[0])]))
    elif args.format == "npy":
        X = np.load(args.input)
        genes = [f"gene_{i}" for i in range(X.shape[0])]
    else:
        X = sparse.load_npz(args.input)
        genes = [f"gene_{i}" for i in range(X.shape[0])]

    reference_genes = None
    if args.reference_genes:
        reference_genes = [x.strip() for x in Path(args.reference_genes).read_text().splitlines() if x.strip()]

    res = run_nmf_ranks(
        X,
        ranks=args.ranks,
        top_genes=args.top_genes,
        topN=args.topN,
        gene_names=genes,
        hvg_mode=args.hvg_mode,
        reference_genes=reference_genes,
        cache_dir=args.cache_dir,
        plot_mode=args.plot_mode,
        out_dir=args.out,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "Genes_nmf_w_basis.npy", res.genes_nmf_w_basis)
    (out / "program_names.json").write_text(json.dumps(res.program_names, ensure_ascii=False, indent=2))
    (out / "best_MPs.json").write_text(json.dumps([asdict(x) for x in res.best_mps], ensure_ascii=False, indent=2))
    (out / "params.json").write_text(json.dumps(res.params, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli_run()
