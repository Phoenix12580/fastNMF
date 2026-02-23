import json
import numpy as np
import pytest
from scipy import sparse

from fastnmf import (
    MetaProgram,
    Program,
    build_cohort_mps,
    compute_plot_data_jaccard,
    fit_global_hvg,
    nmf_fit,
    parse_ranks,
    run_nmf_ranks,
)


def test_parse_ranks_variants():
    assert parse_ranks("4:6") == [4, 5, 6]
    assert parse_ranks("4,6,9") == [4, 6, 9]


def test_loss_monotonic_nonincreasing_mu():
    X = np.abs(np.random.default_rng(0).normal(size=(60, 40)))
    fit = nmf_fit(X, k=4, nrun=1, method="mu", max_iter=30, early_stop_patience=100)
    diffs = np.diff(fit.loss_curve)
    assert np.all(diffs <= 1e-8)


def test_sparse_no_toarray_called(monkeypatch):
    called = {"n": 0}

    def boom(self):
        called["n"] += 1
        raise RuntimeError("toarray called")

    monkeypatch.setattr(sparse.csr_matrix, "toarray", boom, raising=True)
    X = sparse.random(200, 120, density=0.01, format="csr", random_state=1)
    genes = [f"g{i}" for i in range(200)]
    out = run_nmf_ranks(X, ranks=[2], top_genes=100, gene_names=genes, hvg_mode="per_sample", nrun=1, max_iter=3)
    assert len(out["robust_programs"]) >= 0
    assert called["n"] == 0


def test_single_sample_no_mp():
    mps, best = build_cohort_mps({"s1": []})
    assert mps == []
    assert best == []


def test_multi_sample_mp_possible():
    rp = {
        "s1": [
            {"robust_id": "a1", "sample_id": "s1", "member_program_ids": ["p1"], "support_distinct_k": 2, "consensus_top50": [f"g{i}" for i in range(50)]}
            for _ in range(20)
        ],
        "s2": [
            {"robust_id": "b1", "sample_id": "s2", "member_program_ids": ["p2"], "support_distinct_k": 2, "consensus_top50": [f"g{i}" for i in range(50)]}
            for _ in range(20)
        ],
    }
    from fastnmf import RobustProgram

    cast = {k: [RobustProgram(**x) for x in v] for k, v in rp.items()}
    mps, best = build_cohort_mps(cast, mp_min_programs=20, mp_min_tumors=2, mp_median_intra_jaccard=0)
    assert len(mps) >= 1
    assert len(best) >= 1


def test_global_fixed_reference_alignment():
    x1 = sparse.csr_matrix(np.abs(np.random.default_rng(1).normal(size=(6, 5))))
    g1 = ["a", "b", "c", "d", "e", "f"]
    x2 = sparse.csr_matrix(np.abs(np.random.default_rng(2).normal(size=(6, 5))))
    g2 = ["a", "b", "c", "x", "y", "z"]
    ref = ["a", "b", "c"]
    o1 = run_nmf_ranks(x1, ranks=[2], gene_names=g1, hvg_mode="global_fixed", reference_genes=ref, nrun=1, max_iter=2)
    o2 = run_nmf_ranks(x2, ranks=[2], gene_names=g2, hvg_mode="global_fixed", reference_genes=ref, nrun=1, max_iter=2)
    assert o1["genes"] == ref
    assert o2["genes"] == ref


def test_per_sample_gene_names_can_differ():
    x1 = sparse.csr_matrix(np.abs(np.random.default_rng(1).normal(size=(10, 5))))
    x2 = sparse.csr_matrix(np.abs(np.random.default_rng(2).normal(size=(10, 5))))
    o1 = run_nmf_ranks(x1, ranks=[2], gene_names=[f"a{i}" for i in range(10)], hvg_mode="per_sample", nrun=1, max_iter=2)
    o2 = run_nmf_ranks(x2, ranks=[2], gene_names=[f"b{i}" for i in range(10)], hvg_mode="per_sample", nrun=1, max_iter=2)
    assert o1["genes"] != o2["genes"]


def test_jaccard_edge_list_mode_large():
    progs = [Program(f"p{i}", "s", 4, 1, [f"g{j}" for j in range(i % 10, i % 10 + 50)], {}) for i in range(2000)]
    d = compute_plot_data_jaccard(progs, full_matrix=False)
    assert "edges" in d
    assert d["full_matrix"] is None


def test_cache_integrity(tmp_path):
    X = sparse.csr_matrix(np.abs(np.random.default_rng(0).normal(size=(100, 20))))
    genes = [f"g{i}" for i in range(100)]
    r1 = run_nmf_ranks(X, ranks=[2], gene_names=genes, hvg_mode="per_sample", cache_dir=tmp_path, nrun=1, max_iter=2)
    r2 = run_nmf_ranks(X, ranks=[2], gene_names=genes, hvg_mode="per_sample", cache_dir=tmp_path, nrun=1, max_iter=2)
    assert "robust_programs" in r2
    assert "params" in r2
    assert "version" in r2["params"]
    assert r2["params"]["input_hash"] == r1["params"]["input_hash"]
