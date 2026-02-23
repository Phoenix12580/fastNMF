import json
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from fastnmf import (
    MetaProgram,
    Program,
    adapt_input_anndata,
    compute_plot_data_jaccard,
    compute_plot_data_mp_distribution,
    nmf_fit,
    parse_ranks,
    run_nmf_ranks,
    save_plot_data_jaccard,
    select_best_mps,
)


def test_parse_ranks_variants():
    assert parse_ranks("4:9") == [4, 5, 6, 7, 8, 9]
    assert parse_ranks("4,6,9") == [4, 6, 9]
    assert parse_ranks([3, 4]) == [3, 4]


def test_reproducible_seed():
    X = np.abs(np.random.default_rng(0).normal(size=(30, 20)))
    a = nmf_fit(X, k=4, nrun=1, seed=7, max_iter=30)
    b = nmf_fit(X, k=4, nrun=1, seed=7, max_iter=30)
    np.testing.assert_allclose(a.W, b.W, atol=1e-6)
    np.testing.assert_allclose(a.H, b.H, atol=1e-6)


def test_sparse_preserved_flag():
    X = sparse.random(40, 20, density=0.1, format="csr", random_state=0)
    fit = nmf_fit(X, k=3, nrun=1, seed=1, max_iter=10)
    assert fit.sparse_preserved_flag is True


def test_run_nmf_ranks_contract_shape_and_names():
    X = np.abs(np.random.default_rng(1).normal(size=(120, 30)))
    out = run_nmf_ranks(X, ranks=[4, 5], top_genes=50, nrun=1, max_iter=20, plot_mode="none")
    assert out.genes_nmf_w_basis.shape == (50, 9)
    assert out.program_names[0] == "K4_P1"
    assert out.program_names[-1] == "K5_P5"


def test_per_sample_requires_reference_for_reference_export():
    X = np.abs(np.random.default_rng(2).normal(size=(100, 20)))
    with pytest.raises(ValueError):
        run_nmf_ranks(X, ranks=[4], top_genes=50, nrun=1, max_iter=10, hvg_mode="per_sample", export_space="reference", plot_mode="none")


def test_best_mps_redundancy_filter():
    m1 = MetaProgram("MP1", [f"g{i}" for i in range(50)], [], 25, 13, 0.4, 10.0)
    m2 = MetaProgram("MP2", [f"g{i}" for i in range(45)] + ["x", "y", "z", "u", "v"], [], 24, 12, 0.35, 9.0)
    m3 = MetaProgram("MP3", [f"h{i}" for i in range(50)], [], 22, 12, 0.32, 8.0)
    kept = select_best_mps([m2, m1, m3])
    assert [x.mp_id for x in kept] == ["MP1", "MP3"]


def test_anndata_priority_layers_raw_x():
    pytest.importorskip("anndata")
    from anndata import AnnData

    a = AnnData(X=np.ones((10, 5)))
    a.layers["counts"] = sparse.csr_matrix(np.full((10, 5), 2.0))
    a.raw = a.copy()
    mat, src, is_sparse = adapt_input_anndata(a)
    assert src == "layers[counts]"
    assert sparse.issparse(mat)
    assert is_sparse is True


def test_jaccard_plot_data_symmetric_and_serializable(tmp_path):
    p1 = Program("s|K4_P1", "s", 4, 1, [f"g{i}" for i in range(50)], {})
    p2 = Program("s|K4_P2", "s", 4, 2, [f"g{i}" for i in range(25, 75)], {})
    pdata = compute_plot_data_jaccard([p1, p2], entity="program", topN=50)
    mat = pdata["jaccard_matrix"]
    assert np.allclose(mat, mat.T)
    assert np.allclose(np.diag(mat), 1.0)
    assert sorted(pdata["order"]) == [0, 1]

    save_plot_data_jaccard(pdata, tmp_path, "program_jaccard")
    loaded = pd.read_csv(tmp_path / "program_jaccard_jaccard_matrix.csv", index_col=0)
    meta = json.loads((tmp_path / "program_jaccard_metadata.json").read_text())
    assert loaded.shape == (2, 2)
    assert len(meta["order"]) == 2


def test_mp_distribution_toy_assignments():
    genes = [f"g{i}" for i in range(60)]
    X = np.zeros((60, 6), dtype=float)
    X[:30, :3] = 40
    X[30:60, 3:] = 40
    mp1 = MetaProgram("MP1", [f"g{i}" for i in range(30)], [], 30, 20, 0.4, 10)
    mp2 = MetaProgram("MP2", [f"g{i}" for i in range(30, 60)], [], 30, 20, 0.4, 10)
    data = compute_plot_data_mp_distribution({"s1": (X, genes)}, [mp1, mp2], min_genes=25, min_score=1, min_cells=0.05)
    assigns = data["cell_assignments"]["s1"]
    assert assigns[:3] == ["MP1", "MP1", "MP1"]
    assert assigns[3:] == ["MP2", "MP2", "MP2"]
    assert data["mp_freq_global"]["MP1"] == pytest.approx(0.5)
    assert data["mp_freq_global"]["MP2"] == pytest.approx(0.5)
