import numpy as np
import pytest

from fastnmf import FastNMF, extract_gene_modules_3ca


def test_reconstruction_error_decreases():
    rng = np.random.default_rng(0)
    m, n, k = 120, 80, 8
    X = rng.random((m, k)) @ rng.random((k, n))

    model = FastNMF(n_components=k, max_iter=100, tol=1e-6, init="nndsvd", check_every=2, random_state=0)
    res = model.factorize(X)

    assert len(res.history) >= 2
    assert res.history[-1] <= res.history[0]
    assert res.n_iter <= model.max_iter


def test_transform_shape():
    rng = np.random.default_rng(1)
    X = rng.random((60, 40))
    model = FastNMF(n_components=6, max_iter=40, init="random", random_state=1)
    model.fit(X)

    X_new = rng.random((20, 40))
    W_new = model.transform(X_new, n_iter=20)
    assert W_new.shape == (20, 6)


def test_invalid_n_components_raises():
    X = np.ones((10, 4))
    model = FastNMF(n_components=5)
    with pytest.raises(ValueError):
        model.factorize(X)


def test_memory_budget_resolves_batch_size():
    X = np.ones((200, 100), dtype=np.float32)
    model = FastNMF(n_components=10, max_memory_mb=0.05, dtype="float32")
    b = model._resolve_batch_size(X)
    assert b is not None
    assert 1 <= b <= X.shape[1]


def test_invalid_thread_or_memory_raises():
    with pytest.raises(ValueError):
        FastNMF(n_components=4, n_threads=0)
    with pytest.raises(ValueError):
        FastNMF(n_components=4, max_memory_mb=0)


def test_select_k_returns_candidate():
    rng = np.random.default_rng(2)
    X = rng.random((80, 60))
    model = FastNMF(n_components=4, max_iter=20, check_every=2, random_state=2)
    out = model.select_k(X, k_values=[2, 4, 6], metric="reconstruction")
    assert out.best_k in {2, 4, 6}
    assert set(out.scores.keys()) == {2, 4, 6}


def test_extract_gene_modules_3ca_shapes():
    H = np.array([[3.0, 0.2, 1.2], [0.1, 2.5, 0.9]])
    genes = ["A", "B", "C"]
    mods = extract_gene_modules_3ca(H, genes, top_n=2, min_specificity=1.1)
    assert set(mods.modules.keys()) == {0, 1}


def test_fit_best_k_sets_model_state():
    rng = np.random.default_rng(3)
    X = rng.random((50, 30))
    m = FastNMF(n_components=2, max_iter=15, check_every=3, random_state=3)
    sel, res = m.fit_best_k(X, [2, 3])
    assert sel.best_k in {2, 3}
    assert m.H_ is not None
    assert res.H.shape[0] == sel.best_k
