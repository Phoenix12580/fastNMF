# fastNMF (R + Python)

Version: **0.2.1**

- 多 rank NMF (`4:9`)
- NMF 方法：`mu` / `projgrad`（默认 `mu`）
- 稀疏优先输入适配
- robust programs（样本内）
- cohort MPs / best_MPs（队列级）

## Python CLI

```bash
nmf-mp run --input data.h5ad --format h5ad --layer counts --ranks 4:9 --top-genes 7000 --hvg-mode global_fixed --reference-genes ref.txt --out outdir
```

## HVG

- `global_fixed`: 必须传 `reference_genes`
- `per_sample`: 每样本独立选高变基因

## 阈值默认值

- sample_intra_min_jaccard=0.35
- require_distinct_k_support=2
- min_cluster_size_programs=2
- inter_min_jaccard_for_edge=0.30
- mp_min_programs=20
- mp_min_tumors=12
- mp_median_intra_jaccard=0.30

## Tests

```bash
pytest -q
R -q -e 'testthat::test_dir("tests/testthat")'
```
