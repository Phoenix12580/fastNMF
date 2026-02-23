# fastNMF (R + Python)

跨 **R + Python** 的单细胞 NMF + Meta-Program 计算引擎：
- 多 `k/ranks`（默认 `4:9`）
- `HALS/ANLS` 稀疏 NMF（默认 HALS，支持 warm-start、early-stop、float32）
- 两种 HVG 模式：`global_fixed` / `per_sample`
- 样本内 robust programs + 跨样本 MPs + 去冗余 `best_MPs`
- 对齐 3CA 的 `Genes_nmf_w_basis`：`7000 × sum(ranks)`，列名 `K{rank}_P{idx}`

## Python

```bash
pip install -e .
nmf-mp run --input data.h5ad --format h5ad --layer counts --ranks 4:9 --top-genes 7000 --topN 50 --hvg-mode global_fixed --plot-mode save_both --out outdir
```

### AnnData 输入优先级（固定）
`layers["counts"] > raw.X > X`。

## R

```r
source("R/fastnmf.R")
out <- run_nmf_ranks(seu, ranks = 4:9, assay = "RNA", layer = "counts", nrun = 3)
```

### R 输入适配
- `get_expr_matrix()` 统一 Seurat v4/v5、SCE、matrix/dgCMatrix
- Seurat v5 split layers 检测与 `JoinLayers`
- `layer='data'` 且有负值直接报错
- 可强制输出 `dgCMatrix`

## 默认阈值（强筛）
- 样本内 robust：`sample_intra_min_jaccard=0.35`，`require_distinct_k_support=2`，`min_cluster_size_programs=2`
- 跨样本 MP：`inter_min_jaccard_for_edge=0.30`，`mp_min_programs=20`，`mp_min_tumors=12`，`mp_median_intra_jaccard=0.30`
- best_MPs 去冗余：`consensus_top50 Jaccard >= 0.50`

## 可复现与缓存
- 输出参数包含 `seed`、`version`、`input_hash`
- 可设置 `cache_dir` 命中后跳过重复计算

## 测试

```bash
pytest -q
R -q -e 'testthat::test_dir("tests/testthat")'
```


## Plot Engine (3CA style)
- `plot_mode`: `none|save_data|save_plots|save_both` (default `save_both`)
- 导出目录：`out/plot_data` 与 `out/plots`
- Jaccard 图：program / MP heatmap（含可序列化聚类与排序）
- MP distribution：global pie、per-sample bar、per-sample heatmap
- 计算与渲染分离：`compute_*` 仅产出数据，`plot_*` 仅读取 plot_data 作图
