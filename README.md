# fastNMF

针对肿瘤细胞/单细胞表达矩阵场景的 NMF 加速实现，提供 **Python + R** 双版本。

## 功能概览

- Python 版本：`fastnmf.py`
  - 向量化 NMF（MU 更新）
  - `n_threads` 多线程控制
  - `max_memory_mb` 内存预算自动分块
  - K 自动选择（`select_k` / `fit_best_k`）
  - 3CA 风格基因模块提取（`extract_gene_modules_3ca`）
- R 版本：`R/fastnmf.R`
  - `fast_nmf`
  - `select_k_fastnmf`
  - `extract_gene_modules_3ca`

## K 是固定还是范围自动选？

两种都支持：
- **固定 K**：直接传 `n_components=30`
- **范围自动选 K**：传候选集合，例如 `range(10, 61, 10)`，按 `bic` 或 `reconstruction` 自动选最优

## Python 使用示例（自动选 K + 3CA 基因模块）

```python
import numpy as np
from fastnmf import FastNMF

# X: cells x genes, non-negative
X = np.abs(np.random.randn(2000, 1000)).astype("float32")
gene_names = [f"G{i}" for i in range(X.shape[1])]

base = FastNMF(
    n_components=10,
    max_iter=300,
    init="nndsvd",
    n_threads=8,
    max_memory_mb=1024,
    random_state=0,
)

sel, fit = base.fit_best_k(X, k_values=range(10, 61, 10), metric="bic")
print("best_k:", sel.best_k)

modules = base.extract_gene_modules_3ca(
    gene_names=gene_names,
    top_n=50,
    min_specificity=1.5,
)
print(modules.modules[0][:10])
```

## R 使用示例（自动选 K + 3CA 基因模块）

```r
source("R/fastnmf.R")

set.seed(1)
X <- abs(matrix(rnorm(2000*1000), nrow=2000, ncol=1000))
gene_names <- paste0("G", seq_len(ncol(X)))

sel <- select_k_fastnmf(X, k_values=seq(10, 60, by=10), metric="bic", max_iter=120)
print(sel$best_k)

fit <- fast_nmf(X, k=sel$best_k, max_iter=300)
mods <- extract_gene_modules_3ca(fit$H, gene_names, top_n=50, min_specificity=1.5)
print(head(mods$modules[[1]], 10))
```

## 说明：3CA 风格基因模块

这里实现的是基于 NMF `H` 矩阵的“3CA 风格”模块筛选流程：
1. 每个基因分配到主导 factor（最大 loading 的 factor）。
2. 用主导 loading / 次大 loading 作为特异性分数。
3. 仅保留特异性超过阈值的基因，再按该 factor loading 排序取 top-N。

## 测试

```bash
python -m pytest -q
```


## R 版本性能优化（内存 + 编译加速）

`R/fastnmf.R` 现已支持：
- `max_memory_mb`：按内存预算自动计算列分块大小（H 更新分块）
- `block_size`：手工指定分块，优先于自动预算
- `n_threads`：若安装 `RhpcBLASctl`，可限制 BLAS 线程
- `use_compiled=TRUE`：若安装 `fastNMFcpp`（可选 C++ 后端包）则优先调用其矩阵乘法
- 默认启用 `compiler::cmpfun` 对热点函数进行字节码编译

示例：

```r
source("R/fastnmf.R")

fit <- fast_nmf(
  X, k=30, max_iter=300,
  max_memory_mb=1024,
  n_threads=8,
  use_compiled=TRUE
)
```


## 安装指南（Python / R）

### Python 安装

#### 方式 A：Conda（推荐，最省事）

```bash
conda create -n fastnmf python=3.10 -y
conda activate fastnmf
pip install numpy scipy threadpoolctl pytest
```

#### 方式 B：venv + pip

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install numpy scipy threadpoolctl pytest
```

### R 安装

先安装 R（建议 >= 4.2），然后在 R 控制台安装可选加速依赖：

```r
install.packages(c("RhpcBLASctl"))
# 可选：如果你有 fastNMFcpp 包源码/仓库，也可安装以启用 compiled matmul
# remotes::install_github("<your-org>/fastNMFcpp")
```

> 说明：`R/fastnmf.R` 不强依赖这些包；没有时会自动回退到基础 `%*%` 实现。

## 使用示例

### Python 最小示例（固定 K）

```python
import numpy as np
from fastnmf import FastNMF

X = np.abs(np.random.randn(500, 200)).astype("float32")
model = FastNMF(n_components=20, max_iter=200, init="nndsvd", n_threads=4)
W = model.fit_transform(X)
H = model.H_
print(W.shape, H.shape)
```

### Python 进阶示例（自动选 K + 模块提取）

```python
import numpy as np
from fastnmf import FastNMF

X = np.abs(np.random.randn(2000, 1000)).astype("float32")
genes = [f"G{i}" for i in range(X.shape[1])]

base = FastNMF(n_components=10, max_iter=300, n_threads=8, max_memory_mb=1024, random_state=0)
sel, _ = base.fit_best_k(X, k_values=range(10, 61, 10), metric="bic")
mods = base.extract_gene_modules_3ca(genes, top_n=50, min_specificity=1.5)

print("best_k=", sel.best_k)
print("module0 top genes:", mods.modules[0][:10])
```

### R 最小示例（固定 K）

```r
source("R/fastnmf.R")

set.seed(1)
X <- abs(matrix(rnorm(500*200), nrow=500, ncol=200))
fit <- fast_nmf(X, k=20, max_iter=200, n_threads=4)

dim(fit$W)
dim(fit$H)
```

### R 进阶示例（自动选 K + 模块提取）

```r
source("R/fastnmf.R")

set.seed(1)
X <- abs(matrix(rnorm(2000*1000), nrow=2000, ncol=1000))
genes <- paste0("G", seq_len(ncol(X)))

sel <- select_k_fastnmf(
  X,
  k_values=seq(10, 60, by=10),
  metric="bic",
  max_iter=120,
  max_memory_mb=1024,
  n_threads=8
)
fit <- fast_nmf(X, k=sel$best_k, max_iter=300, max_memory_mb=1024, n_threads=8)
mods <- extract_gene_modules_3ca(fit$H, genes, top_n=50, min_specificity=1.5)

print(sel$best_k)
print(head(mods$modules[[1]], 10))
```
