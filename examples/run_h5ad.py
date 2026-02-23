import json
import anndata as ad
from fastnmf import adapt_input_anndata, run_nmf_ranks

adata = ad.read_h5ad("input.h5ad")
X, src, _ = adapt_input_anndata(adata)
res = run_nmf_ranks(X, ranks="4:9", top_genes=7000, topN=50, gene_names=list(adata.var_names), nrun=1)
with open("best_MPs.json", "w") as f:
    json.dump([x.__dict__ for x in res.best_mps], f, ensure_ascii=False, indent=2)
