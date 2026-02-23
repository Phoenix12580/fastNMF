#' Parse rank specification
#' @param ranks Integer vector/range or string like "4:9" / "4,6,9".
#' @export
parse_ranks <- function(ranks = 4:9) {
  if (length(ranks) == 1 && is.character(ranks)) {
    x <- trimws(ranks)
    if (grepl(":", x, fixed = TRUE)) {
      p <- as.integer(strsplit(x, ":", fixed = TRUE)[[1]])
      if (length(p) == 2) return(seq.int(p[1], p[2]))
      if (length(p) == 3) return(seq.int(p[1], p[3], by = p[2]))
      stop("ranks must be start:end or start:step:end")
    }
    if (grepl(",", x, fixed = TRUE)) return(as.integer(strsplit(x, ",", fixed = TRUE)[[1]]))
    return(as.integer(x))
  }
  as.integer(ranks)
}

#' Unified expression matrix getter for Seurat/SCE/matrix.
#' @export
get_expr_matrix <- function(object,
                            assay = if (inherits(object, "Seurat")) Seurat::DefaultAssay(object) else "RNA",
                            layer = "counts",
                            join_layers_if_needed = TRUE,
                            enforce_sparse = TRUE) {
  if (inherits(object, "Seurat")) {
    assay_obj <- object[[assay]]
    if (!is.null(assay_obj[[layer]])) {
      m <- assay_obj[[layer]]
    } else if (requireNamespace("SeuratObject", quietly = TRUE) && "Layers" %in% getNamespaceExports("SeuratObject")) {
      lyr <- tryCatch(SeuratObject::Layers(assay_obj), error = function(e) character(0))
      split_counts <- grep("^counts\\.", lyr, value = TRUE)
      if (length(split_counts) > 1) {
        if (isTRUE(join_layers_if_needed)) {
          object <- SeuratObject::JoinLayers(object, assay = assay)
          assay_obj <- object[[assay]]
        } else {
          stop("Detected split Seurat v5 layers; set join_layers_if_needed=TRUE or call JoinLayers().")
        }
      }
      m <- tryCatch(SeuratObject::LayerData(assay_obj, layer = layer), error = function(e) NULL)
      if (is.null(m)) m <- Seurat::GetAssayData(object, assay = assay, slot = layer)
    } else {
      m <- Seurat::GetAssayData(object, assay = assay, slot = layer)
    }
    if (identical(layer, "data")) {
      has_negative <- if (inherits(m, "dgCMatrix")) any(m@x < 0) else any(m < 0)
      if (has_negative) stop("layer='data' contains negative values.")
    }
  } else if (inherits(object, "SingleCellExperiment")) {
    m <- SummarizedExperiment::assay(object, i = layer)
  } else if (inherits(object, "dgCMatrix") || is.matrix(object)) {
    m <- object
  } else {
    stop("Unsupported input type")
  }

  if (isTRUE(enforce_sparse) && !inherits(m, "dgCMatrix")) {
    if (!requireNamespace("Matrix", quietly = TRUE)) stop("Matrix package required for sparse coercion")
    m <- methods::as(m, "dgCMatrix")
  }
  m
}

#' NMF fit (HALS/ANLS)
#' @export
nmf_fit <- function(x, k, nrun = 10, method = c("hals", "anls"), max_iter = 500, tol = 1e-4,
                    seed = 1, warm_start = TRUE, early_stop_patience = 10, l1_h = 0,
                    return_loss = TRUE, init = NULL, float32 = FALSE) {
  method <- match.arg(method)
  X <- if (inherits(x, "dgCMatrix")) x else methods::as(x, "dgCMatrix")
  if (float32) storage.mode(X@x) <- "single"
  m <- nrow(X); n <- ncol(X)
  best <- NULL; best_loss <- Inf
  for (ri in seq_len(nrun)) {
    set.seed(seed + ri - 1)
    if (!is.null(init) && warm_start) {
      W <- pmax(init$W, 1e-10)
      H <- pmax(init$H, 1e-10)
      if (ncol(W) < k) W <- cbind(W, matrix(W[, ncol(W)], nrow(W), k - ncol(W)))
      if (nrow(H) < k) H <- rbind(H, matrix(H[nrow(H), ], k - nrow(H), ncol(H), byrow = TRUE))
      W <- W[, seq_len(k), drop = FALSE]
      H <- H[seq_len(k), , drop = FALSE]
    } else {
      W <- matrix(runif(m * k), m, k) + 1e-10
      H <- matrix(runif(k * n), k, n) + 1e-10
    }
    history <- numeric(0); stale <- 0L; prev <- Inf
    t0 <- proc.time()[3]
    for (it in seq_len(max_iter)) {
      if (method == "hals") {
        H <- H * (as.matrix(crossprod(W, X)) / (crossprod(W) %*% H + l1_h + 1e-10))
        W <- W * (as.matrix(X %*% t(H)) / (W %*% (H %*% t(H)) + 1e-10))
      } else {
        R <- X - W %*% H
        H <- pmax(H + 0.01 * (crossprod(W, R)) - l1_h, 1e-10)
        R <- X - W %*% H
        W <- pmax(W + 0.01 * (R %*% t(H)), 1e-10)
      }
      R <- X - W %*% H
      loss <- sqrt(sum(R@x^2))
      if (return_loss) history <- c(history, loss)
      rel <- (prev - loss) / max(prev, 1e-10)
      stale <- if (rel < tol) stale + 1L else 0L
      prev <- loss
      if (stale >= early_stop_patience) break
    }
    fit <- list(W = W, H = H, loss_curve = history, n_iter = it, runtime = proc.time()[3] - t0,
                seed = seed + ri - 1, backend = method, sparse_preserved_flag = TRUE)
    final_loss <- ifelse(length(history) > 0, tail(history, 1), Inf)
    if (final_loss < best_loss) { best <- fit; best_loss <- final_loss }
  }
  best
}

#' Run NMF on multiple ranks and export 3CA-aligned basis matrix.
#' @export
run_nmf_ranks <- function(obj_or_matrix, ranks = 4:9, top_genes = 7000, topN = 50,
                          assay = "RNA", layer = "counts", sample_id = "sample1",
                          hvg_mode = c("global_fixed", "per_sample"), reference_genes = NULL,
                          export_space = c("reference", "sample"), ...) {
  hvg_mode <- match.arg(hvg_mode)
  export_space <- match.arg(export_space)
  X <- get_expr_matrix(obj_or_matrix, assay = assay, layer = layer, enforce_sparse = TRUE)
  means <- Matrix::rowMeans(X)
  sqmeans <- Matrix::rowMeans(X^2)
  vars <- sqmeans - means * means
  idx <- head(order(vars, decreasing = TRUE), top_genes)
  genes_all <- rownames(X)
  if (is.null(genes_all)) genes_all <- paste0("gene_", seq_len(nrow(X)))
  genes <- genes_all[idx]
  Xs <- X[idx, , drop = FALSE]

  rr <- parse_ranks(ranks)
  fits <- list(); warm <- NULL
  for (k in rr) {
    fits[[as.character(k)]] <- nmf_fit(Xs, k = k, init = warm, ...)
    warm <- fits[[as.character(k)]]
  }
  basis <- do.call(cbind, lapply(rr, function(k) fits[[as.character(k)]]$W))

  if (hvg_mode == "per_sample" && export_space == "reference") {
    if (is.null(reference_genes)) stop("reference_genes is required for per_sample + reference export")
    out <- matrix(0, nrow = length(reference_genes), ncol = ncol(basis), dimnames = list(reference_genes, NULL))
    hit <- match(genes, reference_genes, nomatch = 0)
    keep <- which(hit > 0)
    out[hit[keep], ] <- basis[keep, , drop = FALSE]
    basis <- out
    genes <- reference_genes
  }

  colnames(basis) <- unlist(lapply(rr, function(k) paste0("K", k, "_P", seq_len(k))))
  list(fits = fits, Genes_nmf_w_basis = basis, genes = genes, topN = topN,
       params = list(
         sample_intra_min_jaccard = 0.35,
         require_distinct_k_support = 2,
         min_cluster_size_programs = 2,
         inter_min_jaccard_for_edge = 0.30,
         mp_min_programs = 20,
         mp_min_tumors = 12,
         mp_median_intra_jaccard = 0.30,
         seed = if (!is.null(list(...)$seed)) list(...)$seed else 1
       ))
}

#' Compute Jaccard plot data (program/MP)
#' @export
compute_plot_data_jaccard <- function(gene_sets, ids = NULL, topN = 50) {
  n <- length(gene_sets)
  if (is.null(ids)) ids <- paste0("item_", seq_len(n))
  gs <- lapply(gene_sets, function(x) unique(head(x, topN)))
  mat <- matrix(1, n, n, dimnames = list(ids, ids))
  for (i in seq_len(n)) {
    for (j in seq_len(n)) {
      if (i < j) {
        a <- gs[[i]]; b <- gs[[j]]
        val <- length(intersect(a, b)) / max(1, length(union(a, b)))
        mat[i, j] <- val; mat[j, i] <- val
      }
    }
  }
  d <- as.dist(1 - mat)
  hc <- if (n > 1) hclust(d, method = "average") else NULL
  ord <- if (!is.null(hc)) hc$order else seq_len(n)
  melted <- as.data.frame(as.table(mat), stringsAsFactors = FALSE)
  names(melted) <- c("row", "col", "jaccard")
  list(jaccard_matrix = mat, distance_matrix = 1 - mat, clustering = hc, order = ord,
       annotations = list(), melted_df = melted)
}

#' Save Jaccard plot data
#' @export
save_plot_data_jaccard <- function(plot_data, out_dir, prefix = "jaccard") {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  utils::write.csv(plot_data$jaccard_matrix, file.path(out_dir, paste0(prefix, "_jaccard_matrix.csv")))
  utils::write.csv(plot_data$distance_matrix, file.path(out_dir, paste0(prefix, "_distance_matrix.csv")))
  utils::write.csv(plot_data$melted_df, file.path(out_dir, paste0(prefix, "_melted_df.csv")), row.names = FALSE)
  meta <- list(order = plot_data$order)
  jsonlite::write_json(meta, file.path(out_dir, paste0(prefix, "_metadata.json")), auto_unbox = TRUE, pretty = TRUE)
}

#' Plot Jaccard heatmap
#' @export
plot_jaccard <- function(plot_data, out_dir, prefix = "jaccard", style = "3ca") {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  ord <- plot_data$order
  mat <- plot_data$jaccard_matrix[ord, ord, drop = FALSE]
  grDevices::png(file.path(out_dir, paste0(prefix, ".png")), width = 1200, height = 1000, res = 150)
  heatmap(mat, Rowv = NA, Colv = NA, scale = "none", col = hcl.colors(50, "Magma"), main = paste("Jaccard", style))
  grDevices::dev.off()
  grDevices::pdf(file.path(out_dir, paste0(prefix, ".pdf")), width = 8, height = 6)
  heatmap(mat, Rowv = NA, Colv = NA, scale = "none", col = hcl.colors(50, "Magma"), main = paste("Jaccard", style))
  grDevices::dev.off()
}
