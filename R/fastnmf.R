# Fast NMF in R (memory-aware + optional compiled backend hooks)

.fastnmf_resolve_block_size <- function(m, n, k, max_memory_mb=NULL, block_size=NULL, bytes_per_num=8) {
  if (!is.null(block_size)) return(max(1L, min(as.integer(block_size), as.integer(n))))
  if (is.null(max_memory_mb)) return(as.integer(n))

  # Approx working set per column block in H update:
  # Xb[m,b] + numer[k,b] + denom[k,b] + H[,b] ~= (m + 3k) * b * bytes
  budget <- as.integer(max_memory_mb * 1024 * 1024)
  bytes_per_col <- (as.integer(m) + 3L * as.integer(k)) * as.integer(bytes_per_num)
  b <- max(1L, budget %/% max(bytes_per_col, 1L))
  min(as.integer(n), as.integer(b))
}

.fastnmf_mul <- function(A, B, use_compiled=TRUE) {
  if (isTRUE(use_compiled) && requireNamespace("fastNMFcpp", quietly=TRUE)) {
    return(fastNMFcpp::matmul(A, B))
  }
  A %*% B
}

fast_nmf <- function(
  X,
  k,
  max_iter=300,
  tol=1e-4,
  seed=1,
  check_every=5,
  patience=3,
  max_memory_mb=NULL,
  block_size=NULL,
  n_threads=NULL,
  use_compiled=TRUE
) {
  if (any(X < 0)) stop("X must be non-negative")
  if (k <= 0) stop("k must be > 0")

  X <- as.matrix(X)
  storage.mode(X) <- "double"
  m <- nrow(X); n <- ncol(X)
  if (k > min(m, n)) stop("k must be <= min(nrow(X), ncol(X))")

  # Optional BLAS thread control (if RhpcBLASctl installed)
  old_threads <- NULL
  if (!is.null(n_threads) && requireNamespace("RhpcBLASctl", quietly=TRUE)) {
    old_threads <- RhpcBLASctl::blas_get_num_procs()
    RhpcBLASctl::blas_set_num_threads(as.integer(n_threads))
  }
  on.exit({
    if (!is.null(old_threads) && requireNamespace("RhpcBLASctl", quietly=TRUE)) {
      RhpcBLASctl::blas_set_num_threads(as.integer(old_threads))
    }
  }, add=TRUE)

  set.seed(seed)
  W <- matrix(runif(m * k), nrow=m, ncol=k) + 1e-10
  H <- matrix(runif(k * n), nrow=k, ncol=n) + 1e-10

  bsz <- .fastnmf_resolve_block_size(m, n, k, max_memory_mb=max_memory_mb, block_size=block_size)

  best <- Inf
  stale <- 0L
  history <- numeric(0)

  for (it in seq_len(max_iter)) {
    WtW <- crossprod(W) # t(W) %*% W

    if (bsz < n) {
      for (j0 in seq.int(1L, n, by=bsz)) {
        j1 <- min(n, j0 + bsz - 1L)
        Xb <- X[, j0:j1, drop=FALSE]
        numer <- crossprod(W, Xb) # t(W) %*% Xb
        denom <- WtW %*% H[, j0:j1, drop=FALSE] + 1e-10
        H[, j0:j1] <- H[, j0:j1, drop=FALSE] * (numer / denom)
      }
    } else {
      numer <- crossprod(W, X)
      denom <- WtW %*% H + 1e-10
      H <- H * (numer / denom)
    }

    HHt <- tcrossprod(H) # H %*% t(H)
    numerW <- .fastnmf_mul(X, t(H), use_compiled=use_compiled)
    denomW <- .fastnmf_mul(W, HHt, use_compiled=use_compiled) + 1e-10
    W <- W * (numerW / denomW)

    if (it %% check_every != 0 && it != max_iter) next

    err <- norm(X - .fastnmf_mul(W, H, use_compiled=use_compiled), type="F")
    history <- c(history, err)
    rel_improve <- (best - err) / max(best, 1e-10)
    if (rel_improve < tol) {
      stale <- stale + 1L
    } else {
      stale <- 0L
      best <- err
    }
    if (stale >= patience) break
  }

  list(
    W=W,
    H=H,
    err=if (length(history) > 0) tail(history, 1) else NA_real_,
    n_iter=it,
    history=history,
    block_size=bsz,
    n_threads=ifelse(is.null(n_threads), NA_integer_, as.integer(n_threads))
  )
}

select_k_fastnmf <- function(
  X,
  k_values,
  metric="bic",
  max_iter=200,
  seed=1,
  max_memory_mb=NULL,
  block_size=NULL,
  n_threads=NULL,
  use_compiled=TRUE
) {
  scores <- c()
  X <- as.matrix(X)
  m <- nrow(X); n <- ncol(X)
  for (k in sort(unique(k_values))) {
    fit <- fast_nmf(
      X, k=k, max_iter=max_iter, seed=seed,
      max_memory_mb=max_memory_mb, block_size=block_size,
      n_threads=n_threads, use_compiled=use_compiled
    )
    if (metric == "reconstruction") {
      score <- fit$err
    } else if (metric == "bic") {
      params <- k * (m + n)
      sigma2 <- max((fit$err^2) / max(m*n, 1), 1e-10)
      ll <- -0.5 * m * n * log(sigma2)
      score <- -2.0 * ll + params * log(max(m*n, 2))
    } else {
      stop("metric must be 'reconstruction' or 'bic'")
    }
    scores[as.character(k)] <- score
  }
  best_k <- as.integer(names(scores)[which.min(scores)])
  list(best_k=best_k, best_score=unname(min(scores)), scores=scores, metric=metric)
}

extract_gene_modules_3ca <- function(H, gene_names, top_n=50, min_specificity=1.5) {
  k <- nrow(H); n_genes <- ncol(H)
  if (length(gene_names) != n_genes) stop("gene_names length mismatch")

  dominant <- max.col(t(H), ties.method="first")
  maxv <- H[cbind(dominant, seq_len(n_genes))]
  second <- apply(H, 2, function(v) {
    if (length(v) == 1) return(1)
    sort(v, decreasing=TRUE)[2]
  })
  specificity <- maxv / (second + 1e-10)

  modules <- vector("list", k)
  names(modules) <- paste0("module_", seq_len(k))

  for (c in seq_len(k)) {
    idx <- which(dominant == c & specificity >= min_specificity)
    if (length(idx) == 0) {
      modules[[c]] <- character(0)
      next
    }
    ord <- idx[order(H[c, idx], decreasing=TRUE)]
    modules[[c]] <- gene_names[head(ord, top_n)]
  }

  list(modules=modules, score_matrix=H)
}

# Byte-compile hot R functions for modest speedup in pure-R fallback
if (requireNamespace("compiler", quietly=TRUE)) {
  fast_nmf <- compiler::cmpfun(fast_nmf)
  select_k_fastnmf <- compiler::cmpfun(select_k_fastnmf)
  extract_gene_modules_3ca <- compiler::cmpfun(extract_gene_modules_3ca)
}
