test_that("parse_ranks works", {
  expect_equal(parse_ranks("4:6"), 4:6)
  expect_equal(parse_ranks("4,7,9"), c(4L,7L,9L))
})

test_that("matrix and sparse adapters keep type", {
  m <- matrix(1, 5, 3)
  a <- get_expr_matrix(m, enforce_sparse = FALSE)
  expect_true(is.matrix(a))
  if (requireNamespace("Matrix", quietly = TRUE)) {
    s <- Matrix::Matrix(m, sparse = TRUE)
    b <- get_expr_matrix(s, enforce_sparse = TRUE)
    expect_true(inherits(b, "dgCMatrix"))
  }
})

test_that("SCE adapter uses assay", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  sce <- SingleCellExperiment::SingleCellExperiment(assays = list(counts = matrix(1,4,3)))
  a <- get_expr_matrix(sce, layer = "counts", enforce_sparse = FALSE)
  expect_equal(dim(a), c(4,3))
})

test_that("jaccard plot data symmetric and diagonal one", {
  pd <- compute_plot_data_jaccard(list(letters[1:5], letters[3:7]), ids = c("a", "b"), topN = 5)
  expect_equal(pd$jaccard_matrix[1, 1], 1)
  expect_equal(pd$jaccard_matrix[2, 2], 1)
  expect_equal(pd$jaccard_matrix[1, 2], pd$jaccard_matrix[2, 1])
})
