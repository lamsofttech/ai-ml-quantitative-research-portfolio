lagrange_interpolate <- function(x_nodes, y_nodes, x_eval) {
  stopifnot(length(x_nodes) > 0, length(x_nodes) == length(y_nodes))
  if (anyDuplicated(x_nodes)) stop("x_nodes must be distinct")

  vapply(x_eval, function(point) {
    exact <- which(point == x_nodes)
    if (length(exact) > 0) return(y_nodes[exact[1]])

    basis <- vapply(seq_along(x_nodes), function(j) {
      others <- seq_along(x_nodes) != j
      prod((point - x_nodes[others]) / (x_nodes[j] - x_nodes[others]))
    }, numeric(1))
    sum(y_nodes * basis)
  }, numeric(1))
}

# A quadratic interpolant for exp(x) at three known points.
x_nodes <- c(0, 0.5, 1)
y_nodes <- exp(x_nodes)
x_grid <- seq(0, 1, length.out = 101)
estimate <- lagrange_interpolate(x_nodes, y_nodes, x_grid)
cat("Maximum absolute error:", max(abs(exp(x_grid) - estimate)), "\n")

stopifnot(isTRUE(all.equal(lagrange_interpolate(x_nodes, y_nodes, x_nodes), y_nodes)))

