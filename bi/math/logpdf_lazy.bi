/**
 * Observe a Poisson variate.
 *
 * - x: The variate.
 * - λ: Rate.
 *
 * Returns: the log probability mass.
 */
function logpdf_lazy_poisson(x:Expression<Integer>, λ:Expression<Real>) ->
    Expression<Real> {
  return x*log(λ) - λ - lgamma(x + 1);
}

/**
 * Observe a Gaussian variate.
 *
 * - x: The variate.
 * - μ: Mean.
 * - σ2: Variance.
 *
 * Returns: the log probability density.
 */
function logpdf_lazy_gaussian(x:Expression<Real>, μ:Expression<Real>,
    σ2:Expression<Real>) -> Expression<Real> {
  return -0.5*(pow(x - μ, 2.0)/σ2 + log(2.0*π*σ2));
}

/**
 * Observe a multivariate Gaussian variate.
 *
 * - x: The variate.
 * - μ: Mean.
 * - Σ: Covariance.
 *
 * Returns: the log probability density.
 */
function logpdf_lazy_multivariate_gaussian(x:Expression<Real[_]>,
    μ:Expression<Real[_]>, Σ:Expression<Real[_,_]>) -> Expression<Real> {
  auto D <- μ.rows();
  auto C <- llt(Σ);
  return -0.5*(dot(x - μ, inv(C)*(x - μ)) + (D*log(2.0*π) + ldet(C)));
}
