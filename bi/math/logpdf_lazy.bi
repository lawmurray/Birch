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
  return x.value()*log(λ) - λ - lgamma(x.value() + 1.0);
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
 * Observe a Student's $t$ variate.
 *
 * - x: The variate.
 * - k: Degrees of freedom.
 *
 * Returns: the log probability density.
 */
function logpdf_lazy_student_t(x:Expression<Real>, k:Expression<Real>) ->
    Expression<Real> {
  auto a <- 0.5*(k + 1.0);
  return lgamma(a) - lgamma(0.5*k) - 0.5*log(π*k) - a*log1p(x*x/k);
}

/**
 * Observe a Student's $t$ variate with location and scale.
 *
 * - x: The variate.
 * - k: Degrees of freedom.
 * - μ: Location.
 * - σ2: Squared scale.
 *
 * Returns: the log probability density.
 */
function logpdf_lazy_student_t(x:Expression<Real>, k:Expression<Real>,
    μ:Expression<Real>, σ2:Expression<Real>) -> Expression<Real> {
  return logpdf_lazy_student_t((x - μ)/sqrt(σ2), k) - 0.5*log(σ2);
}

/**
 * Observe a normal inverse-gamma variate.
 *
 * - x: The variate.
 * - μ: Mean.
 * - a2: Variance scale.
 * - α: Shape of inverse-gamma on variance.
 * - β: Scale of inverse-gamma on variance.
 *
 * Returns: the log probability density.
 */
function logpdf_lazy_normal_inverse_gamma(x:Expression<Real>,
    μ:Expression<Real>, a2:Expression<Real>, α:Expression<Real>,
    β:Expression<Real>) -> Expression<Real> {
  return logpdf_lazy_student_t(x, 2.0*α, μ, a2*β/α);
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
  return -0.5*(dot(x - μ, inv(C)*(x - μ)) + D*log(2.0*π) + ldet(C));
}

/**
 * Observe a Gaussian variate with a multivariate linear normal inverse-gamma
 * prior with linear transformation.
 *
 * - x: The variate.
 * - a: Scale.
 * - ν: Precision times mean.
 * - Λ: Precision.
 * - c: Offset.
 * - α: Shape of the inverse-gamma.
 * - γ: Scale accumulator of the inverse-gamma.
 *
 * Returns: the log probability density.
 */
function logpdf_lazy_linear_multivariate_normal_inverse_gamma_gaussian(
    x:Expression<Real>, a:Expression<Real[_]>, ν:Expression<Real[_]>,
    Λ:Expression<LLT>, c:Expression<Real>, α:Expression<Real>,
    γ:Expression<Real>) -> Expression<Real> {
  auto μ <- inv(Λ)*ν;
  auto β <- γ - 0.5*dot(μ, ν);
  return logpdf_lazy_student_t(x, 2.0*α, dot(a, μ) + c,
      (β/α)*(1.0 + dot(a, inv(Λ)*a)));
}
