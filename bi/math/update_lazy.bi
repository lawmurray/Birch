/**
 * Update the parameters of a Gaussian distribution with a Gaussian
 * likelihood.
 *
 * - x: The variate.
 * - μ: Prior mean.
 * - σ2: Prior variance.
 * - s2: Likelihood variance.
 *
 * Returns: the posterior hyperparameters `μ'` and `σ2'`.
 */
function update_lazy_gaussian_gaussian(x:Expression<Real>,
    μ:Expression<Real>, σ2:Expression<Real>, s2:Expression<Real>) ->
    (Expression<Real>, Expression<Real>) {
  auto λ <- 1.0/σ2;
  auto l <- 1.0/s2;
  auto λ' <- λ + l;
  auto μ' <- (λ*μ + l*x)/λ';
  return (μ', 1.0/λ');
}

/**
 * Update the parameters of a Gaussian distribution with a Gaussian
 * likelihood and scaling.
 *
 * - x: The variate.
 * - a: Scale.
 * - μ: Prior mean.
 * - σ2: Prior variance.
 * - c: Offset.
 * - s2: Likelihood variance.
 *
 * Returns: the posterior hyperparameters `μ'` and `λ'`.
 */
function update_lazy_linear_gaussian_gaussian(x:Expression<Real>,
    a:Expression<Real>, μ:Expression<Real>, σ2:Expression<Real>,
    c:Expression<Real>, s2:Expression<Real>) -> (Expression<Real>,
    Expression<Real>) {
  auto λ <- 1.0/σ2;
  auto l <- 1.0/s2;
  auto λ' <- λ + a*a*l;
  auto μ' <- (λ*μ + a*l*(x - c))/λ';
  return (μ', 1.0/λ');
}

/**
 * Update the parameters of an inverse-gamma distribution that is part
 * of a normal inverse-gamma joint distribution.
 *
 * - x: The variate.
 * - μ: Mean.
 * - λ: Precision.
 * - α: Prior shape of the inverse-gamma.
 * - β: Prior scale of the inverse-gamma.
 *
 * Returns: the posterior hyperparameters `α'` and `β'`.
 */
function update_lazy_normal_inverse_gamma(x:Expression<Real>, μ:Expression<Real>,
    λ:Expression<Real>, α:Expression<Real>, β:Expression<Real>) ->
    (Expression<Real>, Expression<Real>) {
  return (α + 0.5, β + 0.5*pow(x - μ, 2.0)*λ);
}

/**
 * Update the parameters of a multivariate Gaussian distribution with a
 * multivariate Gaussian likelihood.
 *
 * - x: The variate.
 * - μ: Prior mean.
 * - Σ: Prior covariance.
 * - S: Likelihood covariance.
 *
 * Returns: the posterior hyperparameters `μ'` and `Σ'`.
 */
function update_lazy_multivariate_gaussian_multivariate_gaussian(
    x:Expression<Real[_]>, μ:Expression<Real[_]>, Σ:Expression<Real[_,_]>,
    S:Expression<Real[_,_]>) -> (Expression<Real[_]>, Expression<Real[_,_]>) {
  auto K' <- Σ*inv(llt(Σ + S));
  auto μ' <- μ + K'*(x - μ);
  auto Σ' <- Σ - K'*Σ;
  return (μ', Σ');
}

/**
 * Update the parameters of a multivariate Gaussian distribution with a 
 * linear transformation and multivariate Gaussian likelihood.
 *
 * - x: The variate.
 * - A: Scale.
 * - μ: Prior mean.
 * - Σ: Prior covariance.
 * - c: Offset.
 * - S: Likelihood covariance.
 *
 * Returns: the posterior hyperparameters `μ'` and `Σ'`.
 */
function update_lazy_linear_multivariate_gaussian_multivariate_gaussian(
    x:Expression<Real[_]>, A:Expression<Real[_,_]>, μ:Expression<Real[_]>,
    Σ:Expression<Real[_,_]>, c:Expression<Real[_]>,
    S:Expression<Real[_,_]>) -> (Expression<Real[_]>, Expression<Real[_,_]>) {
  auto K' <- Σ*transpose(A)*inv(llt(A*Σ*transpose(A) + S));
  auto μ' <- μ + K'*(x - A*μ - c);
  auto Σ' <- Σ - K'*A*Σ;
  return (μ', Σ');
}

/**
 * Update the parameters of a multivariate Gaussian distribution with a 
 * linear transformation involving a dot product, and a multivariate Gaussian
 * likelihood.
 *
 * - x: The variate.
 * - a: Scale.
 * - μ: Prior mean.
 * - Σ: Prior covariance.
 * - c: Offset.
 * - s2: Likelihood covariance.
 *
 * Returns: the posterior hyperparameters `μ'` and `Σ'`.
 */
function update_lazy_linear_multivariate_gaussian_gaussian(
    x:Expression<Real>, a:Expression<Real[_]>, μ:Expression<Real[_]>,
    Σ:Expression<Real[_,_]>, c:Expression<Real>, s2:Expression<Real>) ->
    (Expression<Real[_]>, Expression<Real[_,_]>) {
  auto k' <- Σ*a/(dot(a, Σ*a) + s2);
  auto μ' <- μ + k'*(x - dot(a, μ) - c);
  auto Σ' <- Σ - outer(k', a)*Σ;
  return (μ', Σ');
}

/**
 * Update the parameters of a normal inverse-gamma distribution with a
 * linear transformation involving a dot product, and Gaussian likelihood.
 *
 * - x: The variate.
 * - A: Scale.
 * - ν: Prior precision times mean.
 * - Λ: Prior precision.
 * - c: Offset.
 * - α: Prior shape of the inverse-gamma.
 * - γ: Prior scale accumulator.
 *
 * Returns: the posterior hyperparameters `μ'`, `Λ'`, `γ'`, `α'` and `β'`.
 */
//function update_lazy_linear_multivariate_normal_inverse_gamma_gaussian(
//    x:Expression<Real>, a:Expression<Real[_]>, ν:Expression<Real[_]>,
//    Λ:Expression<LLT>, c:Expression<Real>, α:Expression<Real>,
//    γ:Expression<Real>) -> (Expression<Real[_]>, Expression<LLT>,
//    Expression<Real>, Expression<Real>) {
//  auto Λ' <- rank_update(Λ, a, 1.0);
//  auto ν' <- ν + a*(x - c);
//  auto α' <- α + 0.5;
//  auto γ' <- γ + 0.5*pow(x - c, 2.0);
//  return (ν', Λ', α', γ');
//}
