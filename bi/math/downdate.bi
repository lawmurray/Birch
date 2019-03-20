/**
 * Downdate the parameters of a Beta distribution with a Bernoulli likelihood.
 *
 * - x: The variate.
 * - α': Posterior first shape.
 * - β': Posterior second shape.
 *
 * Returns: the prior hyperparameters `α` and `β`.
 */
function downdate_beta_bernoulli(x:Boolean, α':Real, β':Real) ->
    (Real, Real) {
  if x {
    return (α' - 1.0, β');
  } else {
    return (α', β' - 1.0);
  }
}

/**
 * Downdate the parameters of a Beta distribution with a Binomial likelihood.
 *
 * - x: The variate.
 * - n: Number of trials.
 * - α': Posterior first shape.
 * - β': Posterior second shape.
 *
 * Returns: the prior hyperparameters `α` and `β`.
 */
function downdate_beta_binomial(x:Integer, n:Integer, α':Real, β':Real) ->
    (Real, Real) {
  assert 0 <= x && x <= n;
  assert 0 <= n;
  assert 0.0 < α';
  assert 0.0 < β';
  return (α' - x, β' - (n - x));
}

/**
 * Downdate the parameters of a Beta distribution with a Negative Binomial
 * likelihood.
 *
 * - x: The variate.
 * - k: Number of successes.
 * - α': Posterior first shape.
 * - β': Posterior second shape.
 *
 * Returns: the prior hyperparameters `α` and `β`.
 */
function downdate_beta_negative_binomial(x:Integer, k:Integer, α':Real,
    β':Real) -> (Real, Real) {
  return (α' - k, β' - x);
}

/**
 * Downdate the parameters of a Gamma distribution with a Poisson likelihood.
 *
 * - x: The variate.
 * - k': Posterior shape.
 * - θ': Posterior scale.
 *
 * Returns: the prior hyperparameters `k` and `θ`.
 */
function downdate_gamma_poisson(x:Integer, k':Real, θ':Real) ->
    (Real, Real) {
  return (k' - x, θ'/(1.0 - θ'));
}

/**
 * Downdate the parameters of a Dirichlet distribution with a categorical
 * likelihood.
 *
 * - x: The variate.
 * - α': Posterior concentrations.
 *
 * Returns: the prior hyperparameters `α`.
 */
function downdate_dirichlet_categorical(x:Integer, α':Real[_]) -> Real[_] {
  auto α <- α';
  α[x] <- α[x] - 1;
  return α;
}

/**
 * Downdate the parameters of a Dirichlet distribution with a multinomial
 * likelihood.
 *
 * - x: The variate.
 * - n: Number of trials.
 * - α': Posterior concentrations.
 *
 * Returns: the prior hyperparameters `α`.
 */
function downdate_dirichlet_multinomial(x:Integer[_], n:Integer,
    α':Real[_]) -> Real[_] {
  assert sum(x) == n;
  return α' - x;
}

/**
 * Downdate the parameters of a Gaussian distribution with a Gaussian
 * likelihood.
 *
 * - x: The variate.
 * - μ': Prior mean.
 * - σ2': Prior variance.
 * - s2: Likelihood variance.
 *
 * Returns: the prior hyperparameters `μ` and `σ2`.
 */
function downdate_gaussian_gaussian(x:Real, μ':Real, σ2':Real, s2:Real) ->
    (Real, Real) {
  λ:Real <- λ' - l;
  μ:Real <- (λ'*μ' - ν*x)/λ;
  return (μ, λ);
}

/**
 * Downdate the parameters of a Gaussian distribution with a Gaussian
 * likelihood and scaling.
 *
 * - x: The variate.
 * - a: Scale.
 * - μ': Posterior mean.
 * - σ2': Posterior variance.
 * - μ_m: Prior marginal mean.
 * - σ2_m: Prior marginal variance.
 *
 * Returns: the prior hyperparameters `μ` and `σ2`.
 */
function downdate_linear_gaussian_gaussian(x:Real, a:Real, μ':Real, σ2':Real,
    μ_m:Real, σ2_m:Real) -> (Real, Real) {

}

/**
 * Downdate the parameters of an inverse-gamma distribution that is part
 * of a normal inverse-gamma joint distribution.
 *
 * - x: The variate.
 * - μ: Mean.
 * - a2: Variance.
 * - α': Posterior shape of the inverse-gamma.
 * - β': Posterior scale of the inverse-gamma.
 *
 * Returns: the prior hyperparameters `α` and `β`.
 */
function downdate_normal_inverse_gamma(x:Real, μ:Real, a2:Real, α':Real,
    β':Real) -> (Real, Real) {
  return (α' - 0.5, β' - 0.5*pow(x - μ, 2.0)/a2);
}

/**
 * Downdate the parameters of an inverse-gamma distribution with a
 * Gaussian likelihood.
 *
 * - x: The variate.
 * - μ: Mean.
 * - α': Posterior shape of the inverse-gamma.
 * - β': Posterior scale of the inverse-gamma.
 *
 * Returns: the prior hyperparameters `α` and `β`.
 */
function downdate_inverse_gamma_gaussian(x:Real, μ:Real, α':Real, β':Real) ->
    (Real, Real) {
  return (α' - 0.5, β' - 0.5*pow(x - μ, 2.0));
}

/**
 * Downdate the parameters of a normal inverse-gamma distribution with a
 * Gaussian likelihood.
 *
 * - x: The variate.
 * - μ': Posterior mean.
 * - a2': Posterior variance.
 * - α': Posterior shape of the inverse-gamma.
 * - β': Posterior scale of the inverse-gamma.
 *
 * Returns: the prior hyperparameters `μ`, `a2`, `α` and `β`.
 */
function downdate_normal_inverse_gamma_gaussian(x:Real, μ':Real, a2':Real,
    α':Real, β':Real) -> (Real, Real, Real, Real) {
  λ':Real <- 1.0/a2';
  λ:Real <- λ' + 1.0;
  μ:Real <- (μ'*λ' - x)/λ;
  α:Real <- α' - 0.5;
  β:Real <- β' - 0.5*(λ/λ')*pow(x - μ, 2.0);
  
  return (μ, 1.0/λ, α, β);
}

/**
 * Downdate the parameters of a normal inverse-gamma distribution with a
 * Gaussian likelihood.
 *
 * - a: Scale.
 * - x: The variate.
 * - c: Offset.
 * - μ': Posterior mean.
 * - a2': Posterior variance.
 * - α': Posterior shape of the inverse-gamma.
 * - β': Posterior scale of the inverse-gamma.
 *
 * Returns: the prior hyperparameters `μ`, `a2`, `α` and `β`.
 */
function downdate_linear_normal_inverse_gamma_gaussian(a:Real, x:Real,
    c:Real, μ':Real, a2':Real, α':Real, β':Real) -> (Real, Real, Real, Real) {
  y:Real <- x - c;
  λ':Real <- 1.0/a2';
  λ:Real <- λ' - a*a;
  μ:Real <- (μ'*λ' - a*y)/λ;
  α:Real <- α' - 0.5;
  β:Real <- β' - 0.5*(y*y + μ*μ*λ - μ'*μ'*λ');
  
  return (μ, 1.0/λ, α, β);
}

/**
 * Downdate the parameters of a multivariate Gaussian distribution with a
 * multivariate Gaussian likelihood.
 *
 * - x: The variate.
 * - μ': Posterior mean.
 * - Σ': Posterior variance.
 * - μ_m: Prior marginal mean.
 * - Σ_m: Prior marginal variance.
 *
 * Returns: the prior hyperparameters `μ` and `Σ`.
 */
function downdate_multivariate_gaussian_gaussian(x:Real[_], μ':Real[_],
    Σ':Real[_,_], μ_m':Real[_], Σ_m':Real[_,_]) -> (Real[_], Real[_,_]) {

}

/**
 * Downdate the parameters of a multivariate Gaussian distribution with a 
 * multivariate Gaussian likelihood and scaling.
 *
 * - x: The variate.
 * - A: Scale.
 * - μ': Posterior mean.
 * - Σ': Posterior variance.
 * - μ_m: Prior marginal mean.
 * - Σ_m: Prior marginal variance.
 *
 * Returns: the prior hyperparameters `μ` and `Σ`.
 */
function downdate_multivariate_linear_gaussian_gaussian(x:Real[_],
    A:Real[_,_], μ':Real[_], Σ':Real[_,_], μ_m:Real[_], Σ_m:Real[_,_]) ->
    (Real[_], Real[_,_]) {

}

/**
 * Downdate the parameters of a multivariate Gaussian distribution with a 
 * univariate Gaussian likelihood and scaling.
 *
 * - x: The variate.
 * - a: Scale.
 * - μ': Posterior mean.
 * - Σ': Posterior variance.
 * - μ_m: Prior marginal mean.
 * - σ2_m: Prior marginal variance.
 *
 * Returns: the prior hyperparameters `μ` and `Σ`.
 */
function downdate_multivariate_dot_gaussian_gaussian(x:Real, a:Real[_],
    μ':Real[_], Σ':Real[_,_], μ_m:Real, σ2_m:Real) -> (Real[_], Real[_,_]) {

}

/**
 * Downdate the parameters of an inverse-gamma distribution that is part
 * of a multivariate normal inverse-gamma joint distribution.
 *
 * - x: The variate.
 * - μ': Mean.
 * - Λ': Precision.
 * - α': Posterior shape of the inverse-gamma.
 * - β': Posterior scale of the inverse-gamma.
 *
 * Returns: the prior hyperparameters `α` and `β`.
 */
function downdate_multivariate_normal_inverse_gamma(x:Real[_], μ':Real[_],
    Λ':Real[_,_], α':Real, β':Real) -> (Real, Real) {

}

/**
 * Downdate the parameters of an inverse-gamma distribution with a
 * multivariate Gaussian likelihood.
 *
 * - x: The variate.
 * - μ: Mean.
 * - α': Posterior shape of the inverse-gamma.
 * - β': Posterior scale of the inverse-gamma.
 *
 * Returns: the prior hyperparameters `α` and `β`.
 */
function downdate_multivariate_inverse_gamma_gaussian(x:Real[_], μ:Real[_],
    α':Real, β':Real) -> (Real, Real) {

}

/**
 * Downdate the parameters of a normal inverse-gamma distribution with a
 * multivariate Gaussian likelihood.
 *
 * - x: The variate.
 * - μ': Posterior mean.
 * - Λ': Posterior precision.
 * - α': Posterior shape of the inverse-gamma.
 * - β': Posterior scale of the inverse-gamma.
 *
 * Returns: the prior hyperparameters `μ`, `Σ`, `α` and `β`.
 */
function downdate_multivariate_normal_inverse_gamma_gaussian(x:Real[_],
    μ':Real[_], Λ':Real[_,_], α':Real, β':Real) -> (Real[_], Real[_,_], Real,
    Real) {

}

/**
 * Downdate the parameters of a normal inverse-gamma distribution with a
 * multivariate Gaussian likelihood and scaling.
 *
 * - x: The variate.
 * - A: Scale.
 * - μ': Posterior mean.
 * - c: Offset.
 * - Λ': Posterior precision.
 * - α': Posterior shape of the inverse-gamma.
 * - β': Posterior scale of the inverse-gamma.
 *
 * Returns: the prior hyperparameters `μ`, `Σ`, `α` and `β`.
 */
function downdate_multivariate_linear_normal_inverse_gamma_gaussian(
    x:Real[_], A:Real[_,_], μ':Real[_], c:Real[_], Λ':Real[_,_], α':Real,
    β':Real) -> (Real[_], Real[_,_], Real, Real) {

}

/**
 * Downdate the parameters of a normal inverse-gamma distribution with a
 * univariate Gaussian likelihood and scaling.
 *
 * - x: The variate.
 * - a: Scale.
 * - μ': Posterior mean.
 * - c: Offset.
 * - Λ': Posterior precision.
 * - α': Posterior shape of the inverse-gamma.
 * - β': Posterior scale of the inverse-gamma.
 *
 * Returns: the prior hyperparameters `μ`, `Σ`, `α` and `β`.
 */
function downdate_multivariate_dot_normal_inverse_gamma_gaussian(
    x:Real, a:Real[_], μ':Real[_], c:Real, Λ':Real[_,_], α':Real, β':Real) ->
    (Real[_], Real[_,_], Real, Real) {

}
