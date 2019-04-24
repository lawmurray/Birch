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
 * Downdate the parameters of a scaled Gamma distribution with a Poisson
 * likelihood.
 *
 * - x: The variate.
 * - a: Scale.
 * - k': Posterior shape.
 * - θ': Posterior scale.
 *
 * Returns: the prior hyperparameters `k` and `θ`.
 */
function downdate_scaled_gamma_poisson(x:Integer, a:Real, k':Real, θ':Real) ->
    (Real, Real) {
  return (k' - x, θ'/(1.0 - a*θ'));
}

/**
 * Downdate the parameters of a Gamma distribution with an exponential
 * likelihood.
 *
 * - x: The variate.
 * - k': Posterior shape.
 * - θ': Posterior scale.
 *
 * Returns: the prior hyperparameters `k` and `θ`.
 */
function downdate_gamma_exponential(x:Real, k':Real, θ':Real) ->
    (Real, Real) {
  return (k' - 1.0, θ'/(1.0 - x*θ'));
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
 * - λ': Prior precision.
 * - l: Likelihood precision.
 *
 * Returns: the prior hyperparameters `μ` and `σ2`.
 */
function downdate_gaussian_gaussian(x:Real, μ':Real, λ':Real, l:Real) ->
    (Real, Real) {
  λ:Real <- λ' - l;
  μ:Real <- (λ'*μ' - l*x)/λ;
  return (μ, λ);
}

/**
 * Downdate the parameters of a Gaussian distribution with a Gaussian
 * likelihood and scaling.
 *
 * - x: The variate.
 * - a: Scale.
 * - μ': Posterior mean.
 * - λ': Posterior variance.
 * - c: Offset.
 * - l: Likelihood precision.
 *
 * Returns: the prior hyperparameters `μ` and `λ`.
 */
function downdate_linear_gaussian_gaussian(x:Real, a:Real, μ':Real, λ':Real,
    c:Real, l:Real) -> (Real, Real) {
  λ:Real <- λ' - a*a*l;
  μ:Real <- (λ'*μ' - a*l*(x - c))/λ;
  return (μ, λ);
}

/**
 * Downdate the parameters of an inverse-gamma distribution that is part
 * of a normal inverse-gamma joint distribution.
 *
 * - x: The variate.
 * - μ: Mean.
 * - λ: Precision.
 * - α': Posterior shape of the inverse-gamma.
 * - β': Posterior scale of the inverse-gamma.
 *
 * Returns: the prior hyperparameters `α` and `β`.
 */
function downdate_normal_inverse_gamma(x:Real, μ:Real, λ:Real, α':Real,
    β':Real) -> (Real, Real) {
  return (α' - 0.5, β' - 0.5*pow(x - μ, 2.0)*λ);
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
 * - λ': Posterior precision.
 * - α': Posterior shape of the inverse-gamma.
 * - β': Posterior scale of the inverse-gamma.
 *
 * Returns: the prior hyperparameters `μ`, `λ`, `α` and `β`.
 */
function downdate_normal_inverse_gamma_gaussian(x:Real, μ':Real, λ':Real,
    α':Real, β':Real) -> (Real, Real, Real, Real) {
  λ:Real <- λ' - 1.0;
  μ:Real <- (λ'*μ' - x)/λ;
  α:Real <- α' - 0.5;
  β:Real <- β' - 0.5*(λ/λ')*pow(x - μ, 2.0);
  
  return (μ, λ, α, β);
}

/**
 * Downdate the parameters of a normal inverse-gamma distribution with a
 * Gaussian likelihood.
 *
 * - a: Scale.
 * - x: The variate.
 * - c: Offset.
 * - μ': Posterior mean.
 * - λ': Posterior precision.
 * - α': Posterior shape of the inverse-gamma.
 * - β': Posterior scale of the inverse-gamma.
 *
 * Returns: the prior hyperparameters `μ`, `λ`, `α` and `β`.
 */
function downdate_linear_normal_inverse_gamma_gaussian(a:Real, x:Real,
    c:Real, μ':Real, λ':Real, α':Real, β':Real) -> (Real, Real, Real, Real) {
  y:Real <- x - c;
  λ:Real <- λ' - a*a;
  μ:Real <- (λ'*μ' - a*y)/λ;
  α:Real <- α' - 0.5;
  β:Real <- β' - 0.5*(y*y + λ*μ*μ - λ'*μ'*μ');
  
  return (μ, λ, α, β);
}

/**
 * Downdate the parameters of an inverse-gamma distribution with a
 * gamma likelihood.
 *
 * - x: The variate.
 * - k: Shape of the gamma.
 * - α': Posterior shape of the inverse-gamma.
 * - β': Posterior scale of the inverse-gamma.
 *
 * Returns: the prior hyperparameters `α` and `β`.
 */
function downdate_inverse_gamma_gamma(x:Real, k:Real, α':Real, β':Real) ->
    (Real, Real) {
  return (α' - k, β' - x);
}

/**
 * Downdate the parameters of a multivariate Gaussian distribution with a
 * multivariate Gaussian likelihood.
 *
 * - x: The variate.
 * - μ': Posterior mean.
 * - Σ': Posterior covariance.
 * - S: Likelihood covariance.
 *
 * Returns: the prior hyperparameters `μ` and `Σ`.
 */
function downdate_multivariate_gaussian_gaussian(x:Real[_], μ':Real[_],
    Σ':Real[_,_], S:Real[_,_]) -> (Real[_], Real[_,_]) {
  auto K <- Σ'*cholinv(Σ' - S);
  auto μ <- μ' + K*(x - μ');
  auto Σ <- Σ' - K*Σ';
  return (μ, Σ);
}

/**
 * Downdate the parameters of a multivariate Gaussian distribution with a 
 * multivariate Gaussian likelihood and scaling.
 *
 * - x: The variate.
 * - A: Scale.
 * - μ': Posterior mean.
 * - Σ': Posterior covariance.
 * - c: Offset.
 * - S: Likelihood covariance.
 *
 * Returns: the prior hyperparameters `μ` and `Σ`.
 */
function downdate_multivariate_linear_gaussian_gaussian(x:Real[_],
    A:Real[_,_], μ':Real[_], Σ':Real[_,_], c:Real[_], S:Real[_,_]) ->
    (Real[_], Real[_,_]) {
  auto K <- Σ'*trans(A)*cholinv(A*Σ'*trans(A) - S);
  auto μ <- μ' + K*(x - A*μ' - c);
  auto Σ <- Σ' - K*A*Σ';
  return (μ, Σ);
}

/**
 * Downdate the parameters of a multivariate Gaussian distribution with a 
 * univariate Gaussian likelihood and scaling.
 *
 * - x: The variate.
 * - a: Scale.
 * - μ': Posterior mean.
 * - Σ': Posterior covariance.
 * - c: Offset.
 * - s2: Likelihood variance.
 *
 * Returns: the prior hyperparameters `μ` and `Σ`.
 */
function downdate_multivariate_dot_gaussian_gaussian(x:Real, a:Real[_],
    μ':Real[_], Σ':Real[_,_], c:Real, s2:Real) -> (Real[_], Real[_,_]) {
  auto K <- Σ'*a/(dot(a, Σ'*a) - s2);
  auto μ <- μ' + K*(x - dot(a, μ') - c);
  auto Σ <- Σ' - K*trans(a)*Σ';
  return (μ, Σ);
}

/**
 * Downdate the parameters of an inverse-gamma distribution that is part
 * of a multivariate normal inverse-gamma joint distribution.
 *
 * - x: The variate.
 * - μ: Mean.
 * - Λ: Precision.
 * - α': Posterior shape of the inverse-gamma.
 * - β': Posterior scale of the inverse-gamma.
 *
 * Returns: the prior hyperparameters `α` and `β`.
 */
function downdate_multivariate_normal_inverse_gamma(x:Real[_], μ:Real[_],
    Λ:Real[_,_], α':Real, β':Real) -> (Real, Real) {
  D:Integer <- length(x);
  return (α' - 0.5*D, β' - 0.5*dot(x - μ, Λ*(x - μ)));
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
  D:Integer <- length(x);
  return (α' - 0.5*D, β' - 0.5*dot(x - μ));
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
  D:Integer <- length(x);
  Λ:Real[_,_] <- Λ' - identity(rows(Λ'));
  μ:Real[_] <- cholsolve(Λ, Λ'*μ' - x);
  α:Real <- α' - D*0.5;
  β:Real <- β' - 0.5*(dot(x) + dot(μ, Λ*μ) - dot(μ', Λ'*μ'));
  return (μ, Λ, α, β);
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
  D:Integer <- length(x);
  Λ:Real[_,_] <- Λ' - trans(A)*A;
  μ:Real[_] <- cholsolve(Λ, Λ'*μ' - trans(A)*(x - c));
  α:Real <- α' - D*0.5;
  β:Real <- β' - 0.5*(dot(x - c) + dot(μ, Λ*μ) - dot(μ', Λ'*μ'));
  return (μ, Λ, α, β);
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
  Λ:Real[_,_] <- Λ' - a*trans(a);
  μ:Real[_] <- cholsolve(Λ, Λ'*μ' - a*(x - c));
  α:Real <- α' - 0.5;
  β:Real <- β' - 0.5*(pow(x - c, 2.0) + dot(μ, Λ*μ) - dot(μ', Λ'*μ'));
  return (μ, Λ, α, β);
}
