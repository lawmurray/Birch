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
 * Downdate the parameters of a scaled Gamma distribution with an exponential
 * likelihood.
 *
 * - x: The variate.
 * - a: Constant scale.
 * - k': Posterior shape.
 * - θ': Posterior scale.
 *
 * Returns: the prior hyperparameters `k` and `θ`.
 */
function downdate_scaled_gamma_exponential(x:Real, a:Real, k':Real, θ':Real) -> (Real, Real) {
  return (k' - 1.0, θ'/(1.0 - x*a*θ'));
}

/**
 * Downdate the parameters of an inverse-gamma distribution with a Weibull
 * likelihood with known shape.
 *
 * - x: The variate.
 * - k: Likelihood shape.
 * - α': Posterior shape.
 * - β': Posterior scale.
 *
 * Returns: the prior hyperparameters `α` and `β`.
 */
function downdate_inverse_gamma_weibull(x:Real, k:Real, α':Real, β':Real) -> (Real, Real) {
  return (α' - 1.0, β' - pow(x, k));
}

/**
 * Downdate the parameters of a scaled inverse-gamma distribution with a
 * Weibull likelihood with known shape.
 *
 * - x: The variate.
 * - k: Likelihood shape.
 * - a: Constant scale.
 * - α': Posterior shape.
 * - β': Posterior scale.
 *
 * Returns: the prior hyperparameters `α` and `β`.
 */
function downdate_scaled_inverse_gamma_weibull(x:Real, k:Real, a:Real, α':Real, β':Real) -> (Real, Real) {
  return (α' - 1.0, β' - pow(x, k)/a);
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
function downdate_multivariate_gaussian_multivariate_gaussian(x:Real[_], μ':Real[_],
    Σ':Real[_,_], S:Real[_,_]) -> (Real[_], Real[_,_]) {
  auto K <- Σ'*inv(llt(Σ' - S));
  auto μ <- μ' + K*(x - μ');
  auto Σ <- Σ' - K*Σ';
  return (μ, Σ);
}

/**
 * Downdate the parameters of a multivariate Gaussian distribution with a 
 * linear transformation and multivariate Gaussian likelihood.
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
function downdate_linear_multivariate_gaussian_multivariate_gaussian(x:Real[_],
    A:Real[_,_], μ':Real[_], Σ':Real[_,_], c:Real[_], S:Real[_,_]) ->
    (Real[_], Real[_,_]) {
  auto K <- Σ'*transpose(A)*inv(llt(A*Σ'*transpose(A) - S));
  auto μ <- μ' + K*(x - A*μ' - c);
  auto Σ <- Σ' - K*A*Σ';
  return (μ, Σ);
}

/**
 * Downdate the parameters of an inverse-gamma distribution with a linear
 * scaling and Gaussian likelihood.
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
    Λ:LLT, α':Real, β':Real) -> (Real, Real) {
  D:Integer <- length(x);
  return (α' - 0.5*D, β' - 0.5*dot(x - μ, Λ*(x - μ)));
}

/**
 * Downdate the parameters of an inverse-gamma distribution with a Gaussian
 * likelihood.
 *
 * - x: The variate.
 * - μ: Mean.
 * - α': Posterior shape of the inverse-gamma.
 * - β': Posterior scale of the inverse-gamma.
 *
 * Returns: the prior hyperparameters `α` and `β`.
 */
function downdate_inverse_gamma_multivariate_gaussian(x:Real[_], μ:Real[_],
    α':Real, β':Real) -> (Real, Real) {
  D:Integer <- length(x);
  return (α' - 0.5*D, β' - 0.5*dot(x - μ));
}

/**
 * Downdate the parameters of a normal inverse-gamma distribution with a
 * multivariate Gaussian likelihood.
 *
 * - x: The variate.
 * - ν': Posterior precision times mean.
 * - Λ': Posterior precision.
 * - α': Posterior shape of the inverse-gamma.
 * - γ': Posterior scale accumulator.
 *
 * Returns: the prior hyperparameters `μ`, `Λ`, `α`, and `γ`.
 */
function downdate_multivariate_normal_inverse_gamma_multivariate_gaussian(
    x:Real[_], ν':Real[_], Λ':LLT, α':Real, γ':Real) -> (Real[_], LLT, Real,
    Real) {
  D:Integer <- length(x);
  Λ:LLT <- rank_update(Λ', identity(rows(Λ')), -1.0);
  ν:Real[_] <- ν' - x;
  α:Real <- α' - 0.5*D;
  γ:Real <- γ' - 0.5*dot(x);
  return (ν, Λ, α, γ);
}

/**
 * Downdate the parameters of a normal inverse-gamma distribution with a
 * multivariate Gaussian likelihood and scaling.
 *
 * - x: The variate.
 * - A: Scale.
 * - ν': Posterior precision times mean.
 * - c: Offset.
 * - Λ': Posterior precision.
 * - α': Posterior shape of the inverse-gamma.
 * - γ': Posterior scale accumulator.
 *
 * Returns: the prior hyperparameters `μ`, `Λ`, `α`, and `γ`.
 */
function downdate_linear_multivariate_normal_inverse_gamma_multivariate_gaussian(
    x:Real[_], A:Real[_,_], ν':Real[_], c:Real[_], Λ':LLT, α':Real,
    γ':Real) -> (Real[_], LLT, Real, Real) {
  D:Integer <- length(x);
  Λ:LLT <- rank_update(Λ', transpose(A), -1.0);
  ν:Real[_] <- ν' - transpose(A)*(x - c);
  α:Real <- α' - 0.5*D;
  γ:Real <- γ' - 0.5*dot(x - c);
  return (ν, Λ, α, γ);
}

/**
 * Downdate the parameters of a matrix normal-inverse-gamma variate.
 *
 * - X: The variate.
 * - M: Mean.
 * - Σ: Covariance.
 * - α': Posterior variance shape.
 * - β': Posterior variance scales.
 *
 * Returns: the prior hyperparameters `α` and `β`.
 */
function downdate_matrix_normal_inverse_gamma(X:Real[_,_], M:Real[_,_], Σ:LLT,
    α':Real, β':Real[_]) -> (Real, Real[_]) {
  auto D <- rows(X);
  return (α' - 0.5*D, β' - 0.5*diagonal(transpose(X - M)*solve(Σ, X - M)));
}

/**
 * Downdate the parameters of a Gaussian variate with linear  transformation
 * of matrix-normal-inverse-gamma prior.
 *
 * - x: The variate.
 * - A: Scale.
 * - N': Posterior precision times mean matrix.
 * - C: Offset.
 * - Λ': Posterior precision.
 * - α': Posterior variance shape.
 * - γ': Posterior squared sum accumulators.
 *
 * Returns: the prioor hyperparameters `N`, `Λ`, `α` and `γ`.
 */
function downdate_linear_matrix_normal_inverse_gamma_matrix_gaussian(
    X:Real[_,_], A:Real[_,_], N':Real[_,_], C:Real[_,_], Λ':LLT, α':Real,
    γ':Real[_]) -> (Real[_,_], LLT, Real, Real[_]) {
  Λ:LLT <- rank_update(Λ', transpose(A), -1.0);
  N:Real[_,_] <- N' - transpose(A)*(X - C);
  α:Real<- α' - 0.5;
  γ:Real[_] <- γ' - 0.5*diagonal(transpose(X - C)*(X - C));
  return (N, Λ, α, γ);
}

/**
 * Downdate the parameters of a Gaussian variate with dot matrix normal
 * inverse-gamma prior.
 *
 * - X: The variate.
 * - A: Scale.
 * - N': Posterior precision times mean matrix.
 * - Λ': Posterior precision.
 * - α': Posterior variance shape.
 * - γ': Posterior squared sum accumulators.
 *
 * Returns: the prior hyperparameters `N`, `Λ`, `α` and `γ`.
 */
function downdate_linear_matrix_normal_inverse_gamma_matrix_gaussian(
    x:Real[_], a:Real[_], N':Real[_,_], Λ':LLT, α':Real, γ':Real[_]) ->
    (Real[_,_], LLT, Real, Real[_]) {
  Λ:LLT <- rank_update(Λ', a, -1.0);
  N:Real[_,_] <- N' - kronecker(a, transpose(x));
  α:Real <- α' - 0.5;
  γ:Real[_] <- γ' - 0.5*hadamard(x, x);
  return (N, Λ, α, γ);
}

/**
 * Downdate parameters for a linear-Gaussian ridge regression.
 *
 * - x: The variate.
 * - N': Posterior precision times mean for weights, where each column
 *       represents the mean of the weight for a separate output. 
 * - Λ': Common posterior precision.
 * - α': Common posterior weight and likelihood covariance shape.
 * - β': Posterior covariance scale accumulators.
 * - u: Input.
 *
 * Returns: the prior hyperparameters `N`, `Λ`, `γ`, and `β`.
 */
function downdate_ridge_regression(x:Real[_], N':Real[_,_], Λ':LLT, α':Real,
    γ':Real[_], u:Real[_]) -> (Real[_,_], LLT, Real, Real[_]) {
  Λ:LLT <- rank_update(Λ', u, -1.0);
  N:Real[_,_] <- N' - kronecker(u, transpose(x));
  α:Real <- α' - 0.5;
  γ:Real[_] <- γ' - 0.5*hadamard(x, x);
  return (N, Λ, α, γ);
}
