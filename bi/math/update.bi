/**
 * Update the parameters of a Beta distribution with a Bernoulli likelihood.
 *
 * - x: The variate.
 * - α: Prior first shape.
 * - β: Prior second shape.
 *
 * Returns: the posterior hyperparameters `α'` and `β'`.
 */
function update_beta_bernoulli(x:Boolean, α:Real, β:Real) -> (Real, Real) {
  if x {
    return (α + 1.0, β);
  } else {
    return (α, β + 1.0);
  }
}

/**
 * Update the parameters of a Beta distribution with a Binomial likelihood.
 *
 * - x: The variate.
 * - n: Number of trials.
 * - α: Prior first shape.
 * - β: Prior second shape.
 *
 * Returns: the posterior hyperparameters `α'` and `β'`.
 */
function update_beta_binomial(x:Integer, n:Integer, α:Real, β:Real) ->
    (Real, Real) {
  assert 0 <= x && x <= n;
  assert 0 <= n;
  assert 0.0 < α;
  assert 0.0 < β;
  return (α + x, β + (n - x));
}

/**
 * Update the parameters of a Beta distribution with a Negative Binomial likelihood.
 *
 * - x: The variate.
 * - k: Number of successes.
 * - α: Prior first shape.
 * - β: Prior second shape.
 *
 * Returns: the posterior hyperparameters `α'` and `β'`.
 */
function update_beta_negative_binomial(x:Integer, k:Integer, α:Real,
    β:Real) -> (Real, Real) {
  return (α + k, β + x);
}

/**
 * Update the parameters of a Gamma distribution with a Poisson likelihood.
 *
 * - x: The variate.
 * - k: Prior shape.
 * - θ: Prior scale.
 *
 * Returns: the posterior hyperparameters `k'` and `θ'`.
 */
function update_gamma_poisson(x:Integer, k:Real, θ:Real) -> (Real, Real) {
  return (k + x, θ/(θ + 1.0));
}

/**
 * Update the parameters of a scaled Gamma distribution with a Poisson
 * likelihood.
 *
 * - x: The variate.
 * - a: Scale.
 * - k: Prior shape.
 * - θ: Prior scale.
 *
 * Returns: the posterior hyperparameters `k'` and `θ'`.
 */
function update_scaled_gamma_poisson(x:Integer, a:Real, k:Real, θ:Real) ->
    (Real, Real) {
  return (k + x, θ/(a*θ + 1.0));
}

/**
 * Update the parameters of a Gamma distribution with an exponential
 * likelihood.
 *
 * - x: The variate.
 * - k: Prior shape.
 * - θ: Prior scale.
 *
 * Returns: the posterior hyperparameters `k'` and `θ'`.
 */
function update_gamma_exponential(x:Real, k:Real, θ:Real) -> (Real, Real) {
  return (k + 1.0, θ/(1.0 + x*θ));
}

/**
 * Update the parameters of a scaled Gamma distribution with an exponential
 * likelihood.
 *
 * - x: The variate.
 * - a: Constant scale.
 * - k: Prior shape.
 * - θ: Prior scale.
 *
 * Returns: the posterior hyperparameters `k'` and `θ'`.
 */
function update_scaled_gamma_exponential(x:Real, a:Real, k:Real, θ:Real) -> (Real, Real) {
  return (k + 1.0, θ/(1.0 + x*a*θ));
}

/**
 * Update the parameters of an inverse-gamma distribution with a Weibull
 * likelihood with known shape.
 *
 * - x: The variate.
 * - k: Likelihood shape.
 * - α: Prior shape.
 * - β: Prior scale.
 *
 * Returns: the posterior hyperparameters `α'` and `β'`.
 */
function update_inverse_gamma_weibull(x:Real, k:Real, α:Real, β:Real) -> (Real, Real) {
  return (α + 1.0, β + pow(x, k));
}

/**
 * Update the parameters of a scaled inverse-gamma distribution with a
 * Weibull likelihood with known shape.
 *
 * - x: The variate.
 * - k: Likelihood shape.
 * - a: Constant scale.
 * - α: Prior shape.
 * - β: Prior scale.
 *
 * Returns: the posterior hyperparameters `α'` and `β'`.
 */
function update_scaled_inverse_gamma_weibull(x:Real, k:Real, a:Real, α:Real, β:Real) -> (Real, Real) {
  return (α + 1.0, β + pow(x, k)/a);
}

/**
 * Update the parameters of a Dirichlet distribution with a categorical
 * likelihood.
 *
 * - x: The variate.
 * - α: Prior concentrations.
 *
 * Returns: the posterior hyperparameters `α'`.
 */
function update_dirichlet_categorical(x:Integer, α:Real[_]) -> Real[_] {
  auto α' <- α;
  α'[x] <- α'[x] + 1;
  return α';
}

/**
 * Update the parameters of a Dirichlet distribution with a multinomial
 * likelihood.
 *
 * - x: The variate.
 * - n: Number of trials.
 * - α': Prior concentrations.
 *
 * Returns: the posterior hyperparameters `α'`.
 */
function update_dirichlet_multinomial(x:Integer[_], n:Integer, α:Real[_]) ->
    Real[_] {
  assert sum(x) == n;
  return α + x;
}

/**
 * Update the parameters of a Gaussian distribution with a Gaussian
 * likelihood.
 *
 * - x: The variate.
 * - μ: Prior mean.
 * - λ: Prior precision.
 * - l: Likelihood precision.
 *
 * Returns: the posterior hyperparameters `μ'` and `λ'`.
 */
function update_gaussian_gaussian(x:Real, μ:Real, λ:Real, l:Real) ->
    (Real, Real) {
  λ':Real <- λ + l;
  μ':Real <- (λ*μ + l*x)/λ';
  return (μ', λ');
}

/**
 * Update the parameters of a Gaussian distribution with a Gaussian
 * likelihood and scaling.
 *
 * - x: The variate.
 * - a: Scale.
 * - μ: Prior mean.
 * - λ: Prior precision.
 * - c: Offset.
 * - l: Likelihood precision.
 *
 * Returns: the posterior hyperparameters `μ'` and `λ'`.
 */
function update_linear_gaussian_gaussian(x:Real, a:Real, μ:Real, λ:Real,
    c:Real, l:Real) -> (Real, Real) {
  λ':Real <- λ + a*a*l;
  μ':Real <- (λ*μ + a*l*(x - c))/λ';
  return (μ', λ');
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
function update_normal_inverse_gamma(x:Real, μ:Real, λ:Real, α:Real,
    β:Real) -> (Real, Real) {
  return (α + 0.5, β + 0.5*pow(x - μ, 2.0)*λ);
}

/**
 * Update the parameters of an inverse-gamma distribution with a
 * Gaussian likelihood.
 *
 * - x: The variate.
 * - μ: Mean.
 * - α: Prior shape of the inverse-gamma.
 * - β: Prior scale of the inverse-gamma.
 *
 * Returns: the posterior hyperparameters `α'` and `β'`.
 */
function update_inverse_gamma_gaussian(x:Real, μ:Real, α:Real, β:Real) ->
    (Real, Real) {
  return (α + 0.5, β + 0.5*pow(x - μ, 2.0));
}

/**
 * Update the parameters of a normal inverse-gamma distribution with a
 * Gaussian likelihood.
 *
 * - x: The variate.
 * - μ: Mean.
 * - λ: Precision.
 * - α: Prior shape of the inverse-gamma.
 * - β: Prior scale of the inverse-gamma.
 *
 * Returns: the posterior hyperparameters `μ'`, `λ'`, `α'` and `β'`.
 */
function update_normal_inverse_gamma_gaussian(x:Real, μ:Real, λ:Real,
    α:Real, β:Real) -> (Real, Real, Real, Real) {
  λ':Real <- λ + 1.0;
  μ':Real <- (λ*μ + x)/λ';
  α':Real <- α + 0.5;
  β':Real <- β + 0.5*(λ/λ')*pow(x - μ, 2.0);
  
  return (μ', λ', α', β');
}

/**
 * Update the parameters of a normal inverse-gamma distribution with a
 * Gaussian likelihood.
 *
 * - a: Scale.
 * - x: The variate.
 * - c: Offset.
 * - μ: Prior mean.
 * - λ: Prior precision.
 * - α: Prior shape of the inverse-gamma.
 * - β: Prior scale of the inverse-gamma.
 *
 * Returns: the posterior hyperparameters `μ'`, `λ'`, `α'` and `β'`.
 */
function update_linear_normal_inverse_gamma_gaussian(a:Real, x:Real, c:Real,
    μ:Real, λ:Real, α:Real, β:Real) -> (Real, Real, Real, Real) {
  y:Real <- x - c;
  λ':Real <- λ + a*a;
  μ':Real <- (λ*μ + a*y)/λ';
  α':Real <- α + 0.5;
  β':Real <- β + 0.5*(y*y + μ*μ*λ - μ'*μ'*λ');
  
  return (μ', λ', α', β');
}

/**
 * Update the parameters of an inverse-gamma distribution with a
 * gamma likelihood.
 *
 * - x: The variate.
 * - k: Shape of the gamma.
 * - α: Prior shape of the inverse-gamma.
 * - β: Prior scale of the inverse-gamma.
 *
 * Returns: the posterior hyperparameters `α'` and `β'`.
 */
function update_inverse_gamma_gamma(x:Real, k:Real, α:Real, β:Real) ->
    (Real, Real) {
  return (α + k, β + x);
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
function update_multivariate_gaussian_multivariate_gaussian(x:Real[_], μ:Real[_],
    Σ:Real[_,_], S:Real[_,_]) -> (Real[_], Real[_,_]) {
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
function update_linear_multivariate_gaussian_multivariate_gaussian(x:Real[_], A:Real[_,_],
    μ:Real[_], Σ:Real[_,_], c:Real[_], S:Real[_,_]) -> (Real[_], Real[_,_]) {
  auto K' <- Σ*transpose(A)*inv(llt(A*Σ*transpose(A) + S));
  auto μ' <- μ + K'*(x - A*μ - c);
  auto Σ' <- Σ - K'*A*Σ;
  return (μ', Σ');
}

/**
 * Update the parameters of an inverse-gamma distribution with a Gaussian
 * likelihood.
 *
 * - x: The variate.
 * - μ: Mean.
 * - α: Prior shape of the inverse-gamma.
 * - β: Prior scale of the inverse-gamma.
 *
 * Returns: the posterior hyperparameters `α'` and `β'`.
 */
function update_inverse_gamma_multivariate_gaussian(x:Real[_], μ:Real[_],
    α:Real, β:Real) -> (Real, Real) {
  D:Integer <- length(x);
  return (α + 0.5*D, β + 0.5*dot(x - μ));
}

/**
 * Update the parameters of an inverse-gamma distribution with a linear
 * scaling and Gaussian likelihood.
 *
 * - x: The variate.
 * - ν: Precision times mean.
 * - Λ: Precision.
 * - α: Prior shape of the inverse-gamma.
 * - β: Prior scale of the inverse-gamma.
 *
 * Returns: the posterior hyperparameters `α'` and `β'`.
 */
function update_multivariate_normal_inverse_gamma(x:Real[_], ν:Real[_],
    Λ:LLT, α:Real, β:Real) -> (Real, Real) {
  auto D <- length(x);
  auto μ <- solve(Λ, ν);
  return (α + 0.5*D, β + 0.5*dot(x - μ, Λ*(x - μ)));
}

/**
 * Update the parameters of a normal inverse-gamma distribution with a
 * multivariate Gaussian likelihood.
 *
 * - x: The variate.
 * - ν: Prior precision times mean.
 * - Λ: Prior precision.
 * - α: Prior shape of the inverse-gamma.
 * - γ: Prior scale accumulator.
 *
 * Returns: the posterior hyperparameters `μ'`, `Λ'`, `α'` and `γ'`.
 */
function update_multivariate_normal_inverse_gamma_multivariate_gaussian(
    x:Real[_], ν:Real[_], Λ:LLT, α:Real, γ:Real) -> (Real[_], LLT, Real,
    Real) {
  auto D <- length(x);
  auto Λ' <- rank_update(Λ, identity(rows(Λ)), 1.0);
  auto ν' <- ν + x;
  auto α' <- α + 0.5*D;
  auto γ' <- γ + 0.5*dot(x);
  return (ν', Λ', α', γ');
}

/**
 * Update the parameters of a normal inverse-gamma distribution with a
 * multivariate Gaussian likelihood and scaling.
 *
 * - x: The variate.
 * - A: Scale.
 * - ν: Prior precision times mean.
 * - c: Offset.
 * - Λ: Prior precision.
 * - α: Prior shape of the inverse-gamma.
 * - γ: Prior scale accumulator.
 *
 * Returns: the posterior hyperparameters `μ'`, `Λ'`, `γ'`, `α'` and `β'`.
 */
function update_linear_multivariate_normal_inverse_gamma_multivariate_gaussian(
    x:Real[_], A:Real[_,_], ν:Real[_], c:Real[_], Λ:LLT, α:Real, γ:Real) ->
    (Real[_], LLT, Real, Real) {
  auto D <- length(x);
  auto Λ' <- rank_update(Λ, transpose(A), 1.0);
  auto ν' <- ν + transpose(A)*(x - c);
  auto α' <- α + 0.5*D;
  auto γ' <- γ + 0.5*dot(x - c);
  return (ν', Λ', α', γ');
}

/**
 * Update the parameters of a matrix normal-inverse-gamma variate.
 *
 * - X: The variate.
 * - N: Precision times mean.
 * - Λ: Precision.
 * - α: Prior variance shape.
 * - γ: Prior variance scale accumulators.
 *
 * Returns: the posterior hyperparameters `α'` and `β'`.
 */
function update_matrix_normal_inverse_gamma(X:Real[_,_], N:Real[_,_], Λ:LLT,
    α:Real, γ:Real[_]) -> (Real, Real[_]) {
  auto D <- rows(X);
  auto M <- solve(Λ, N);
  auto β <- γ - 0.5*diagonal(transpose(M)*N);
  auto α' <- α + 0.5*D;
  auto β' <- β + 0.5*diagonal(transpose(X - M)*Λ*(X - M));
  return (α', β');
}

/**
 * Update the parameters of a Gaussian variate with
 * matrix-normal-inverse-gamma prior.
 *
 * - x: The variate.
 * - N: Prior precision times mean matrix.
 * - Λ: Prior precision.
 * - α: Prior variance shape.
 * - γ: Prior variance scale accumulators.
 *
 * Returns: the posterior hyperparameters `N'`, `Λ'`, `α'` and `γ'`.
 */
function update_matrix_normal_inverse_gamma_matrix_gaussian(
    X:Real[_,_], N:Real[_,_], Λ:LLT, α:Real, γ:Real[_]) ->
    (Real[_,_], LLT, Real, Real[_]) {
  auto D <- rows(X);
  auto Λ' <- rank_update(Λ, identity(rows(N)), 1.0);
  auto N' <- N + X;
  auto α' <- α + 0.5*D;
  auto γ' <- γ + 0.5*diagonal(transpose(X)*X);
  return (N', Λ', α', γ');
}

/**
 * Update the parameters of a Gaussian variate with linear transformation
 * of matrix-normal-inverse-gamma prior.
 *
 * - x: The variate.
 * - A: Scale.
 * - N: Prior precision times mean matrix.
 * - C: Offset.
 * - Λ: Prior precision.
 * - α: Prior variance shape.
 * - γ: Prior variance scale accumulators.
 *
 * Returns: the posterior hyperparameters `N'`, `Λ'`, `α'` and `γ'`.
 */
function update_linear_matrix_normal_inverse_gamma_matrix_gaussian(
    X:Real[_,_], A:Real[_,_], N:Real[_,_], C:Real[_,_], Λ:LLT, α:Real,
    γ:Real[_]) -> (Real[_,_], LLT, Real, Real[_]) {
  auto D <- rows(X);
  auto Λ' <- rank_update(Λ, transpose(A), 1.0);
  auto N' <- N + transpose(A)*(X - C);
  auto α' <- α + 0.5*D;
  auto γ' <- γ + 0.5*diagonal(transpose(X - C)*(X - C));
  return (N', Λ', α', γ');
}

/**
 * Update the parameters of a matrix normal-inverse-Wishart variate.
 *
 * - X: The variate.
 * - N: Precision times mean.
 * - Λ: Precision.
 * - V: Prior variance shape.
 * - k: Prior degrees of freedom.
 *
 * Returns: the posterior hyperparameters `V'` and `k'`.
 */
function update_matrix_normal_inverse_wishart(X:Real[_,_], N:Real[_,_], Λ:LLT,
    V:Real[_,_], k:Real) -> (Real[_,_], Real) {
  auto D <- rows(X);
  auto M <- solve(Λ, N);
  auto V' <- V + transpose(X - M)*(X - M);
  auto k' <- k + D;
  return (V', k');
}

/**
 * Update the parameters of a Gaussian variate with
 * matrix-normal-inverse-Wishart prior.
 *
 * - x: The variate.
 * - N: Prior precision times mean matrix.
 * - Λ: Prior precision.
 * - V: Prior variance shape.
 * - k: Prior degrees of freedom.
 *
 * Returns: the posterior hyperparameters `N'`, `Λ'`, `V'` and `k'`.
 */
function update_matrix_normal_inverse_wishart_matrix_gaussian(
    X:Real[_,_], N:Real[_,_], Λ:LLT, V:Real[_,_], k:Real) ->
    (Real[_,_], LLT, Real[_,_], Real) {
  auto D <- rows(X);
  auto Λ' <- rank_update(Λ, identity(rows(N)), 1.0);
  auto N' <- N + X;
  auto M <- solve(Λ, N);
  auto M' <- solve(Λ', N');
  auto V' <- V + transpose(X - M')*(X - M') + transpose(M' - M)*Λ*(M' - M);
  auto k' <- k + D;
  return (N', Λ', V', k');
}

/**
 * Update the parameters of a Gaussian variate with linear transformation
 * of matrix-normal-inverse-Wishart prior.
 *
 * - x: The variate.
 * - A: Scale.
 * - N: Prior precision times mean matrix.
 * - C: Offset.
 * - Λ: Prior precision.
 * - V: Prior variance shape.
 * - k: Prior degrees of freedom.
 *
 * Returns: the posterior hyperparameters `N'`, `Λ'`, `V'` and `k'`.
 */
function update_linear_matrix_normal_inverse_wishart_matrix_gaussian(
    X:Real[_,_], A:Real[_,_], N:Real[_,_], C:Real[_,_], Λ:LLT, V:Real[_,_],
    k:Real) -> (Real[_,_], LLT, Real[_,_], Real) {
  auto D <- rows(X);
  auto Λ' <- rank_update(Λ, transpose(A), 1.0);
  auto N' <- N + transpose(A)*(X - C);
  auto M <- solve(Λ, N);
  auto M' <- solve(Λ', N');
  auto V' <- V + transpose(X - A*M' - C)*(X - A*M' - C) + transpose(M' - M)*Λ*(M' - M);
  auto k' <- k + D;
  return (N', Λ', V', k');
}
