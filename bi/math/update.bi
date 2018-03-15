/**
 * Update the parameters of a Gaussian distribution with a Gaussian
 * likelihood.
 *
 * - x: The variate.
 * - μ: Prior mean.
 * - σ2: Prior variance.
 * - μ_m: Marginal mean.
 * - σ2_m: Marginal variance.
 *
 * Returns: the updated parameters `μ` and `σ2`.
 */
function update_gaussian_gaussian(x:Real, μ:Real, σ2:Real,
    μ_m:Real, σ2_m:Real) -> (Real, Real) {
  k:Real <- σ2/σ2_m;
  return (μ + k*(x - μ_m), σ2 - k*σ2);
}

/**
 * Update the parameters of a Gaussian distribution with a Gaussian
 * likelihood and scaling.
 *
 * - x: The variate.
 * - a: Scale.
 * - μ: Prior mean.
 * - σ2: Prior variance.
 * - μ_m: Marginal mean.
 * - σ2_m: Marginal variance.
 *
 * Returns: the updated parameters `μ` and `σ2`.
 */
function update_affine_gaussian_gaussian(x:Real, a:Real, μ:Real,
    σ2:Real, μ_m:Real, σ2_m:Real) -> (Real, Real) {
  k:Real <- σ2*a/σ2_m;
  return (μ + k*(x - μ_m), σ2 - k*a*σ2);
}

/**
 * Update the parameters of an inverse-gamma distribution that is part
 * of a normal inverse-gamma joint distribution.
 *
 * - x: The variate.
 * - μ: Mean.
 * - a2: Variance.
 * - α: Shape of the inverse-gamma.
 * - β: Scale of the inverse-gamma.
 *
 * Returns: the updated parameters `α` and `β`.
 */
function update_normal_inverse_gamma(x:Real, μ:Real, a2:Real, α:Real,
    β:Real) -> (Real, Real) {
  return (α + 0.5, β + 0.5*pow(x - μ, 2.0)/a2);
}

/**
 * Update the parameters of an inverse-gamma distribution with a
 * Gaussian likelihood.
 *
 * - x: The variate.
 * - μ: Mean.
 * - α: Shape of the inverse-gamma.
 * - β: Scale of the inverse-gamma.
 *
 * Returns: the updated parameters `α` and `β`.
 */
function update_inverse_gamma_gaussian(x:Real, μ:Real, α:Real,
    β:Real) -> (Real, Real) {
  return (α + 0.5, β + 0.5*pow(x - μ, 2.0));
}

/**
 * Update the parameters of a normal inverse-gamma distribution with a
 * Gaussian likelihood.
 *
 * - x: The variate.
 * - μ: Mean.
 * - a2: Variance.
 * - α: Shape of the inverse-gamma.
 * - β: Scale of the inverse-gamma.
 *
 * Returns: the updated parameters `μ`, `a2`, `α` and `β`.
 */
function update_normal_inverse_gamma_gaussian(x:Real, μ:Real, a2:Real,
    α:Real, β:Real) -> (Real, Real, Real, Real) {
  λ:Real <- 1.0/a2;
  μ_1:Real <- (λ*μ + x)/(λ + 1.0);
  λ_1:Real <- λ + 1.0;
  α_1:Real <- α + 0.5;
  β_1:Real <- β + 0.5*(λ/λ_1)*pow(x - μ, 2.0);
  
  return (μ_1, 1.0/λ_1, α_1, β_1);
}

/**
 * Update the parameters of a normal inverse-gamma distribution with a
 * Gaussian likelihood.
 *
 * - a: Scale.
 * - x: The variate.
 * - c: Offset.
 * - μ: Mean.
 * - a2: Variance.
 * - α: Shape of the inverse-gamma.
 * - β: Scale of the inverse-gamma.
 *
 * Returns: the updated parameters `μ`, `a2`, `α` and `β`.
 */
function update_affine_normal_inverse_gamma_gaussian(a:Real, x:Real,
    c:Real, μ:Real, a2:Real, α:Real, β:Real) -> (Real, Real, Real, Real) {
  y:Real <- x - c;
  λ:Real <- 1.0/a2;
  μ_1:Real <- (λ*μ + a*y)/(λ + a*a);
  λ_1:Real <- λ + a*a;
  α_1:Real <- α + 0.5;
  β_1:Real <- β + 0.5*(y*y + μ*μ*λ - μ_1*μ_1*λ_1);
  
  return (μ_1, 1.0/λ_1, α_1, β_1);
}

/**
 * Update the parameters of a multivariate Gaussian distribution with a
 * multivariate Gaussian likelihood.
 *
 * - x: The variate.
 * - μ: Prior mean.
 * - Σ: Prior variance.
 * - μ_m: Marginal mean.
 * - Σ_m: Marginal variance.
 *
 * Returns: the updated parameters `μ` and `Σ`.
 */
function update_multivariate_gaussian_gaussian(x:Real[_], μ:Real[_],
    Σ:Real[_,_], μ_m:Real[_], Σ_m:Real[_,_]) -> (Real[_], Real[_,_]) {
  K:Real[_,_] <- Σ*inv(Σ_m);
  return (μ + K*(x - μ_m), Σ - K*Σ);
}

/**
 * Update the parameters of a multivariate Gaussian distribution with a 
 * multivariate Gaussian likelihood and scaling.
 *
 * - x: The variate.
 * - A: Scale.
 * - μ: Prior mean.
 * - Σ: Prior variance.
 * - μ_m: Marginal mean.
 * - Σ_m: Marginal variance.
 *
 * Returns: the updated parameters `μ` and `Σ`.
 */
function update_multivariate_affine_gaussian_gaussian(x:Real[_], A:Real[_,_],
    μ:Real[_], Σ:Real[_,_], μ_m:Real[_], Σ_m:Real[_,_]) -> (Real[_], Real[_,_]) {
  K:Real[_,_] <- Σ*trans(A)*inv(Σ_m);
  return (μ + K*(x - μ_m), Σ - K*A*Σ);
}

/**
 * Update the parameters of an inverse-gamma distribution that is part
 * of a multivariate normal inverse-gamma joint distribution.
 *
 * - x: The variate.
 * - μ: Mean.
 * - Σ: Variance.
 * - α: Shape of the inverse-gamma.
 * - β: Scale of the inverse-gamma.
 *
 * Returns: the updated parameters `α` and `β`.
 */
function update_multivariate_normal_inverse_gamma(x:Real[_], μ:Real[_],
    Σ:Real[_,_], α:Real, β:Real) -> (Real, Real) {
  D:Integer <- length(μ);
  return (α + 0.5*D, β + 0.5*dot(solve(chol(Σ), x - μ)));
}

/**
 * Update the parameters of an inverse-gamma distribution with a multivariate
 * Gaussian likelihood.
 *
 * - x: The variate.
 * - μ: Mean.
 * - α: Shape of the inverse-gamma.
 * - β: Scale of the inverse-gamma.
 *
 * Returns: the updated parameters `α` and `β`.
 */
function update_multivariate_inverse_gamma_gaussian(x:Real[_], μ:Real[_],
    α:Real, β:Real) -> (Real, Real) {
  D:Integer <- length(μ);
  return (α + D*0.5, β + 0.5*dot(x - μ));
}

/**
 * Update the parameters of a normal inverse-gamma distribution with a
 * multivariate Gaussian likelihood.
 *
 * - x: The variate.
 * - μ: Mean.
 * - Σ: Variance.
 * - α: Shape of the inverse-gamma.
 * - β: Scale of the inverse-gamma.
 *
 * Returns: the updated parameters `μ`, `Σ`, `α` and `β`.
 */
function update_multivariate_normal_inverse_gamma_gaussian(x:Real[_],
    μ:Real[_], Σ:Real[_,_], α:Real, β:Real) -> (Real[_], Real[_,_], Real, Real) {
  D:Integer <- length(μ);
  
  Λ:Real[_,_] <- inv(Σ);
  Λ_1:Real[_,_] <- Λ + identity(D);
  μ_1:Real[_] <- inv(Λ_1)*(Λ*μ + x);
  L:Real[_,_] <- chol(Λ);
  L_1:Real[_,_] <- chol(Λ_1);

  α_1:Real <- α + D*0.5;
  β_1:Real <- β + 0.5*(dot(x) + dot(trans(L)*μ) - dot(trans(L_1)*μ_1));

  return (μ_1, inv(Λ_1), α_1, β_1);
  ///@todo Introduce more efficient matrix operations to use here
}

/**
 * Update the parameters of a normal inverse-gamma distribution with a
 * multivariate Gaussian likelihood and scaling.
 *
 * - A: Scale.
 * - x: The variate.
 * - c: Offset.
 * - μ: Mean.
 * - Σ: Variance.
 * - α: Shape of the inverse-gamma.
 * - β: Scale of the inverse-gamma.
 *
 * Returns: the updated parameters `μ`, `Σ`, `α` and `β`.
 */
function update_multivariate_affine_normal_inverse_gamma_gaussian(
    A:Real[_,_], x:Real[_], c:Real[_], μ:Real[_], Σ:Real[_,_], α:Real,
    β:Real) -> (Real[_], Real[_,_], Real, Real) {
  D:Integer <- length(μ);
  
  Λ:Real[_,_] <- inv(Σ);
  Λ_1:Real[_,_] <- Λ + A*trans(A);
  μ_1:Real[_] <- inv(Λ_1)*(Λ*μ + A*(x - c));
  L:Real[_,_] <- chol(Λ);
  L_1:Real[_,_] <- chol(Λ_1);
  
  α_1:Real <- α + D*0.5;
  β_1:Real <- β + 0.5*(dot(x - c) + dot(trans(L)*μ) - dot(trans(L_1)*μ_1));

  return (μ_1, inv(Λ_1), α_1, β_1);
  ///@todo Introduce more efficient matrix operations to use here
}
