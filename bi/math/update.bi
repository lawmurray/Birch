/**
 * Update the parameters of a Gaussian distribution with a Gaussian
 * likelihood.
 *
 * - x: The variate.
 * - μ: Prior mean.
 * - σ2: Prior variance.
 * - μ_m: Marginal mean.
 * - σ2_m: Marginal variance.
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
 */
function update_affine_gaussian_gaussian(x:Real, a:Real, μ:Real,
    σ2:Real, μ_m:Real, σ2_m:Real) -> (Real, Real) {
  k:Real <- σ2*a/σ2_m;
  return (μ + k*(x - μ_m), σ2 - k*a*σ2);
}

/**
 * Update the parameters of an inverse-gama distribution that is part
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
function update_gaussian_normal_inverse_gamma(x:Real, μ:Real, a2:Real,
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
function update_affine_gaussian_normal_inverse_gamma(a:Real, x:Real,
    c:Real, μ:Real, a2:Real, α:Real, β:Real) -> (Real, Real, Real, Real) {
  y:Real <- x - c;
  λ:Real <- 1.0/a2;
  μ_1:Real <- (λ*μ + a*y)/(λ + a*a);
  λ_1:Real <- λ + a*a;
  α_1:Real <- α + 0.5;
  β_1:Real <- β + 0.5*(y*y + μ*μ*λ - μ_1*μ_1*λ_1);
  
  return (μ_1, 1.0/λ_1, α_1, β_1);
}
