/**
 * Affine-Gaussian-log-Gaussian random variable with delayed sampling.
 */
class DelayAffineGaussianLogGaussian(x:Random<Real>, a:Real, μ_0:DelayGaussian,
    c:Real, σ2:Real) < DelayLogGaussian(x, a*μ_0.μ + c, a*a*μ_0.σ2 + σ2) {
  /**
   * Scale.
   */
  a:Real <- a;
    
  /**
   * Prior mean.
   */
  μ_0:DelayGaussian <- μ_0;

  /**
   * Offset.
   */
  c:Real <- c;

  /**
   * Marginal mean.
   */
  μ_m:Real <- μ;

  /**
   * Marginal variance.
   */
  σ2_m:Real <- σ2;

  function doCondition(x:Real) {
    (μ_0.μ, μ_0.σ2) <- update_affine_gaussian_gaussian(log(x), a, μ_0.μ, μ_0.σ2, μ_m, σ2_m);
  }
}

function DelayAffineGaussianLogGaussian(x:Random<Real>, a:Real,
    μ_0:DelayGaussian, c:Real, σ2:Real) -> DelayAffineGaussianLogGaussian {
  m:DelayAffineGaussianLogGaussian(x, a, μ_0, c, σ2);
  return m;
}
