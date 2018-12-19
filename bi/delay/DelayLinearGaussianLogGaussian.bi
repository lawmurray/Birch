/*
 * Delayed linear-Gaussian-log-Gaussian random variate.
 */
class DelayLinearGaussianLogGaussian(x:Random<Real>&, a:Real,
    μ_0:DelayGaussian, c:Real, σ2:Real) < DelayLogGaussian(x, a*μ_0.μ + c,
    a*a*μ_0.σ2 + σ2) {
  /**
   * Scale.
   */
  a:Real <- a;
    
  /**
   * Prior mean.
   */
  μ_0:DelayGaussian& <- μ_0;

  /**
   * Offset.
   */
  c:Real <- c;

  /**
   * Marginal mean.
   */
  μ_m:Real <- this.μ;

  /**
   * Marginal variance.
   */
  σ2_m:Real <- this.σ2;

  function condition(x:Real) {
    (μ_0!.μ, μ_0!.σ2) <- update_linear_gaussian_gaussian(log(x), a, μ_0!.μ, μ_0!.σ2, μ_m, σ2_m);
  }
}

function DelayLinearGaussianLogGaussian(x:Random<Real>&, a:Real,
    μ_0:DelayGaussian, c:Real, σ2:Real) -> DelayLinearGaussianLogGaussian {
  m:DelayLinearGaussianLogGaussian(x, a, μ_0, c, σ2);
  μ_0.setChild(m);
  return m;
}
