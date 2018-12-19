/*
 * Delayed Gaussian-Gaussian random variate.
 */
class DelayGaussianGaussian(x:Random<Real>&, μ_0:DelayGaussian, σ2:Real) <
    DelayGaussian(x, μ_0.μ, μ_0.σ2 + σ2) {
  /**
   * Prior mean.
   */
  μ_0:DelayGaussian& <- μ_0;

  /**
   * Marginal mean.
   */
  μ_m:Real <- this.μ;

  /**
   * Marginal variance.
   */
  σ2_m:Real <- this.σ2;

  function condition(x:Real) {
    (μ_0!.μ, μ_0!.σ2) <- update_gaussian_gaussian(x, μ_0!.μ, μ_0!.σ2, μ_m, σ2_m);
  }
}

function DelayGaussianGaussian(x:Random<Real>&, μ_0:DelayGaussian,
    σ2:Real) -> DelayGaussianGaussian {
  m:DelayGaussianGaussian(x, μ_0, σ2);
  μ_0.setChild(m);
  return m;
}
