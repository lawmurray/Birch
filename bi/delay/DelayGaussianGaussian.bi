/*
 * Delayed Gaussian-Gaussian random variate.
 */
class DelayGaussianGaussian(μ_0:DelayGaussian, σ2:Real) <
    DelayGaussian(μ_0.μ, μ_0.σ2 + σ2) {
  /**
   * Prior mean.
   */
  μ_0:DelayGaussian <- μ_0;

  /**
   * Marginal mean.
   */
  μ_m:Real <- μ;

  /**
   * Marginal variance.
   */
  σ2_m:Real <- σ2;

  function condition(x:Real) {
    (μ_0.μ, μ_0.σ2) <- update_gaussian_gaussian(x, μ_0.μ, μ_0.σ2, μ_m, σ2_m);
  }
}

function DelayGaussianGaussian(μ_0:DelayGaussian, σ2:Real) ->
    DelayGaussianGaussian {
  m:DelayGaussianGaussian(μ_0, σ2);
  return m;
}
