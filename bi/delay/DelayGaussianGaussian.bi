/*
 * Delayed Gaussian-Gaussian random variate.
 */
class DelayGaussianGaussian(x:Random<Real>&, m:DelayGaussian, s2:Real) <
    DelayGaussian(x, m.μ, m.σ2 + s2) {
  /**
   * Prior mean.
   */
  m:DelayGaussian& <- m;

  /**
   * Likelihood variance.
   */
  s2:Real <- s2;

  function condition(x:Real) {
    (m!.μ, m!.σ2) <- update_gaussian_gaussian(x, m!.μ, m!.σ2, s2);
  }
}

function DelayGaussianGaussian(x:Random<Real>&, μ:DelayGaussian,
    σ2:Real) -> DelayGaussianGaussian {
  m:DelayGaussianGaussian(x, μ, σ2);
  μ.setChild(m);
  return m;
}
