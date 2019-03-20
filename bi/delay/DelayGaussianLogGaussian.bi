/*
 * Delayed Gaussian-log-Gaussian random variate.
 */
class DelayGaussianLogGaussian(x:Random<Real>&, m:DelayGaussian, s2:Real) <
    DelayLogGaussian(x, m.μ, m.σ2 + s2) {
  /**
   * Prior mean.
   */
  m:DelayGaussian& <- m;

  /**
   * Likelihood variance.
   */
  s2:Real <- s2;

  function condition(x:Real) {
    (m!.μ, m!.σ2) <- update_gaussian_gaussian(log(x), m!.μ, m!.σ2, s2);
  }
}

function DelayGaussianLogGaussian(x:Random<Real>&, μ:DelayGaussian,
    σ2:Real) -> DelayGaussianLogGaussian {
  m:DelayGaussianLogGaussian(x, μ, σ2);
  μ.setChild(m);
  return m;
}
