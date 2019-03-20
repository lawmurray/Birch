/*
 * Delayed linear-Gaussian-Gaussian random variate.
 */
class DelayLinearGaussianGaussian(x:Random<Real>&, a:Real, m:DelayGaussian,
    c:Real, s2:Real) < DelayGaussian(x, a*m.μ + c, a*a*m.σ2 + s2) {
  /**
   * Scale.
   */
  a:Real <- a;
    
  /**
   * Mean.
   */
  m:DelayGaussian& <- m;

  /**
   * Offset.
   */
  c:Real <- c;

  /**
   * Likelihood variance.
   */
  s2:Real <- s2;

  function condition(x:Real) {
    (m!.μ, m!.σ2) <- update_linear_gaussian_gaussian(x, a, m!.μ, m!.σ2, c, s2);
  }
}

function DelayLinearGaussianGaussian(x:Random<Real>&, a:Real,
    μ:DelayGaussian, c:Real, σ2:Real) -> DelayLinearGaussianGaussian {
  m:DelayLinearGaussianGaussian(x, a, μ, c, σ2);
  μ.setChild(m);
  return m;
}
