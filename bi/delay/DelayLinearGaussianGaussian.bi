/*
 * Delayed linear-Gaussian-Gaussian random variate.
 */
final class DelayLinearGaussianGaussian(future:Real?, futureUpdate:Boolean,
    a:Real, m:DelayGaussian, c:Real, s2:Real) < DelayGaussian(future,
    futureUpdate, a*m.μ + c, a*a/m.λ + s2) {
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
   * Likelihood precision.
   */
  l:Real <- 1.0/s2;

  function update(x:Real) {
    (m!.μ, m!.λ) <- update_linear_gaussian_gaussian(x, a, m!.μ, m!.λ, c, l);
  }

  function downdate(x:Real) {
    (m!.μ, m!.λ) <- downdate_linear_gaussian_gaussian(x, a, m!.μ, m!.λ, c, l);
  }
}

function DelayLinearGaussianGaussian(future:Real?, futureUpdate:Boolean,
    a:Real, μ:DelayGaussian, c:Real, σ2:Real) -> DelayLinearGaussianGaussian {
  m:DelayLinearGaussianGaussian(future, futureUpdate, a, μ, c, σ2);
  μ.setChild(m);
  return m;
}
