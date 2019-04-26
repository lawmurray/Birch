/*
 * Delayed Gaussian-log-Gaussian random variate.
 */
final class DelayGaussianLogGaussian(future:Real?, futureUpdate:Boolean,
    m:DelayGaussian, s2:Real) < DelayLogGaussian(future, futureUpdate, m.μ,
    1.0/m.λ + s2) {
  /**
   * Mean.
   */
  m:DelayGaussian& <- m;

  /**
   * Likelihood precision.
   */
  l:Real <- 1.0/s2;

  function update(x:Real) {
    (m!.μ, m!.λ) <- update_gaussian_gaussian(log(x), m!.μ, m!.λ, l);
  }

  function downdate(x:Real) {
    (m!.μ, m!.λ) <- downdate_gaussian_gaussian(log(x), m!.μ, m!.λ, l);
  }
}

function DelayGaussianLogGaussian(future:Real?, futureUpdate:Boolean,
    μ:DelayGaussian, σ2:Real) -> DelayGaussianLogGaussian {
  m:DelayGaussianLogGaussian(future, futureUpdate, μ, σ2);
  μ.setChild(m);
  return m;
}
