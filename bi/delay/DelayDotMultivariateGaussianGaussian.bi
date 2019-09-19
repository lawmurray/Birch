/*
 * Delayed dot-Gaussian-Gaussian random variate. This is univariate, where
 * the prior over the mean is given by a dot product with a multivariate
 * Gaussian random variable, plus scalar.
 */
final class DelayDotMultivariateGaussianMultivariateGaussian(future:Real?,
    futureUpdate:Boolean, a:Real[_], m:DelayMultivariateGaussian, c:Real,
    s2:Real) < DelayGaussian(future, futureUpdate, dot(a, m.μ) + c,
    dot(a, m.Σ*a) + s2) {
  /**
   * Scale.
   */
  a:Real[_] <- a;
    
  /**
   * Mean.
   */
  m:DelayMultivariateGaussian& <- m;

  /**
   * Offset.
   */
  c:Real <- c;

  /**
   * Likelihood variance.
   */
  s2:Real <- s2;

  function update(x:Real) {
    (m!.μ, m!.Σ) <- update_dot_multivariate_gaussian_multivariate_gaussian(x, a, m!.μ,
        m!.Σ, c, s2);
  }

  function downdate(x:Real) {
    (m!.μ, m!.Σ) <- downdate_dot_multivariate_gaussian_multivariate_gaussian(x, a, m!.μ,
        m!.Σ, c, s2);
  }

  function write(buffer:Buffer) {
    buffer.set(value());
  }
}

function DelayDotMultivariateGaussianMultivariateGaussian(future:Real?,
    futureUpdate:Boolean, a:Real[_], μ:DelayMultivariateGaussian, c:Real,
    σ2:Real) -> DelayDotMultivariateGaussianMultivariateGaussian {
  m:DelayDotMultivariateGaussianMultivariateGaussian(future, futureUpdate, a, μ, c, σ2);
  μ.setChild(m);
  return m;
}
