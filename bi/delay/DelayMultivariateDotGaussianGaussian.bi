/*
 * Delayed dot-Gaussian-Gaussian random variate. This is univariate, where
 * the prior over the mean is given by a dot product with a multivariate
 * Gaussian random variable, plus scalar.
 */
final class DelayMultivariateDotGaussianGaussian(x:Random<Real>&,
    a:Real[_], m:DelayMultivariateGaussian, c:Real, s2:Real) <
    DelayGaussian(x, dot(a, m.μ) + c, scalar(trans(a)*cholinv(m.Λ)*a) + s2) {
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
   * Likelihood precision.
   */
  l:Real <- 1.0/s2;

  function update(x:Real) {
    (m!.μ, m!.Λ) <- update_multivariate_dot_gaussian_gaussian(x, a,
        m!.μ, m!.Λ, c, l);
  }

  function downdate(x:Real) {
    (m!.μ, m!.Λ) <- downdate_multivariate_dot_gaussian_gaussian(x, a,
        m!.μ, m!.Λ, c, l);
  }
}

function DelayMultivariateDotGaussianGaussian(x:Random<Real>&,
    a:Real[_], μ:DelayMultivariateGaussian, c:Real, σ2:Real) ->
    DelayMultivariateDotGaussianGaussian {
  m:DelayMultivariateDotGaussianGaussian(x, a, μ, c, σ2);
  μ.setChild(m);
  return m;
}
