/*
 * Delayed multivariate Gaussian random variate.
 */
class DelayMultivariateGaussian(μ:Real[_], Σ:Real[_,_]) <
    DelayValue<Real[_]> {
  /**
   * Mean.
   */
  μ:Real[_] <- μ;

  /**
   * Covariance.
   */
  Σ:Real[_,_] <- Σ;

  function size() -> Integer {
    return length(μ);
  }

  function simulate() -> Real[_] {
    return simulate_multivariate_gaussian(μ, Σ);
  }
  
  function observe(x:Real[_]) -> Real {
    return observe_multivariate_gaussian(x, μ, Σ);
  }

  function pdf(x:Real[_]) -> Real {
    return pdf_multivariate_gaussian(x, μ, Σ);
  }
}

function DelayMultivariateGaussian(μ:Real[_], Σ:Real[_,_]) ->
    DelayMultivariateGaussian {
  m:DelayMultivariateGaussian(μ, Σ);
  return m;
}
