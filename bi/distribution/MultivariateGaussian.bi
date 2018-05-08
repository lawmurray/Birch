/**
 * Multivariate Gaussian distribution.
 */
class MultivariateGaussian(μ:Expression<Real[_]>, Σ:Expression<Real[_,_]>) <
    Distribution<Real[_]> {
  /**
   * Mean.
   */
  μ:Expression<Real[_]> <- μ;
  
  /**
   * Covariance.
   */
  Σ:Expression<Real[_,_]> <- Σ;
  
  function graft() -> DelayValue<Real[_]> {
    m1:TransformMultivariateLinearGaussian?;
    m2:DelayMultivariateGaussian?;
    if (m1 <- μ.graftMultivariateLinearGaussian())? {
      return DelayMultivariateLinearGaussianGaussian(m1!.A, m1!.x, m1!.c, Σ);
    } else if (m2 <- μ.graftMultivariateGaussian())? {
      return DelayMultivariateGaussianGaussian(m2!, Σ);
    } else {
      return DelayMultivariateGaussian(μ, Σ);
    }
  }

  function graftMultivariateGaussian() -> DelayMultivariateGaussian? {
    m1:TransformMultivariateLinearGaussian?;
    m2:DelayMultivariateGaussian?;
    if (m1 <- μ.graftMultivariateLinearGaussian())? {
      return DelayMultivariateLinearGaussianGaussian(m1!.A, m1!.x, m1!.c, Σ);
    } else if (m2 <- μ.graftMultivariateGaussian())? {
      return DelayMultivariateGaussianGaussian(m2!, Σ);
    } else {
      return DelayMultivariateGaussian(μ, Σ);
    }
  }

  function graftMultivariateNormalInverseGamma(σ2:Expression<Real>) ->
      DelayMultivariateNormalInverseGamma? {
    S:TransformMultivariateScaledInverseGamma?;
    if (S <- Σ.graftMultivariateScaledInverseGamma(σ2))? {
      return DelayMultivariateNormalInverseGamma(μ, S!.A, S!.σ2);
    } else {
      return nil;
    }
  }
}

/**
 * Create multivariate Gaussian distribution.
 */
function Gaussian(μ:Expression<Real[_]>, Σ:Expression<Real[_,_]>) ->
    MultivariateGaussian {
  m:MultivariateGaussian(μ, Σ);
  return m;
}

/**
 * Create multivariate Gaussian distribution.
 */
function Gaussian(μ:Expression<Real[_]>, Σ:Real[_,_]) ->
    MultivariateGaussian {
  return Gaussian(μ, Boxed(Σ));
}

/**
 * Create multivariate Gaussian distribution.
 */
function Gaussian(μ:Real[_], Σ:Expression<Real[_,_]>) ->
    MultivariateGaussian {
  return Gaussian(Boxed(μ), Σ);
}

/**
 * Create multivariate Gaussian distribution.
 */
function Gaussian(μ:Real[_], Σ:Real[_,_]) -> MultivariateGaussian {
  return Gaussian(Boxed(μ), Boxed(Σ));
}
