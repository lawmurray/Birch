/**
 * Multivariate Gaussian distribution.
 */
class MultivariateGaussian(μ:Expression<Real[_]>, Σ:Expression<Real[_,_]>) <
    Random<Real[_]> {
  /**
   * Mean.
   */
  μ:Expression<Real[_]> <- μ;
  
  /**
   * Covariance.
   */
  Σ:Expression<Real[_,_]> <- Σ;
  
  function doGraft() -> DelayValue<Real[_]>? {
    m1:TransformMultivariateAffineGaussian?;
    m2:DelayMultivariateGaussian?;

    if (m1 <- μ.graftMultivariateAffineGaussian())? {
      return DelayMultivariateAffineGaussianGaussian(this, m1!.A, m1!.x, m1!.c, Σ);
    } else if (m2 <- μ.graftMultivariateGaussian())? {
      return DelayMultivariateGaussianGaussian(this, m2!, Σ);
    } else {
      return DelayMultivariateGaussian(this, μ, Σ);
    }
  }

  function doGraftMultivariateGaussian() -> DelayMultivariateGaussian? {
    return DelayMultivariateGaussian(this, μ, Σ);
  }

  function doGraftMultivariateNormalInverseGamma(σ2:Expression<Real>) ->
      DelayMultivariateNormalInverseGamma? {
    S:TransformMultivariateScaledInverseGamma?;
    if (S <- Σ.graftMultivariateScaledInverseGamma(σ2))? {
      return DelayMultivariateNormalInverseGamma(this, μ, S!.A, S!.σ2);
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
