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
  
  function doGraft() -> Delay? {
    σ2:DelayInverseGamma?;
    m1:DelayMultivariateAffineNormalInverseGamma?;
    m2:DelayMultivariateNormalInverseGamma?;
    m3:TransformMultivariateAffineGaussian?;
    m4:DelayMultivariateGaussian?;
        
    if (σ2 <- Σ.graftInverseGamma())? {
      if (m1 <- μ.graftMultivariateAffineNormalInverseGamma(σ2))? {
        return DelayMultivariateAffineNormalInverseGammaGaussian(this, m1!);
      } else if (m2 <- μ.graftMultivariateNormalInverseGamma(σ2))? {
        return DelayMultivariateNormalInverseGammaGaussian(this, m2!);
      } else {
        return DelayMultivariateInverseGammaGaussian(this, μ, σ2!);
      }
    } else if (m3 <- μ.graftMultivariateAffineGaussian())? {
      return DelayMultivariateAffineGaussian(this, m3!, Σ);
    } else if (m4 <- μ.graftMultivariateGaussian())? {
      return DelayMultivariateGaussianGaussian(this, m4!, Σ);
    } else {
      return DelayMultivariateGaussian(this, μ, Σ);
    }
  }

  function doGraftMultivariateGaussian() -> DelayMultivariateGaussian? {
    return DelayMultivariateGaussian(this, μ, Σ);
  }

  function doGraftMultivariateNormalInverseGamma(σ2:DelayInverseGamma) ->
      DelayMultivariateNormalInverseGamma? {
    S:TransformMultivariateScaledInverseGamma?;
    if (S <- Σ.graftMultivariateScaledInverseGamma(σ2))? {
      return DelayMultivariateNormalInverseGamma(this, μ, S!);
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
