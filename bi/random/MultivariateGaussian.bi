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
  
  function doGraft() {
    σ2:DelayInverseGamma? <- Σ.graftInverseGamma();
    m1:DelayMultivariateAffineNormalInverseGamma? <-
        μ.graftMultivariateAffineNormalInverseGamma(σ2);
    m2:DelayMultivariateNormalInverseGamma? <-
        μ.graftMultivariateNormalInverseGamma(σ2);
    m3:TransformMultivariateAffineGaussian? <-
        μ.graftMultivariateAffineGaussian();
    m4:DelayMultivariateGaussian? <- μ.graftMultivariateGaussian();
        
    if (σ2?) {
      if (m1?) {
        delay <- DelayMultivariateAffineNormalInverseGammaGaussian(this,
            m1!);
      } else if (m2?) {
        delay <- DelayMultivariateNormalInverseGammaGaussian(this, m2!);
      } else {
        delay <- DelayMultivariateInverseGammaGaussian(this, μ, σ2!);
      }
    } else if (m3?) {
      delay <- DelayMultivariateAffineGaussian(this, m3!, Σ);
    } else if (m4?) {
      delay <- DelayMultivariateGaussianGaussian(this, m4!, Σ);
    } else {
      delay <- DelayMultivariateGaussian(this, μ, Σ);
    }
  }

  function doGraftMultivariateGaussian() -> DelayMultivariateGaussian? {
    return DelayMultivariateGaussian(this, μ, Σ);
  }

  function doGraftMultivariateNormalInverseGamma(σ2:DelayInverseGamma) ->
      DelayMultivariateNormalInverseGamma? {
    S:TransformMultivarateScaledInverseGamma? <-
        Σ.graftMultivariateScaledInverseGamma(σ2);
    if (S?) {
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
