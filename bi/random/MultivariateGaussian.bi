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
  
  function graft() -> Delay? {
    if (delay?) {
      return delay;
    } else {
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
  }

  function graftMultivariateGaussian() -> DelayMultivariateGaussian? {
    if (delay?) {
      return DelayMultivariateGaussian?(delay);
    } else {
      return DelayMultivariateGaussian(this, μ, Σ);
    }
  }

  function graftMultivariateNormalInverseGamma(σ2:Expression<Real>) ->
      DelayMultivariateNormalInverseGamma? {
    if (delay?) {
      m:DelayMultivariateNormalInverseGamma?;
      s2:DelayInverseGamma?;
      if (m <- DelayMultivariateNormalInverseGamma?(delay))? &&
         (s2 <- σ2.graftInverseGamma())? && m!.σ2 == s2! {
        return m;
      } else {
        return nil;
      }
    } else {
      S:TransformMultivariateScaledInverseGamma?;
      if (S <- Σ.graftMultivariateScaledInverseGamma(σ2))? {
        return DelayMultivariateNormalInverseGamma(this, μ, S!.A, S!.σ2);
      } else {
        return nil;
      }
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
