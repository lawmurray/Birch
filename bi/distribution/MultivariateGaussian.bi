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
  
  function graft() {
    if delay? {
      delay!.prune();
    } else {
      m1:TransformMultivariateLinearGaussian?;
      m2:DelayMultivariateGaussian?;
      if (m1 <- μ.graftMultivariateLinearGaussian())? {
        delay <- DelayMultivariateLinearGaussianGaussian(x, m1!.A, m1!.x,
            m1!.c, Σ);
      } else if (m2 <- μ.graftMultivariateGaussian())? {
        delay <- DelayMultivariateGaussianGaussian(x, m2!, Σ);
      } else {
        delay <- DelayMultivariateGaussian(x, μ, Σ);
      }
    }
  }

  function graftMultivariateGaussian() -> DelayMultivariateGaussian? {
    if delay? {
      delay!.prune();
    } else {
      m1:TransformMultivariateLinearGaussian?;
      m2:DelayMultivariateGaussian?;
      if (m1 <- μ.graftMultivariateLinearGaussian())? {
        delay <- DelayMultivariateLinearGaussianGaussian(x, m1!.A, m1!.x,
            m1!.c, Σ);
      } else if (m2 <- μ.graftMultivariateGaussian())? {
        delay <- DelayMultivariateGaussianGaussian(x, m2!, Σ);
      } else {
        delay <- DelayMultivariateGaussian(x, μ, Σ);
      }
    }
    return DelayMultivariateGaussian?(delay);
  }

  function graftMultivariateNormalInverseGamma(σ2:Expression<Real>) ->
      DelayMultivariateNormalInverseGamma? {
    if delay? {
      delay!.prune();
      
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
        delay <- DelayMultivariateNormalInverseGamma(x, μ, S!.A, S!.σ2);
      }
      return DelayMultivariateNormalInverseGamma?(delay);
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
