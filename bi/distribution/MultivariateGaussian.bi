/**
 * Multivariate Gaussian distribution.
 */
final class MultivariateGaussian(μ:Expression<Real[_]>, Σ:Expression<Real[_,_]>) <
    Distribution<Real[_]> {
  /**
   * Mean.
   */
  μ:Expression<Real[_]> <- μ;
  
  /**
   * Covariance.
   */
  Σ:Expression<Real[_,_]> <- Σ;
  
  function valueForward() -> Real[_] {
    assert !delay?;
    return simulate_multivariate_gaussian(μ, Σ);
  }

  function observeForward(x:Real[_]) -> Real {
    assert !delay?;
    return logpdf_multivariate_gaussian(x, μ, Σ);
  }
  
  function graft(force:Boolean) {
    if delay? {
      delay!.prune();
    } else {
      m1:TransformMultivariateLinearGaussian?;
      m2:DelayMultivariateGaussian?;
      if (m1 <- μ.graftMultivariateLinearGaussian())? {
        delay <- DelayMultivariateLinearGaussianGaussian(future, futureUpdate, m1!.A, m1!.x,
            m1!.c, Σ);
      } else if (m2 <- μ.graftMultivariateGaussian())? {
        delay <- DelayMultivariateGaussianGaussian(future, futureUpdate, m2!, Σ);
      } else if force {
        /* try a normal inverse gamma first, then a regular Gaussian */
        if !graftMultivariateNormalInverseGamma()? {
          delay <- DelayMultivariateGaussian(future, futureUpdate, μ, Σ);
        }
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
        delay <- DelayMultivariateLinearGaussianGaussian(future, futureUpdate, m1!.A, m1!.x,
            m1!.c, Σ);
      } else if (m2 <- μ.graftMultivariateGaussian())? {
        delay <- DelayMultivariateGaussianGaussian(future, futureUpdate, m2!, Σ);
      } else {
        delay <- DelayMultivariateGaussian(future, futureUpdate, μ, Σ);
      }
    }
    return DelayMultivariateGaussian?(delay);
  }

  function graftMultivariateNormalInverseGamma() ->
      DelayMultivariateNormalInverseGamma? {
    if delay? {
      delay!.prune();
    } else {
      S:TransformMultivariateScaledInverseGamma?;
      if (S <- Σ.graftMultivariateScaledInverseGamma())? {
        delay <- DelayMultivariateNormalInverseGamma(future, futureUpdate, μ, S!.A, S!.σ2);
      }
    }
    return DelayMultivariateNormalInverseGamma?(delay);
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
