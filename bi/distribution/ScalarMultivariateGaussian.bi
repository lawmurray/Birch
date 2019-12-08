/**
 * Multivariate Gaussian distribution where the covariance is given as a
 * matrix multiplied by a scalar.
 */
final class ScalarMultivariateGaussian(μ:Expression<Real[_]>,
    Σ:Expression<Real[_,_]>, σ2:Expression<Real>) < Distribution<Real[_]> {
  /**
   * Mean.
   */
  μ:Expression<Real[_]> <- μ;
  
  /**
   * Covariance.
   */
  Σ:Expression<Real[_,_]> <- Σ;

  /**
   * Covariance scale.
   */
  σ2:Expression<Real> <- σ2;
  
  function valueForward() -> Real[_] {
    assert !delay?;
    return simulate_multivariate_gaussian(μ, Σ.value()*σ2.value());
  }

  function observeForward(x:Real[_]) -> Real {
    assert !delay?;
    return logpdf_multivariate_gaussian(x, μ, Σ.value()*σ2.value());
  }
  
  function graft(force:Boolean) {
    if delay? {
      delay!.prune();
    } else {
      s1:DelayInverseGamma?;
      if (s1 <- σ2.graftInverseGamma())? {
        delay <- DelayMultivariateNormalInverseGamma(future, futureUpdate, μ, Σ, s1!);
      } else if force {
        delay <- DelayMultivariateGaussian(future, futureUpdate, μ, Σ*σ2);
      }
    }
  }

  function graftMultivariateGaussian() -> DelayMultivariateGaussian? {
    if delay? {
      delay!.prune();
    } else {
      delay <- DelayMultivariateGaussian(future, futureUpdate, μ, Σ*σ2);
    }
    return DelayMultivariateGaussian?(delay);
  }

  function graftMultivariateNormalInverseGamma() -> DelayMultivariateNormalInverseGamma? {
    if delay? {
      delay!.prune();
    } else {
      s1:DelayInverseGamma?;
      if (s1 <- σ2.graftInverseGamma())? {
        delay <- DelayMultivariateNormalInverseGamma(future, futureUpdate, μ, Σ, s1!);
      }
    }
    return DelayMultivariateNormalInverseGamma?(delay);
  }
}

/**
 * Create multivariate Gaussian distribution where the covariance is given
 * as a matrix multiplied by a scalar. This is usually used for establishing
 * a multivariate normal-inverse-gamma, where the final argument is
 * inverse-gamma distributed.
 */
function Gaussian(μ:Expression<Real[_]>, Σ:Expression<Real[_,_]>,
    σ2:Expression<Real>) -> ScalarMultivariateGaussian {
  m:ScalarMultivariateGaussian(μ, Σ, σ2);
  return m;
}

/**
 * Create multivariate Gaussian distribution where the covariance is given
 * as a matrix multiplied by a scalar. This is usually used for establishing
 * a multivariate normal-inverse-gamma, where the final argument is
 * inverse-gamma distributed.
 */
function Gaussian(μ:Expression<Real[_]>, Σ:Expression<Real[_,_]>,
    σ2:Real) -> ScalarMultivariateGaussian {
  return Gaussian(μ, Σ, Boxed(σ2));
}

/**
 * Create multivariate Gaussian distribution where the covariance is given
 * as a matrix multiplied by a scalar. This is usually used for establishing
 * a multivariate normal-inverse-gamma, where the final argument is
 * inverse-gamma distributed.
 */
function Gaussian(μ:Expression<Real[_]>, Σ:Real[_,_],
    σ2:Expression<Real>) -> ScalarMultivariateGaussian {
  return Gaussian(μ, Boxed(Σ), σ2);
}

/**
 * Create multivariate Gaussian distribution where the covariance is given
 * as a matrix multiplied by a scalar. This is usually used for establishing
 * a multivariate normal-inverse-gamma, where the final argument is
 * inverse-gamma distributed.
 */
function Gaussian(μ:Expression<Real[_]>, Σ:Real[_,_], σ2:Real) ->
      ScalarMultivariateGaussian {
  return Gaussian(μ, Boxed(Σ), Boxed(σ2));
}

/**
 * Create multivariate Gaussian distribution where the covariance is given
 * as a matrix multiplied by a scalar. This is usually used for establishing
 * a multivariate normal-inverse-gamma, where the final argument is
 * inverse-gamma distributed.
 */
function Gaussian(μ:Real[_], Σ:Expression<Real[_,_]>,
    σ2:Expression<Real>) -> ScalarMultivariateGaussian {
  return Gaussian(Boxed(μ), Σ, σ2);
}

/**
 * Create multivariate Gaussian distribution where the covariance is given
 * as a matrix multiplied by a scalar. This is usually used for establishing
 * a multivariate normal-inverse-gamma, where the final argument is
 * inverse-gamma distributed.
 */
function Gaussian(μ:Real[_], Σ:Expression<Real[_,_]>, σ2:Real) ->
    ScalarMultivariateGaussian {
  return Gaussian(Boxed(μ), Σ, Boxed(σ2));
}

/**
 * Create multivariate Gaussian distribution where the covariance is given
 * as a matrix multiplied by a scalar. This is usually used for establishing
 * a multivariate normal-inverse-gamma, where the final argument is
 * inverse-gamma distributed.
 */
function Gaussian(μ:Real[_], Σ:Real[_,_], σ2:Expression<Real>) ->
    ScalarMultivariateGaussian {
  return Gaussian(Boxed(μ), Boxed(Σ), σ2);
}

/**
 * Create multivariate Gaussian distribution where the covariance is given
 * as a matrix multiplied by a scalar. This is usually used for establishing
 * a multivariate normal-inverse-gamma, where the final argument is
 * inverse-gamma distributed.
 */
function Gaussian(μ:Real[_], Σ:Real[_,_], σ2:Real) -> ScalarMultivariateGaussian {
  return Gaussian(Boxed(μ), Boxed(Σ), Boxed(σ2));
}
