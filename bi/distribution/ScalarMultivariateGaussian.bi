/**
 * Multivariate Gaussian distribution where the covariance is given as a
 * matrix multiplied by a scalar.
 */
final class ScalarMultivariateGaussian(future:Real[_]?, futureUpdate:Boolean,
    μ:Expression<Real[_]>, Σ:Expression<Real[_,_]>, σ2:Expression<Real>) <
    Distribution<Real[_]>(future, futureUpdate) {
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

  function rows() -> Integer {
    return μ.rows();
  }

  function graft() -> Distribution<Real[_]> {
    prune();
    s1:InverseGamma?;
    if (s1 <- σ2.graftInverseGamma())? {
      return MultivariateNormalInverseGamma(future, futureUpdate, μ, Σ, s1!);
    } else {
      return MultivariateGaussian(future, futureUpdate, μ, Σ*σ2);
    }
  }

  function graftMultivariateGaussian() -> MultivariateGaussian? {
    prune();
    return MultivariateGaussian(future, futureUpdate, μ, Σ*σ2);
  }

  function graftMultivariateNormalInverseGamma() -> MultivariateNormalInverseGamma? {
    prune();
    s1:InverseGamma?;
    if (s1 <- σ2.graftInverseGamma())? {
      return MultivariateNormalInverseGamma(future, futureUpdate, μ, Σ, s1!);
    }
    return nil;
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
  m:ScalarMultivariateGaussian(nil, true, μ, Σ, σ2);
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
