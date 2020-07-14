/**
 * Multivariate Gaussian distribution where the covariance is given as a
 * matrix multiplied by a scalar.
 */
final class ScalarMultivariateGaussian(μ:Expression<Real[_]>,
    Σ:Expression<LLT>, σ2:Expression<Real>) < Distribution<Real[_]> {
  /**
   * Mean.
   */
  μ:Expression<Real[_]> <- μ;
  
  /**
   * Covariance.
   */
  Σ:Expression<LLT> <- Σ;

  /**
   * Covariance scale.
   */
  σ2:Expression<Real> <- σ2;

  function rows() -> Integer {
    return μ.rows();
  }

  function supportsLazy() -> Boolean {
    return true;
  }

  function simulate() -> Real[_] {
    return simulate_multivariate_gaussian(μ.value(), σ2.value());
  }

  function simulateLazy() -> Real[_]? {
    return simulate_multivariate_gaussian(μ.get(), σ2.get());
  }
  
  function logpdf(x:Real[_]) -> Real {
    return logpdf_multivariate_gaussian(x, μ.value(), σ2.value());
  }

  function logpdfLazy(x:Expression<Real[_]>) -> Expression<Real>? {
    return logpdf_lazy_multivariate_gaussian(x, μ, σ2);
  }

  function graft() -> Distribution<Real[_]> {
    prune();
    s1:InverseGamma?;
    r:Distribution<Real[_]> <- this;
    
    /* match a template */
    if (s1 <- σ2.graftInverseGamma())? {
      r <- MultivariateNormalInverseGamma(μ, Σ, s1!);
    }

    return r;
  }

  function graftMultivariateGaussian() -> MultivariateGaussian? {
    prune();
    return Gaussian(μ, matrix(Σ)*σ2);
  }

  function graftMultivariateNormalInverseGamma(compare:Distribution<Real>) ->
      MultivariateNormalInverseGamma? {
    prune();
    s1:InverseGamma?;
    r:MultivariateNormalInverseGamma?;

    /* match a template */    
    if (s1 <- σ2.graftInverseGamma())? && s1! == compare {
      r <- MultivariateNormalInverseGamma(μ, Σ, s1!);
    }

    return r;
  }
}

/**
 * Create multivariate Gaussian distribution where the covariance is given
 * as a matrix multiplied by a scalar. This is usually used for establishing
 * a multivariate normal-inverse-gamma, where the final argument is
 * inverse-gamma distributed.
 */
function Gaussian(μ:Expression<Real[_]>, Σ:Expression<LLT>,
    σ2:Expression<Real>) -> ScalarMultivariateGaussian {
  return construct<ScalarMultivariateGaussian>(μ, Σ, σ2);
}

/**
 * Create multivariate Gaussian distribution where the covariance is given
 * as a matrix multiplied by a scalar. This is usually used for establishing
 * a multivariate normal-inverse-gamma, where the final argument is
 * inverse-gamma distributed.
 */
function Gaussian(μ:Expression<Real[_]>, Σ:Expression<LLT>, σ2:Real) ->
    ScalarMultivariateGaussian {
  return Gaussian(μ, Σ, box(σ2));
}

/**
 * Create multivariate Gaussian distribution where the covariance is given
 * as a matrix multiplied by a scalar. This is usually used for establishing
 * a multivariate normal-inverse-gamma, where the final argument is
 * inverse-gamma distributed.
 */
function Gaussian(μ:Expression<Real[_]>, Σ:LLT, σ2:Expression<Real>) ->
    ScalarMultivariateGaussian {
  return Gaussian(μ, box(Σ), σ2);
}

/**
 * Create multivariate Gaussian distribution where the covariance is given
 * as a matrix multiplied by a scalar. This is usually used for establishing
 * a multivariate normal-inverse-gamma, where the final argument is
 * inverse-gamma distributed.
 */
function Gaussian(μ:Expression<Real[_]>, Σ:LLT, σ2:Real) ->
      ScalarMultivariateGaussian {
  return Gaussian(μ, box(Σ), box(σ2));
}

/**
 * Create multivariate Gaussian distribution where the covariance is given
 * as a matrix multiplied by a scalar. This is usually used for establishing
 * a multivariate normal-inverse-gamma, where the final argument is
 * inverse-gamma distributed.
 */
function Gaussian(μ:Real[_], Σ:Expression<LLT>, σ2:Expression<Real>) ->
    ScalarMultivariateGaussian {
  return Gaussian(box(μ), Σ, σ2);
}

/**
 * Create multivariate Gaussian distribution where the covariance is given
 * as a matrix multiplied by a scalar. This is usually used for establishing
 * a multivariate normal-inverse-gamma, where the final argument is
 * inverse-gamma distributed.
 */
function Gaussian(μ:Real[_], Σ:Expression<LLT>, σ2:Real) ->
    ScalarMultivariateGaussian {
  return Gaussian(box(μ), Σ, box(σ2));
}

/**
 * Create multivariate Gaussian distribution where the covariance is given
 * as a matrix multiplied by a scalar. This is usually used for establishing
 * a multivariate normal-inverse-gamma, where the final argument is
 * inverse-gamma distributed.
 */
function Gaussian(μ:Real[_], Σ:LLT, σ2:Expression<Real>) ->
    ScalarMultivariateGaussian {
  return Gaussian(box(μ), box(Σ), σ2);
}

/**
 * Create multivariate Gaussian distribution where the covariance is given
 * as a matrix multiplied by a scalar. This is usually used for establishing
 * a multivariate normal-inverse-gamma, where the final argument is
 * inverse-gamma distributed.
 */
function Gaussian(μ:Real[_], Σ:LLT, σ2:Real) -> ScalarMultivariateGaussian {
  return Gaussian(box(μ), box(Σ), box(σ2));
}

/**
 * Create multivariate Gaussian distribution where the covariance is given
 * as a matrix multiplied by a scalar. This is usually used for establishing
 * a multivariate normal-inverse-gamma, where the final argument is
 * inverse-gamma distributed.
 */
function Gaussian(μ:Expression<Real[_]>, Σ:Expression<Real[_,_]>,
    σ2:Expression<Real>) -> ScalarMultivariateGaussian {
  m:ScalarMultivariateGaussian(μ, llt(Σ), σ2);
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
  return Gaussian(μ, llt(Σ), σ2);
}

/**
 * Create multivariate Gaussian distribution where the covariance is given
 * as a matrix multiplied by a scalar. This is usually used for establishing
 * a multivariate normal-inverse-gamma, where the final argument is
 * inverse-gamma distributed.
 */
function Gaussian(μ:Expression<Real[_]>, Σ:Real[_,_],
    σ2:Expression<Real>) -> ScalarMultivariateGaussian {
  return Gaussian(μ, llt(Σ), σ2);
}

/**
 * Create multivariate Gaussian distribution where the covariance is given
 * as a matrix multiplied by a scalar. This is usually used for establishing
 * a multivariate normal-inverse-gamma, where the final argument is
 * inverse-gamma distributed.
 */
function Gaussian(μ:Expression<Real[_]>, Σ:Real[_,_], σ2:Real) ->
      ScalarMultivariateGaussian {
  return Gaussian(μ, llt(Σ), σ2);
}

/**
 * Create multivariate Gaussian distribution where the covariance is given
 * as a matrix multiplied by a scalar. This is usually used for establishing
 * a multivariate normal-inverse-gamma, where the final argument is
 * inverse-gamma distributed.
 */
function Gaussian(μ:Real[_], Σ:Expression<Real[_,_]>, σ2:Expression<Real>) ->
    ScalarMultivariateGaussian {
  return Gaussian(μ, llt(Σ), σ2);
}

/**
 * Create multivariate Gaussian distribution where the covariance is given
 * as a matrix multiplied by a scalar. This is usually used for establishing
 * a multivariate normal-inverse-gamma, where the final argument is
 * inverse-gamma distributed.
 */
function Gaussian(μ:Real[_], Σ:Expression<Real[_,_]>, σ2:Real) ->
    ScalarMultivariateGaussian {
  return Gaussian(μ, llt(Σ), σ2);
}

/**
 * Create multivariate Gaussian distribution where the covariance is given
 * as a matrix multiplied by a scalar. This is usually used for establishing
 * a multivariate normal-inverse-gamma, where the final argument is
 * inverse-gamma distributed.
 */
function Gaussian(μ:Real[_], Σ:Real[_,_], σ2:Expression<Real>) ->
    ScalarMultivariateGaussian {
  return Gaussian(μ, llt(Σ), σ2);
}

/**
 * Create multivariate Gaussian distribution where the covariance is given
 * as a matrix multiplied by a scalar. This is usually used for establishing
 * a multivariate normal-inverse-gamma, where the final argument is
 * inverse-gamma distributed.
 */
function Gaussian(μ:Real[_], Σ:Real[_,_], σ2:Real) ->
    ScalarMultivariateGaussian {
  return Gaussian(μ, llt(Σ), σ2);
}
