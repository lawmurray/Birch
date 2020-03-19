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

  function rows() -> Integer {
    return μ.rows();
  }

  function simulate() -> Real[_] {
    return simulate_multivariate_gaussian(μ.value(), σ2.value());
  }
  
  function logpdf(x:Real[_]) -> Real {
    return logpdf_multivariate_gaussian(x, μ.value(), σ2.value());
  }

  function graft() -> Distribution<Real[_]> {
    prune();
    s1:InverseGamma?;
    r:Distribution<Real[_]>?;
    
    /* match a template */
    if (s1 <- σ2.graftInverseGamma())? {
      r <- MultivariateNormalInverseGamma(μ, Σ, s1!);
    }

    /* finalize, and if not valid, use default template */
    if !r? || !r!.graftFinalize() {
      r <- GraftedMultivariateGaussian(μ, Σ*σ2);
      r!.graftFinalize();
    }
    return r!;
  }

  function graftMultivariateGaussian() -> MultivariateGaussian? {
    prune();
    auto r <- GraftedMultivariateGaussian(μ, Σ*σ2);
    r!.graftFinalize();
    return r;
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

    /* finalize, and if not valid, return nil */
    if !r? || !r!.graftFinalize() {
      r <- nil;
    }
    return r;
  }

  function graftFinalize() -> Boolean {
    assert false;  // should have been replaced during graft
    return false;
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
