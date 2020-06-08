/**
 * Multivariate Gaussian distribution.
 *
 * !!! note
 *     See Gaussian for associated factory functions for the creation of
 *     MultivariateGaussian objects.
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

  function rows() -> Integer {
    return μ.rows();
  }

  function supportsLazy() -> Boolean {
    return true;
  }

  function simulate() -> Real[_] {
    return simulate_multivariate_gaussian(μ.value(), Σ.value());
  }

  function simulateLazy() -> Real[_]? {
    return simulate_multivariate_gaussian(μ.pilot(), Σ.pilot());
  }
  
  function logpdf(x:Real[_]) -> Real {
    return logpdf_multivariate_gaussian(x, μ.value(), Σ.value());
  }

  function logpdfLazy(x:Expression<Real[_]>) -> Expression<Real>? {
    return logpdf_lazy_multivariate_gaussian(x, μ, Σ);
  }

  function graft() -> Distribution<Real[_]> {
    prune();
    m1:TransformLinearMultivariate<MultivariateGaussian>?;
    m2:MultivariateGaussian?;
    r:Distribution<Real[_]> <- this;
    
    /* match a template */
    if (m1 <- μ.graftLinearMultivariateGaussian())? {
      r <- LinearMultivariateGaussianMultivariateGaussian(m1!.A, m1!.x, m1!.c, Σ);
    } else if (m2 <- μ.graftMultivariateGaussian())? {
      r <- MultivariateGaussianMultivariateGaussian(m2!, Σ);
    }

    return r;
  }

  function graftMultivariateGaussian() -> MultivariateGaussian? {
    prune();
    m1:TransformLinearMultivariate<MultivariateGaussian>?;
    m2:MultivariateGaussian?;
    r:MultivariateGaussian <- this;
    
    /* match a template */
    if (m1 <- μ.graftLinearMultivariateGaussian())? {
      r <- LinearMultivariateGaussianMultivariateGaussian(m1!.A, m1!.x, m1!.c, Σ);
    } else if (m2 <- μ.graftMultivariateGaussian())? {
      r <- MultivariateGaussianMultivariateGaussian(m2!, Σ);
    }

    return r;
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "MultivariateGaussian");
    buffer.set("μ", μ);
    buffer.set("Σ", Σ);
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
  return Gaussian(μ, box(Σ));
}

/**
 * Create multivariate Gaussian distribution.
 */
function Gaussian(μ:Real[_], Σ:Expression<Real[_,_]>) ->
    MultivariateGaussian {
  return Gaussian(box(μ), Σ);
}

/**
 * Create multivariate Gaussian distribution.
 */
function Gaussian(μ:Real[_], Σ:Real[_,_]) -> MultivariateGaussian {
  return Gaussian(box(μ), box(Σ));
}
