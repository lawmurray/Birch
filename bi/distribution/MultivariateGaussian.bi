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

  function simulate() -> Real[_] {
    return simulate_multivariate_gaussian(μ.value(), Σ.value());
  }
  
  function logpdf(x:Real[_]) -> Real {
    return logpdf_multivariate_gaussian(x, μ.value(), Σ.value());
  }

  function logpdfLazy(x:Expression<Real[_]>) -> Expression<Real>? {
    return logpdf_lazy_multivariate_gaussian(x, μ, Σ);
  }

  function updateLazy(x:Expression<Real[_]>) {
    //
  }

  function graft() -> Distribution<Real[_]> {
    if !hasValue() {
      prune();
      m1:TransformLinearMultivariate<MultivariateGaussian>?;
      m2:MultivariateGaussian?;
      r:Distribution<Real[_]>?;
    
      /* match a template */
      if (m1 <- μ.graftLinearMultivariateGaussian())? {
        r <- LinearMultivariateGaussianMultivariateGaussian(m1!.A, m1!.x, m1!.c, Σ);
      } else if (m2 <- μ.graftMultivariateGaussian())? {
        r <- MultivariateGaussianMultivariateGaussian(m2!, Σ);
      }

      /* finalize, and if not valid, use default template */
      if !r? || !r!.graftFinalize() {
        r <- GraftedMultivariateGaussian(μ, Σ);
        r!.graftFinalize();
      }
      return r!;
    } else {
      return this;
    }
  }

  function graftMultivariateGaussian() -> MultivariateGaussian? {
    if !hasValue() {
      prune();
      m1:TransformLinearMultivariate<MultivariateGaussian>?;
      m2:MultivariateGaussian?;
      r:MultivariateGaussian?;
    
      /* match a template */
      if (m1 <- μ.graftLinearMultivariateGaussian())? {
        r <- LinearMultivariateGaussianMultivariateGaussian(m1!.A, m1!.x, m1!.c, Σ);
      } else if (m2 <- μ.graftMultivariateGaussian())? {
        r <- MultivariateGaussianMultivariateGaussian(m2!, Σ);
      }

      /* finalize, and if not valid, use default template */
      if !r? || !r!.graftFinalize() {
        r <- GraftedMultivariateGaussian(μ, Σ);
        r!.graftFinalize();
      }
      return r;
    } else {
      return nil;
    }
  }

  function graftFinalize() -> Boolean {
    assert false;  // should have been replaced during graft
    return false;
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
