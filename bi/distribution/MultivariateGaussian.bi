/*
 * ed multivariate Gaussian random variate.
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
    return simulate_multivariate_gaussian(μ, Σ);
  }
  
  function logpdf(x:Real[_]) -> Real {
    return logpdf_multivariate_gaussian(x, μ, Σ);
  }

  function logpdfLazy(x:Expression<Real[_]>) -> Expression<Real>? {
    return logpdf_lazy_multivariate_gaussian(x, μ, Σ);
  }

  function updateLazy(x:Expression<Real[_]>) {
    //
  }

  function graft() -> Distribution<Real[_]> {
    prune();
    m1:TransformLinearMultivariate<MultivariateGaussian>?;
    m2:MultivariateGaussian?;
    if (m1 <- μ.graftLinearMultivariateGaussian())? {
      return LinearMultivariateGaussianMultivariateGaussian(m1!.A, m1!.x,
          m1!.c, Σ);
    } else if (m2 <- μ.graftMultivariateGaussian())? {
      return MultivariateGaussianMultivariateGaussian(m2!, Σ);
    } else {
      return this;
    }
  }

  function graftMultivariateGaussian() -> MultivariateGaussian? {
    prune();
    m1:TransformLinearMultivariate<MultivariateGaussian>?;
    m2:MultivariateGaussian?;
    if (m1 <- μ.graftLinearMultivariateGaussian())? {
      return LinearMultivariateGaussianMultivariateGaussian(m1!.A, m1!.x,
          m1!.c, Σ);
    } else if (m2 <- μ.graftMultivariateGaussian())? {
      return MultivariateGaussianMultivariateGaussian(m2!, Σ);
    } else {
      return this;
    }
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
