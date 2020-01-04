/*
 * ed multivariate Gaussian random variate.
 */
class MultivariateGaussian(future:Real[_]?, futureUpdate:Boolean,
    μ:Expression<Real[_]>, Σ:Expression<Real[_,_]>) <
    Distribution<Real[_]>(future, futureUpdate) {
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

  function graft() {
    if delay? {
      delay!.prune();
    } else {
      m1:TransformLinearMultivariate<MultivariateGaussian>?;
      m3:MultivariateGaussian?;
      if (m1 <- μ.graftLinearMultivariateGaussian())? {
        delay <- LinearMultivariateGaussianMultivariateGaussian(future,
            futureUpdate, m1!.A, m1!.x, m1!.c, Σ);
      } else if (m3 <- μ.graftMultivariateGaussian())? {
        delay <- MultivariateGaussianMultivariateGaussian(future, futureUpdate, m3!, Σ);
      } else {
        /* try a normal inverse gamma first, then a regular Gaussian */
        if !graftMultivariateNormalInverseGamma()? {
          delay <- MultivariateGaussian(future, futureUpdate, μ, Σ);
        }
      }
    }
  }

  function graftMultivariateGaussian() -> MultivariateGaussian? {
    if delay? {
      delay!.prune();
    } else {
      m1:TransformLinearMultivariate<MultivariateGaussian>?;
      m2:MultivariateGaussian?;
      if (m1 <- μ.graftLinearMultivariateGaussian())? {
        delay <- LinearMultivariateGaussianMultivariateGaussian(future,
            futureUpdate, m1!.A, m1!.x, m1!.c, Σ);
      } else if (m2 <- μ.graftMultivariateGaussian())? {
        delay <- MultivariateGaussianMultivariateGaussian(future, futureUpdate, m2!, Σ);
      } else {
        delay <- MultivariateGaussian(future, futureUpdate, μ, Σ);
      }
    }
    return MultivariateGaussian?(delay);
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "MultivariateGaussian");
    buffer.set("μ", μ);
    buffer.set("Σ", Σ);
  }
}

function MultivariateGaussian(future:Real[_]?, futureUpdate:Boolean,
    μ:Real[_], Σ:Real[_,_]) ->
    MultivariateGaussian {
  m:MultivariateGaussian(future, futureUpdate, μ, Σ);
  return m;
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
