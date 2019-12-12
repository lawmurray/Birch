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

  function rows() -> Integer {
    return μ.rows();
  }

  function graft() {
    if delay? {
      delay!.prune();
    } else {
      m1:TransformLinearMultivariate<DelayMultivariateGaussian>?;
      m3:DelayMultivariateGaussian?;
      if (m1 <- μ.graftLinearMultivariateGaussian())? {
        delay <- DelayLinearMultivariateGaussianMultivariateGaussian(future,
            futureUpdate, m1!.A, m1!.x, m1!.c, Σ);
      } else if (m3 <- μ.graftMultivariateGaussian())? {
        delay <- DelayMultivariateGaussianMultivariateGaussian(future, futureUpdate, m3!, Σ);
      } else {
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
      m1:TransformLinearMultivariate<DelayMultivariateGaussian>?;
      m2:DelayMultivariateGaussian?;
      if (m1 <- μ.graftLinearMultivariateGaussian())? {
        delay <- DelayLinearMultivariateGaussianMultivariateGaussian(future,
            futureUpdate, m1!.A, m1!.x, m1!.c, Σ);
      } else if (m2 <- μ.graftMultivariateGaussian())? {
        delay <- DelayMultivariateGaussianMultivariateGaussian(future, futureUpdate, m2!, Σ);
      } else {
        delay <- DelayMultivariateGaussian(future, futureUpdate, μ, Σ);
      }
    }
    return DelayMultivariateGaussian?(delay);
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
