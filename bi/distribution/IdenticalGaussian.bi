/**
 * Multivariate Gaussian distribution with independent and identical
 * variance.
 */
final class IdenticalGaussian(μ:Expression<Real[_]>,
    σ2:Expression<Real>) < Distribution<Real[_]> {
  /**
   * Mean.
   */
  μ:Expression<Real[_]> <- μ;
  
  /**
   * Variance.
   */
  σ2:Expression<Real> <- σ2;

  function rows() -> Integer {
    return μ.rows();
  }

  function graft() {
    if delay? {
      delay!.prune();
    } else {
      s1:DelayInverseGamma?;
      m1:TransformLinearMultivariate<DelayMultivariateNormalInverseGamma>?;
      m2:DelayMultivariateNormalInverseGamma?;
      m3:TransformLinearMultivariate<DelayMultivariateGaussian>?;
      m4:DelayMultivariateGaussian?;

      if (m1 <- μ.graftLinearMultivariateNormalInverseGamma())? &&
           m1!.x.σ2 == σ2.getDelay() {
        delay <- DelayLinearMultivariateNormalInverseGammaMultivariateGaussian(
            future, futureUpdate, m1!.A, m1!.x, m1!.c);
      } else if (m2 <- μ.graftMultivariateNormalInverseGamma())? &&
          m2!.σ2 == σ2.getDelay() {
        delay <- DelayMultivariateNormalInverseGammaMultivariateGaussian(
            future, futureUpdate, m2!);
      } else if (m3 <- μ.graftLinearMultivariateGaussian())? {
        delay <- DelayLinearMultivariateGaussianMultivariateGaussian(future,
            futureUpdate, m3!.A, m3!.x, m3!.c, diagonal(σ2.value(), m3!.rows()));
      } else if (m4 <- μ.graftMultivariateGaussian())? {
        delay <- DelayMultivariateGaussianMultivariateGaussian(future,
            futureUpdate, m4!, diagonal(σ2, m4!.rows()));
      } else if (s1 <- σ2.graftInverseGamma())? {
        delay <- DelayMultivariateNormalInverseGamma(future, futureUpdate, μ,
            identity(μ.rows()), s1!);
      } else {
        delay <- DelayMultivariateGaussian(future, futureUpdate, μ,
            diagonal(σ2, μ.rows()));
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
            futureUpdate, m1!.A, m1!.x, m1!.c, diagonal(σ2, length(m1!.c)));
      } else if (m2 <- μ.graftMultivariateGaussian())? {
        delay <- DelayMultivariateGaussianMultivariateGaussian(future,
            futureUpdate, m2!, diagonal(σ2, m2!.rows()));
      } else {
        delay <- DelayMultivariateGaussian(future, futureUpdate, μ,
            diagonal(σ2, μ.rows()));
      }
    }
    return DelayMultivariateGaussian?(delay);
  }

  function graftMultivariateNormalInverseGamma() -> DelayMultivariateNormalInverseGamma? {
    if delay? {
      delay!.prune();
    } else {
      s1:DelayInverseGamma?;
      if (s1 <- σ2.graftInverseGamma())? {
        delay <- DelayMultivariateNormalInverseGamma(future, futureUpdate, μ,
            identity(μ.rows()), s1!);
      }
    }
    return DelayMultivariateNormalInverseGamma?(delay);
  }
}

/**
 * Create multivariate Gaussian distribution.
 */
function Gaussian(μ:Expression<Real[_]>, σ2:Expression<Real>) ->
    IdenticalGaussian {
  m:IdenticalGaussian(μ, σ2);
  return m;
}

/**
 * Create multivariate Gaussian distribution.
 */
function Gaussian(μ:Expression<Real[_]>, σ2:Real) -> IdenticalGaussian {
  return Gaussian(μ, Boxed(σ2));
}

/**
 * Create multivariate Gaussian distribution.
 */
function Gaussian(μ:Real[_], σ2:Expression<Real>) -> IdenticalGaussian {
  return Gaussian(Boxed(μ), σ2);
}

/**
 * Create multivariate Gaussian distribution.
 */
function Gaussian(μ:Real[_], σ2:Real) -> IdenticalGaussian {
  return Gaussian(Boxed(μ), Boxed(σ2));
}
