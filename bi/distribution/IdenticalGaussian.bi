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
  
  function valueForward() -> Real[_] {
    assert !delay?;
    return simulate_multivariate_gaussian(μ, σ2);
  }

  function observeForward(x:Real[_]) -> Real {
    assert !delay?;
    return logpdf_multivariate_gaussian(x, μ, σ2);
  }
  
  function graft(force:Boolean) {
    if delay? {
      delay!.prune();
    } else {
      s2:DelayInverseGamma?;
      m1:TransformMultivariateLinearNormalInverseGamma?;
      m2:DelayIdenticalNormalInverseGamma?;
      m3:TransformMultivariateLinearGaussian?;
      m4:DelayMultivariateGaussian?;

      if (m1 <- μ.graftMultivariateLinearNormalInverseGamma())? && m1!.x.σ2 == σ2.getDelay() {
        delay <- DelayLinearIdenticalNormalInverseGammaGaussian(future, futureUpdate, m1!.A, m1!.x, m1!.c);
      } else if (m2 <- μ.graftIdenticalNormalInverseGamma())? && m2!.σ2 == σ2.getDelay() {
        delay <- DelayIdenticalNormalInverseGammaGaussian(future, futureUpdate, m2!);
      } else if (m3 <- μ.graftMultivariateLinearGaussian())? {
        delay <- DelayLinearMultivariateGaussianGaussian(future, futureUpdate, m3!.A, m3!.x, m3!.c, diagonal(σ2.value(), m3!.size()));
      } else if (m4 <- μ.graftMultivariateGaussian())? {
        delay <- DelayMultivariateGaussianGaussian(future, futureUpdate, m4!, diagonal(σ2, m4!.size()));
      } else {
        /* trigger a sample of μ, and double check that this doesn't cause
         * a sample of σ2 before we try creating an inverse-gamma Gaussian */
        μ.value();
        if (s2 <- σ2.graftInverseGamma())? {
          delay <- DelayIdenticalInverseGammaGaussian(future, futureUpdate, μ, s2!);
        } else if force {
          /* try a normal inverse gamma first, then a regular Gaussian */
          if !graftIdenticalNormalInverseGamma()? {
            delay <- DelayMultivariateGaussian(future, futureUpdate, μ.value(), diagonal(σ2, length(μ.value())));
          }
        }
      }
    }
  }

  function graftMultivariateGaussian() -> DelayMultivariateGaussian? {
    if delay? {
      delay!.prune();
    } else {
      m1:TransformMultivariateLinearGaussian?;
      m2:DelayMultivariateGaussian?;
      if (m1 <- μ.graftMultivariateLinearGaussian())? {
        delay <- DelayLinearMultivariateGaussianGaussian(future, futureUpdate, m1!.A, m1!.x,
            m1!.c, diagonal(σ2, length(m1!.c)));
      } else if (m2 <- μ.graftMultivariateGaussian())? {
        delay <- DelayMultivariateGaussianGaussian(future, futureUpdate, m2!, diagonal(σ2,
            length(m2!.μ)));
      } else {
        μ1:Real[_] <- μ.value();
        delay <- DelayMultivariateGaussian(future, futureUpdate, μ1, diagonal(σ2, length(μ1)));
      }
    }
    return DelayMultivariateGaussian?(delay);
  }

  function graftIdenticalNormalInverseGamma() -> DelayIdenticalNormalInverseGamma? {
    if delay? {
      delay!.prune();
    } else {
      s1:TransformScaledInverseGamma?;
      s2:DelayInverseGamma?;
      if (s1 <- σ2.graftScaledInverseGamma())? {
        μ1:Real[_] <- μ.value();
        D:Integer <- length(μ1);
        delay <- DelayIdenticalNormalInverseGamma(future, futureUpdate, μ1, diagonal(s1!.a2, D), s1!.σ2);
      } else if (s2 <- σ2.graftInverseGamma())? {
        μ1:Real[_] <- μ.value();
        D:Integer <- length(μ1);
        delay <- DelayIdenticalNormalInverseGamma(future, futureUpdate, μ1, identity(D), s2!);
      }
    }
    return DelayIdenticalNormalInverseGamma?(delay);
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
