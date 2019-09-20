/**
 * Gaussian distribution.
 */
final class Gaussian(μ:Expression<Real>, σ2:Expression<Real>) < Distribution<Real> {
  /**
   * Mean.
   */
  μ:Expression<Real> <- μ;
  
  /**
   * Variance.
   */
  σ2:Expression<Real> <- σ2;

  function valueForward() -> Real {
    assert !delay?;
    return simulate_gaussian(μ, σ2);
  }

  function observeForward(x:Real) -> Real {
    assert !delay?;
    return logpdf_gaussian(x, μ, σ2);
  }

  function graft(force:Boolean) {
    if delay? {
      delay!.prune();
    } else {
      m1:TransformLinear<DelayNormalInverseGamma>?;
      m2:TransformDot<DelayMultivariateNormalInverseGamma>?;
      m3:DelayNormalInverseGamma?;
      m4:TransformLinear<DelayGaussian>?;
      m5:TransformDot<DelayMultivariateGaussian>?;
      m6:DelayGaussian?;
      s2:DelayInverseGamma?;

      if (m1 <- μ.graftLinearNormalInverseGamma())? && m1!.x.σ2 == σ2.getDelay() {
        delay <- DelayLinearNormalInverseGammaGaussian(future, futureUpdate, m1!.a, m1!.x, m1!.c);
      } else if (m2 <- μ.graftDotMultivariateNormalInverseGamma())? && m2!.x.σ2 == σ2.getDelay() {
        delay <- DelayDotMultivariateNormalInverseGammaGaussian(future, futureUpdate, m2!.a, m2!.x, m2!.c);
      } else if (m3 <- μ.graftNormalInverseGamma())? && m3!.σ2 == σ2.getDelay() {
        delay <- DelayNormalInverseGammaGaussian(future, futureUpdate, m3!);
      } else if (m4 <- μ.graftLinearGaussian())? {
        delay <- DelayLinearGaussianGaussian(future, futureUpdate, m4!.a, m4!.x, m4!.c, σ2);
      } else if (m5 <- μ.graftDotMultivariateGaussian())? {
        delay <- DelayDotMultivariateGaussianMultivariateGaussian(future, futureUpdate, m5!.a, m5!.x, m5!.c, σ2);
      } else if (m6 <- μ.graftGaussian())? {
        delay <- DelayGaussianGaussian(future, futureUpdate, m6!, σ2);
      } else if (s2 <- σ2.graftInverseGamma())? {
        delay <- DelayNormalInverseGamma(future, futureUpdate, μ, 1.0, s2!);
      } else if force {
        delay <- DelayGaussian(future, futureUpdate, μ, σ2);
      }
    }
  }

  function graftGaussian() -> DelayGaussian? {
    if delay? {
      delay!.prune();
    } else {
      m1:TransformLinear<DelayGaussian>?;
      m2:DelayGaussian?;
      if (m1 <- μ.graftLinearGaussian())? {
        delay <- DelayLinearGaussianGaussian(future, futureUpdate, m1!.a, m1!.x, m1!.c, σ2);
      } else if (m2 <- μ.graftGaussian())? {
        delay <- DelayGaussianGaussian(future, futureUpdate, m2!, σ2);
      } else {
        delay <- DelayGaussian(future, futureUpdate, μ, σ2);
      }
    }
    return DelayGaussian?(delay);
  }

  function graftNormalInverseGamma() -> DelayNormalInverseGamma? {
    if delay? {
      delay!.prune();
    } else {
      s1:DelayInverseGamma?;
      if (s1 <- σ2.graftInverseGamma())? {
        delay <- DelayNormalInverseGamma(future, futureUpdate, μ, 1.0, s1!);
      }
    }
    return DelayNormalInverseGamma?(delay);
  }
}

/**
 * Create Gaussian distribution.
 */
function Gaussian(μ:Expression<Real>, σ2:Expression<Real>) -> Gaussian {
  m:Gaussian(μ, σ2);
  return m;
}

/**
 * Create Gaussian distribution.
 */
function Gaussian(μ:Expression<Real>, σ2:Real) -> Gaussian {
  return Gaussian(μ, Boxed(σ2));
}

/**
 * Create Gaussian distribution.
 */
function Gaussian(μ:Real, σ2:Expression<Real>) -> Gaussian {
  return Gaussian(Boxed(μ), σ2);
}

/**
 * Create Gaussian distribution.
 */
function Gaussian(μ:Real, σ2:Real) -> Gaussian {
  return Gaussian(Boxed(μ), Boxed(σ2));
}
