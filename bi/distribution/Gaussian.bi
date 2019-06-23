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

  function simulateForward() -> Real {
    assert !delay?;
    return simulate_gaussian(μ, σ2);
  }

  function logpdfForward(x:Real) -> Real {
    assert !delay?;
    return logpdf_gaussian(x, μ, σ2);
  }

  function graft() {
    if delay? {
      delay!.prune();
    } else {
      m1:TransformLinearNormalInverseGamma?;
      m2:TransformMultivariateDotNormalInverseGamma?;
      m3:DelayNormalInverseGamma?;
      m4:TransformLinearGaussian?;
      m5:TransformMultivariateDotGaussian?;
      m6:DelayGaussian?;
      s2:DelayInverseGamma?;

      if (m1 <- μ.graftLinearNormalInverseGamma(σ2))? {
        delay <- DelayLinearNormalInverseGammaGaussian(future, futureUpdate, m1!.a, m1!.x, m1!.c);
      } else if (m2 <- μ.graftMultivariateDotNormalInverseGamma(σ2))? {
        delay <- DelayMultivariateDotNormalInverseGammaGaussian(future, futureUpdate, m2!.a, m2!.x, m2!.c);
      } else if (m3 <- μ.graftNormalInverseGamma(σ2))? {
        delay <- DelayNormalInverseGammaGaussian(future, futureUpdate, m3!);
      } else if (m4 <- μ.graftLinearGaussian())? {
        delay <- DelayLinearGaussianGaussian(future, futureUpdate, m4!.a, m4!.x, m4!.c, σ2);
      } else if (m5 <- μ.graftMultivariateDotGaussian())? {
        delay <- DelayMultivariateDotGaussianGaussian(future, futureUpdate, m5!.a, m5!.x, m5!.c, σ2);
      } else if (m6 <- μ.graftGaussian())? {
        delay <- DelayGaussianGaussian(future, futureUpdate, m6!, σ2);
      } else {
        /* trigger a sample of μ, and double check that this doesn't cause
         * a sample of σ2 before we try creating an inverse-gamma Gaussian */
        μ.value();
        if (s2 <- σ2.graftInverseGamma())? {
          delay <- DelayInverseGammaGaussian(future, futureUpdate, μ, s2!);
        } else {
          delay <- DelayGaussian(future, futureUpdate, μ, σ2);
        }
      }
    }
  }

  function graftGaussian() -> DelayGaussian? {
    if delay? {
      delay!.prune();
    } else {
      m1:TransformLinearGaussian?;
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

  function graftNormalInverseGamma(σ2:Expression<Real>) ->
      DelayNormalInverseGamma? {
    if delay? {
      delay!.prune();
      
      m:DelayNormalInverseGamma?;
      s2:DelayInverseGamma?;
      if (m <- DelayNormalInverseGamma?(delay))? && σ2.hasDelay() &&
          σ2.getDelay()! == m!.σ2! {
        return m;
      } else {
        return nil;
      }
    } else {
      s1:TransformScaledInverseGamma?;
      s2:DelayInverseGamma?;
      if (s1 <- this.σ2.graftScaledInverseGamma(σ2))? {
        delay <- DelayNormalInverseGamma(future, futureUpdate, μ, s1!.a2, s1!.σ2);
      } else if this.σ2 == σ2 && (s2 <- this.σ2.graftInverseGamma())? {
        delay <- DelayNormalInverseGamma(future, futureUpdate, μ, 1.0, s2!);
      }
      return DelayNormalInverseGamma?(delay);
    }
  }

  function write(buffer:Buffer) {
    if delay? {
      delay!.write(buffer);
    } else {
      buffer.set("class", "Gaussian");
      buffer.set("μ", μ.value());
      buffer.set("σ2", σ2.value());
    }
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
