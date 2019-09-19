/**
 * Log-Gaussian distribution.
 */
final class LogGaussian(μ:Expression<Real>, σ2:Expression<Real>) < Distribution<Real> {
  /**
   * Mean after log transformation.
   */
  μ:Expression<Real> <- μ;
  
  /**
   * Variance after log transformation.
   */
  σ2:Expression<Real> <- σ2;

  function valueForward() -> Real {
    assert !delay?;
    return simulate_log_gaussian(μ, σ2);
  }

  function observeForward(x:Real) -> Real {
    assert !delay?;
    return logpdf_log_gaussian(x, μ, σ2);
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
        delay <- DelayLinearNormalInverseGammaLogGaussian(future, futureUpdate, m1!.a, m1!.x, m1!.c);
      } else if (m2 <- μ.graftDotMultivariateNormalInverseGamma())? && m2!.x.σ2 == σ2.getDelay() {
        delay <- DelayDotMultivariateNormalInverseGammaLogGaussian(future, futureUpdate, m2!.a, m2!.x, m2!.c);
      } else if (m3 <- μ.graftNormalInverseGamma())? && m3!.σ2 == σ2.getDelay() {
        delay <- DelayNormalInverseGammaLogGaussian(future, futureUpdate, m3!);
      } else if (m4 <- μ.graftLinearGaussian())? {
        delay <- DelayLinearGaussianLogGaussian(future, futureUpdate, m4!.a, m4!.x, m4!.c, σ2);
      } else if (m5 <- μ.graftDotMultivariateGaussian())? {
        delay <- DelayDotMultivariateGaussianLogGaussian(future, futureUpdate, m5!.a, m5!.x, m5!.c, σ2);
      } else if (m6 <- μ.graftGaussian())? {
        delay <- DelayGaussianLogGaussian(future, futureUpdate, m6!, σ2);
      } else {
        /* trigger a sample of μ, and double check that this doesn't cause
         * a sample of σ2 before we try creating an inverse-gamma Gaussian */
        μ.value();
        if (s2 <- σ2.graftInverseGamma())? {
          delay <- DelayInverseGammaLogGaussian(future, futureUpdate, μ, s2!);
        } else if force {
          delay <- DelayLogGaussian(future, futureUpdate, μ, σ2);
        }
      }
    }
  }
}

/**
 * Create log-Gaussian distribution.
 */
function LogGaussian(μ:Expression<Real>, σ2:Expression<Real>) ->
    LogGaussian {
  m:LogGaussian(μ, σ2);
  return m;
}

/**
 * Create log-Gaussian distribution.
 */
function LogGaussian(μ:Expression<Real>, σ2:Real) -> LogGaussian {
  return LogGaussian(μ, Boxed(σ2));
}

/**
 * Create log-Gaussian distribution.
 */
function LogGaussian(μ:Real, σ2:Expression<Real>) -> LogGaussian {
  return LogGaussian(Boxed(μ), σ2);
}

/**
 * Create log-Gaussian distribution.
 */
function LogGaussian(μ:Real, σ2:Real) -> LogGaussian {
  return LogGaussian(Boxed(μ), Boxed(σ2));
}
