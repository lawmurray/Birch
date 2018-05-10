/**
 * Log-Gaussian distribution.
 */
class LogGaussian(μ:Expression<Real>, σ2:Expression<Real>) < Distribution<Real> {
  /**
   * Mean after log transformation.
   */
  μ:Expression<Real> <- μ;
  
  /**
   * Variance after log transformation.
   */
  σ2:Expression<Real> <- σ2;

  function graft() {
    if delay? {
      delay!.prune();
    } else {
      s2:DelayInverseGamma?;
      m1:TransformLinearNormalInverseGamma?;
      m2:DelayNormalInverseGamma?;
      m3:TransformLinearGaussian?;
      m4:DelayGaussian?;
      
      if (s2 <- σ2.graftInverseGamma())? {
        if (m1 <- μ.graftLinearNormalInverseGamma(σ2))? {
          delay <- DelayLinearNormalInverseGammaLogGaussian(x, m1!.a, m1!.x,
              m1!.c);
        } else if (m2 <- μ.graftNormalInverseGamma(σ2))? {
          delay <- DelayNormalInverseGammaLogGaussian(x, m2!);
        } else {
          delay <- DelayInverseGammaLogGaussian(x, μ, s2!);
        }
      } else if (m3 <- μ.graftLinearGaussian())? {
        delay <- DelayLinearGaussianLogGaussian(x, m3!.a, m3!.x, m3!.c, σ2);
      } else if (m4 <- μ.graftGaussian())? {
        delay <- DelayGaussianLogGaussian(x, m4!, σ2);
      } else {
        delay <- DelayLogGaussian(x, μ, σ2);
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
