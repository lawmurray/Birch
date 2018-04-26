/**
 * Log-Gaussian distribution.
 */
class LogGaussian(μ:Expression<Real>, σ2:Expression<Real>) < Random<Real> {
  /**
   * Mean after log transformation.
   */
  μ:Expression<Real> <- μ;
  
  /**
   * Variance after log transformation.
   */
  σ2:Expression<Real> <- σ2;

  function graft() {
    if (μ.isNormalInverseGamma(σ2)) {
      m:DelayNormalInverseGammaLogGaussian(this, μ.getNormalInverseGamma(σ2));
      m.graft();
      delay <- m;
    } else if (μ.isAffineNormalInverseGamma(σ2)) {
      a:Real;
      μ_0:DelayNormalInverseGamma?;
      c:Real;
      (a, μ_0, c) <- μ.getAffineNormalInverseGamma(σ2);
      m:DelayAffineNormalInverseGammaLogGaussian(this, a, μ_0!, c);
      m.graft();
      delay <- m;
    } else if (σ2.isInverseGamma()) {
      m:DelayInverseGammaLogGaussian(this, μ.value(), σ2.getInverseGamma());
      m.graft();
      delay <- m;
    } else if (μ.isGaussian()) {
      m:DelayGaussianLogGaussian(this, μ.getGaussian(), σ2.value());
      m.graft();
      delay <- m;
    } else if (μ.isAffineGaussian()) {
      a:Real;
      μ_0:DelayGaussian?;
      c:Real;
      (a, μ_0, c) <- μ.getAffineGaussian();
      m:DelayAffineGaussianLogGaussian(this, a, μ_0!, c, σ2.value());
      m.graft();
      delay <- m;
    } else {
      m:DelayLogGaussian(this, μ.value(), σ2.value());
      m.graft();
      delay <- m;
    }
  }
}

/**
 * Create log-Gaussian distribution.
 */
function LogGaussian(μ:Expression<Real>, σ2:Expression<Real>) -> LogGaussian {
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
