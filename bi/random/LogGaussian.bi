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

  /**
   * Marginal mean.
   */
  μ_m:Real;
  
  /**
   * Marginal variance
   */
  σ2_m:Real;
  
  /**
   * Updated mean.
   */
  μ_p:Real;
  
  /**
   * Updated variance.
   */
  σ2_p:Real;

  /**
   * Prior mean on `μ`;
   */
  μ_0:Real;

  /**
   * Prior variance on `σ2`;
   */
  σ2_0:Real;

  /**
   * Scaling of `μ` for affine transformation.
   */
  a:Real;

  /**
   * Translation of `μ` for affine transformation.
   */
  c:Real;

  function graft() {
    if (μ.isGaussian() || μ.isAffineGaussian()) {
      return μ;
    } else {
      return nil;
    }
  }

  function doMarginalize() {
    if (μ.isGaussian()) {
      (μ_0, σ2_0) <- μ.getGaussian();
      μ_m <- μ_0;
      σ2_m <- σ2_0 + σ2.value();
    } else if (μ.isAffineGaussian()) {
      (a, μ_0, σ2_0, c) <- μ.getAffineGaussian();
      μ_m <- a*μ_0 + c;
      σ2_m <- a*a*σ2_0 + σ2.value();
    } else {
      μ_m <- μ.value();
      σ2_m <- σ2.value();
    }
    μ_p <- μ_m;
    σ2_p <- σ2_m;
  }

  function doSimulate() -> Real {
    return simulate_log_gaussian(μ_p, σ2_p);
  }
  
  function doObserve(x:Real) -> Real {
    return observe_log_gaussian(x, μ_p, σ2_p);
  }

  function doCondition(x:Real) {
    if (μ.isGaussian()) {
      μ.setGaussian(update_gaussian_gaussian(log(x), μ_0, σ2_0, μ_m, σ2_m));
    } else if (μ.isAffineGaussian()) {
      μ.setAffineGaussian(update_affine_gaussian_gaussian(log(x), a, μ_0, σ2_0, μ_m, σ2_m));
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
  return LogGaussian(μ, Literal(σ2));
}

/**
 * Create log-Gaussian distribution.
 */
function LogGaussian(μ:Real, σ2:Expression<Real>) -> LogGaussian {
  return LogGaussian(Literal(μ), σ2);
}

/**
 * Create log-Gaussian distribution.
 */
function LogGaussian(μ:Real, σ2:Real) -> LogGaussian {
  return LogGaussian(Literal(μ), Literal(σ2));
}
