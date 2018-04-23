/**
 * Gaussian distribution.
 */
class Gaussian(μ:Expression<Real>, σ2:Expression<Real>) < Random<Real> {
  /**
   * Mean.
   */
  μ:Expression<Real> <- μ;
  
  /**
   * Variance.
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

  function isGaussian() -> Boolean {
    return isMissing();
  }

  function getGaussian() -> (Real, Real) {
    return (μ_p, σ2_p);
  }

  function setGaussian(θ:(Real, Real)) {
    (μ_p, σ2_p) <- θ;
  }

  function doParent() -> Delay? {
    if (μ.isGaussian()) {
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
    return simulate_gaussian(μ_p, σ2_p);
  }
  
  function doObserve(x:Real) -> Real {
    return observe_gaussian(x, μ_p, σ2_p);
  }

  function doCondition(x:Real) {
    if (μ.isGaussian()) {
      μ.setGaussian(update_gaussian_gaussian(x, μ_0, σ2_0, μ_m, σ2_m));
    } else if (μ.isAffineGaussian()) {
      μ.setAffineGaussian(update_affine_gaussian_gaussian(x, a, μ_0, σ2_0, μ_m, σ2_m));
    }
  }
}

/**
 * Create Gaussian distribution.
 */
function Gaussian(μ:Expression<Real>, σ2:Expression<Real>) -> Gaussian {
  m:Gaussian(μ, σ2);
  m.initialize();
  return m;
}

/**
 * Create Gaussian distribution.
 */
function Gaussian(μ:Expression<Real>, σ2:Real) -> Gaussian {
  return Gaussian(μ, Literal(σ2));
}

/**
 * Create Gaussian distribution.
 */
function Gaussian(μ:Real, σ2:Expression<Real>) -> Gaussian {
  return Gaussian(Literal(μ), σ2);
}

/**
 * Create Gaussian distribution.
 */
function Gaussian(μ:Real, σ2:Real) -> Gaussian {
  return Gaussian(Literal(μ), Literal(σ2));
}
