/**
 * Gaussian distribution.
 *
 * ### Unknown mean and variance
 *
 * Normal inverse-gamma distribution. This represents the joint
 * distribution:
 *
 * $$x \mid \sigma^2 \sim \mathrm{N}(\mu, a^2 \sigma^2),$$
 * $$\sigma^2 \sim \Gamma^{-1}(\alpha, \beta).$$
 *
 * which may be denoted:
 *
 * $$(x, \sigma^2) \sim \mathrm{N-}\Gamma^{-1}(\mu, a^2, \alpha, \beta),$$
 *
 * and is the conjugate prior of a Gaussian distribution with both
 * unknown mean and unknown variance.
 *
 * It is established via code such as the following:
 *
 *     σ2 ~ InverseGamma(α, β);
 *     x ~ Gaussian(μ, a2*σ2);
 *     y ~ Gaussian(x, σ2);
 *
 * where the last argument in the distribution of `y` appears in the
 * last argument of the distribution of `x`. The operation of `a2` on
 * `σ2` may be multiplication on left (as above) or right, or division
 * on right.
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

  function isAffineGaussian() -> Boolean {
    return isMissing();
  }

  function getAffineGaussian() -> (Real, Real, Real, Real) {
    return (1.0, μ_p, σ2_p, 0.0);
  }

  function setAffineGaussian(θ:(Real, Real)) {
    (μ_p, σ2_p) <- θ;
  }

  function isNormalInverseGamma(σ2:Expression<Real>) -> Boolean {
    return σ2.isScaledInverseGamma(σ2);
  }
  
  function getNormalInverseGamma(σ2:Expression<Real>) -> (Real, Real, Real, Real) {
    a2:Real;
    α:Real;
    β:Real;
    (a2, α, β) <- σ2.getScaledInverseGamma(σ2);
    // ^ a2 is ignored, initializes σ2_p but may have since been updated
    return (μ_p, σ2_p, α, β);
  }

  function setNormalInverseGamma(σ2:Expression<Real>, θ:(Real, Real, Real, Real)) {
    α:Real;
    β:Real;
    (μ_p, σ2_p, α, β) <- θ;
    σ2.setScaledInverseGamma(σ2, (α, β));
  }

  function isAffineNormalInverseGamma(σ2:Expression<Real>) -> Boolean {
    return σ2.isScaledInverseGamma(σ2);
  }
  
  function getAffineNormalInverseGamma(σ2:Expression<Real>) -> (Real, Real, Real, Real, Real, Real) {
    a2:Real;
    α:Real;
    β:Real;
    (a2, α, β) <- σ2.getScaledInverseGamma(σ2);
    // ^ a2 is ignored, initializes σ2_p but may have since been updated
    return (1.0, μ_p, σ2_p, 0.0, α, β);
  }

  function setAffineNormalInverseGamma(σ2:Expression<Real>, θ:(Real, Real, Real, Real)) {
    α:Real;
    β:Real;
    (μ_p, σ2_p, α, β) <- θ;
    σ2.setScaledInverseGamma(σ2, (α, β));
  }
  
  function doParent() -> Delay? {
    if (μ.isNormalInverseGamma(σ2) || μ.isAffineNormalInverseGamma(σ2)) {
      return μ;
    } else if (σ2.isInverseGamma()) {
      return σ2;
    } else if (μ.isGaussian() || μ.isAffineGaussian()) {
      return μ;
    } else {
      return nil;
    }
  }

  function doMarginalize() {
    α:Real;
    β:Real;
    if (μ.isNormalInverseGamma(σ2)) {
      (μ_m, σ2_m, α, β) <- μ.getNormalInverseGamma(σ2);
    } else if (μ.isAffineNormalInverseGamma(σ2)) {
      (a, μ_m, c, σ2_m, α, β) <- μ.getAffineNormalInverseGamma(σ2);
    } else if (σ2.isInverseGamma()) {
      μ_m <- μ.value();
      σ2_m <- 1.0;
    } else if (μ.isGaussian()) {
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
    μ1:Real;
    a2:Real;
    α:Real;
    β:Real;
    
    if (μ.isNormalInverseGamma(σ2)) {
      (μ1, a2, α, β) <- μ.getNormalInverseGamma(σ2);
      return simulate_normal_inverse_gamma_gaussian(μ1, a2, α, β);
    } else if (μ.isAffineNormalInverseGamma(σ2)) {
      (a, μ1, c, a2, α, β) <- μ.getAffineNormalInverseGamma(σ2);
      return simulate_affine_normal_inverse_gamma_gaussian(a, μ1, c, a2, α, β);
    } else if (σ2.isInverseGamma()) {
      (α, β) <- σ2.getInverseGamma();
      return simulate_inverse_gamma_gaussian(μ.value(), α, β);
    } else {
      return simulate_gaussian(μ_p, σ2_p);
    }
  }
  
  function doObserve(x:Real) -> Real {
    μ1:Real;
    a2:Real;
    α:Real;
    β:Real;
    
    if (μ.isNormalInverseGamma(σ2)) {
      (μ1, a2, α, β) <- μ.getNormalInverseGamma(σ2);
      return observe_normal_inverse_gamma_gaussian(x, μ1, a2, α, β);
    } else if (μ.isAffineNormalInverseGamma(σ2)) {
      (a, μ1, c, a2, α, β) <- μ.getAffineNormalInverseGamma(σ2);
      return observe_affine_normal_inverse_gamma_gaussian(x, a, μ1, c, a2, α, β);
    } else if (σ2.isInverseGamma()) {
      (α, β) <- σ2.getInverseGamma();
      return observe_inverse_gamma_gaussian(x, μ.value(), α, β);
    } else {
      return observe_gaussian(x, μ_p, σ2_p);
    }
  }

  function doCondition(x:Real) {
    μ1:Real;
    a2:Real;
    α:Real;
    β:Real;
    
    if (μ.isNormalInverseGamma(σ2)) {
      (μ1, a2, α, β) <- μ.getNormalInverseGamma(σ2);
      μ.setNormalInverseGamma(σ2, update_normal_inverse_gamma_gaussian(x, μ1, a2, α, β));
    } else if (μ.isAffineNormalInverseGamma(σ2)) {
      (a, μ1, c, a2, α, β) <- μ.getAffineNormalInverseGamma(σ2);
      μ.setAffineNormalInverseGamma(σ2, update_affine_normal_inverse_gamma_gaussian(x, a, μ1, c, a2, α, β));
    } else if (σ2.isInverseGamma()) {
      (α, β) <- σ2.getInverseGamma();
      σ2.setInverseGamma(update_inverse_gamma_gaussian(x, μ.value(), α, β));
    } else if (μ.isGaussian()) {
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
