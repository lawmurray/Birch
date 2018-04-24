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

  function isGaussian() -> Boolean {
    return isMissing();
  }

  function getGaussian() -> DelayGaussian {
    assert isGaussian();
    return DelayGaussian?(delay)!;
  }

  function isAffineGaussian() -> Boolean {
    return isGaussian();
  }

  function getAffineGaussian() -> (Real, DelayGaussian, Real) {
    return (1.0, getGaussian(), 0.0);
  }

  function isNormalInverseGamma(σ2:Expression<Real>) -> Boolean {
    return σ2.isScaledInverseGamma(σ2);
  }
  
  function getNormalInverseGamma(σ2:Expression<Real>) -> DelayNormalInverseGamma {
    return DelayNormalInverseGamma?(delay)!;
  }

  function isAffineNormalInverseGamma(σ2:Expression<Real>) -> Boolean {
    return isNormalInverseGamma(σ2);
  }
  
  function getAffineNormalInverseGamma(σ2:Expression<Real>) -> (Real, DelayNormalInverseGamma, Real) {
    return (1.0, getNormalInverseGamma(σ2), 0.0);
  }
  
  function graft() {
    if (μ.isNormalInverseGamma(σ2)) {
      m:DelayNormalInverseGammaGaussian(this, μ.getNormalInverseGamma(σ2));
      m.graft();
      delay <- m;
    } else if (μ.isAffineNormalInverseGamma(σ2)) {
      a:Real;
      μ_0:DelayNormalInverseGamma?;
      c:Real;
      (a, μ_0, c) <- μ.getAffineNormalInverseGamma(σ2);
      m:DelayAffineNormalInverseGammaGaussian(this, a, μ_0!, c);
      m.graft();
      delay <- m;
    } else if (σ2.isInverseGamma()) {
      m:DelayInverseGammaGaussian(this, μ.value(), σ2.getInverseGamma());
      m.graft();
      delay <- m;
    } else if (μ.isGaussian()) {
      m:DelayGaussianGaussian(this, μ.getGaussian(), σ2.value());
      m.graft();
      delay <- m;
    } else if (μ.isAffineGaussian()) {
      a:Real;
      μ_0:DelayGaussian?;
      c:Real;
      (a, μ_0, c) <- μ.getAffineGaussian();
      m:DelayAffineGaussianGaussian(this, a, μ_0!, c, σ2.value());
      m.graft();
      delay <- m;
    } else {
      m:DelayGaussian(this, μ.value(), σ2.value());
      m.graft();
      delay <- m;
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
