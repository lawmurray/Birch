/**
 * Gaussian with normal inverse-gamma prior on mean and variance,
 * where the normal on the mean is further modified with an affine
 * transformation.
 */
class AffineNormalInverseGammaGaussian < Random<Real> {
  /**
   * Scale.
   */
  a:Real;

  /**
   * Random variable.
   */
  x:NormalInverseGamma;
  
  /**
   * Offset.
   */
  c:Real;

  /**
   * Variance.
   */
  σ2:InverseGamma;

  function initialize(a:Real, x:NormalInverseGamma, c:Real, σ2:InverseGamma) {
    //assert x.σ2 == σ2;
    super.initialize(x);
    this.a <- a;
    this.x <- x;
    this.c <- c;
    this.σ2 <- σ2;
  }
  
  function doCondition() {
    y:Real <- value() - c;

    μ_0:Real <- x.μ;
    λ_0:Real <- 1.0/x.a2;
    α_0:Real <- σ2.α;
    β_0:Real <- σ2.β;

    μ_1:Real <- (λ_0*μ_0 + a*y)/(λ_0 + a*a);
    λ_1:Real <- λ_0 + a*a;
    α_1:Real <- α_0 + 0.5;
    β_1:Real <- β_0 + 0.5*(y*y + μ_0*μ_0*λ_0 - μ_1*μ_1*λ_1);
    
    x.update(μ_1, 1.0/λ_1);
    σ2.update(α_1, β_1);
  }

  function doRealize() {
    if (x.isRealized() && σ2.isRealized()) {
      μ_1:Real <- a*x.value() + c;
      σ2_1:Real <- σ2.value();
      if (isMissing()) {
        set(simulate_gaussian(μ_1, σ2_1));
      } else {
        setWeight(observe_gaussian(value(), μ_1, σ2_1));
      }
    } else if (x.isRealized() && !σ2.isRealized()) {
      /* just like InverseGammaGaussian */
      μ_1:Real <- a*x.value() + c;
      ν:Real <- 2.0*σ2.α;
      s2:Real <- σ2.β/σ2.α;
      if (isMissing()) {
        set(simulate_student_t(ν, μ_1, s2));
      } else {
        setWeight(observe_student_t(value(), ν, μ_1, s2));
      }
    } else if (!x.isRealized() && σ2.isRealized()) {
      /* just like GaussianGaussian */
      μ_1:Real <- a*x.μ + c;
      σ2_1:Real <- (a*a*x.a2 + 1.0)*σ2.value();
      if (isMissing()) {
        set(simulate_gaussian(μ_1, σ2_1));
      } else {
        setWeight(observe_gaussian(value(), μ_1, σ2_1));
      }
    } else {
      ν:Real <- 2.0*σ2.α;
      μ_1:Real <- a*x.μ + c;
      σ2_1:Real <- σ2.β*(1.0 + a*a*x.a2)/σ2.α;
      if (isMissing()) {
        set(simulate_student_t(ν, μ_1, σ2_1));
      } else {
        setWeight(observe_student_t(value(), ν, μ_1, σ2_1));
      }
    }
  }
}

/**
 * Create Gaussian distribution.
 */
function Gaussian(μ:AffineExpression, σ2:InverseGamma) -> Random<Real> {
  x:NormalInverseGamma? <- NormalInverseGamma?(μ.x);
  if (x?) {  // and σ2 match
    y:AffineNormalInverseGammaGaussian;
    y.initialize(μ.a, x!, μ.c, σ2);
    return y;
  } else {
    return Gaussian(μ.value(), σ2);
  }
}

/**
 * Create Gaussian distribution.
 */
function Gaussian(μ:AffineExpression, σ2:Random<Real>) -> Random<Real> {
  σ2_1:InverseGamma? <- InverseGamma?(σ2);
  if (σ2_1?) {
    return Gaussian(μ, σ2_1!);
  } else {
    return Gaussian(μ, σ2.value());
  }
}
