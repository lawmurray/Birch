/**
 * Gaussian with normal inverse-gamma prior on mean and variance.
 */
class NormalInverseGammaGaussian < Random<Real> {
  /**
   * Mean.
   */
  μ:NormalInverseGamma;

  /**
   * Variance.
   */
  σ2:InverseGamma;

  function initialize(μ:NormalInverseGamma, σ2:InverseGamma) {
    //assert μ.σ2 == σ2;
    super.initialize(μ);
    this.μ <- μ;
    this.σ2 <- σ2;
  }
  
  function doCondition() {
    μ_0:Real <- μ.μ;
    λ_0:Real <- 1.0/μ.a2;
    α_0:Real <- σ2.α;
    β_0:Real <- σ2.β;

    μ_1:Real <- (λ_0*μ_0 + value())/(λ_0 + 1.0);
    λ_1:Real <- λ_0 + 1.0;
    α_1:Real <- α_0 + 0.5;
    β_1:Real <- β_0 + 0.5*(λ_0/λ_1)*pow(value() - μ_0, 2.0);
    
    μ.update(μ_1, 1.0/λ_1);
    σ2.update(α_1, β_1);
  }

  function doRealize() {
    if (μ.isRealized() && σ2.isRealized()) {
      /* just like Gaussian */
      if (isMissing()) {
        set(simulate_gaussian(μ.value(), σ2.value()));
      } else {
        setWeight(observe_gaussian(value(), μ.value(), σ2.value()));
      }
    } else if (μ.isRealized() && !σ2.isRealized()) {
      /* just like InverseGammaGaussian */
      ν:Real <- 2.0*σ2.α;    // degrees of freedom
      s2:Real <- σ2.β/σ2.α;  // squared scale
      if (isMissing()) {
        set(simulate_student_t(ν, μ.value(), s2));
      } else {
        setWeight(observe_student_t(value(), ν, μ.value(), s2));
      }
    } else if (!μ.isRealized() && σ2.isRealized()) {
      /* just like GaussianGaussian */
      s2:Real <- (μ.a2 + 1.0)*σ2.value();
      if (isMissing()) {
        set(simulate_gaussian(μ.μ, s2));
      } else {
        setWeight(observe_gaussian(value(), μ.μ, s2));
      }
    } else {
      ν:Real <- 2.0*σ2.α;         // degrees of freedom
      s2:Real <- σ2.β*(1.0 + μ.a2)/σ2.α;  // squared scale
      if (isMissing()) {
        set(simulate_student_t(ν, μ.μ, s2));
      } else {
        setWeight(observe_student_t(value(), ν, μ.μ, s2));
      }
    }
  }
}

/**
 * Create Gaussian distribution.
 */
function Gaussian(μ:NormalInverseGamma, σ2:InverseGamma) -> NormalInverseGammaGaussian {
  ///@todo Check μ.σ2 == σ2
  x:NormalInverseGammaGaussian;
  x.initialize(μ, σ2);
  return x;
}

/**
 * Create Gaussian distribution.
 */
function Gaussian(μ:Random<Real>, σ2:Random<Real>) -> Random<Real> {
  μ_1:NormalInverseGamma? <- NormalInverseGamma?(μ);
  σ2_1:InverseGamma? <- InverseGamma?(σ2);
  if (μ_1? && σ2_1?) {
    return Gaussian(μ_1!, σ2_1!);
  } else if (μ_1?) {
    return Gaussian(μ_1!, σ2.value());
  } else if (σ2_1?) {
    return Gaussian(μ.value(), σ2_1!);
  } else {
    return Gaussian(μ.value(), σ2.value());
  }
}
