/*
 * Gaussian with conjugate normal inverse-gamma prior on mean and
 * variance.
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
    μ_1:Real;
    a2_1:Real;
    α_1:Real;
    β_1:Real;
    
    if (μ.isRealized() && !σ2.isRealized()) {
      (α_1, β_1) <- update_inverse_gamma_gaussian(value(), μ.value(), σ2.α, σ2.β);
      σ2.update(α_1, β_1);
    } else if (!μ.isRealized() && σ2.isRealized()) {
      μ_1 <- μ.μ;
      a2_1 <- μ.a2*σ2.value();
      (μ_1, a2_1) <- update_gaussian_gaussian(value(), μ_1, a2_1, μ_1, a2_1 + σ2.value());
      μ.update(μ_1, a2_1);
    } else {
      (μ_1, a2_1, α_1, β_1) <- update_normal_inverse_gamma_gaussian(value(), μ.μ, μ.a2, σ2.α, σ2.β);
      μ.update(μ_1, a2_1);
      σ2.update(α_1, β_1);
    }
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
      if (isMissing()) {
        set(simulate_inverse_gamma_gaussian(μ.value(), σ2.α, σ2.β));
      } else {
        setWeight(observe_inverse_gamma_gaussian(value(), μ.value(),
            σ2.α, σ2.β));
      }
    } else if (!μ.isRealized() && σ2.isRealized()) {
      μ_1:Real <- μ.μ;
      σ2_1:Real <- (μ.a2 + 1.0)*σ2.value();
      if (isMissing()) {
        set(simulate_gaussian(μ_1, σ2_1));
      } else {
        setWeight(observe_gaussian(value(), μ_1, σ2_1));
      }
    } else {
      if (isMissing()) {
        set(simulate_normal_inverse_gamma_gaussian(μ.μ, μ.a2, σ2.α,
            σ2.β));
      } else {
        setWeight(observe_normal_inverse_gamma_gaussian(value(), μ.μ,
            μ.a2, σ2.α, σ2.β));
      }
    }
  }
}

/**
 * Create Gaussian distribution.
 */
function Gaussian(μ:NormalInverseGamma, σ2:InverseGamma) ->
    NormalInverseGammaGaussian {
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
