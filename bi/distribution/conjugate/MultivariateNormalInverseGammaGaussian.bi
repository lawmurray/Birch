/*
 * Multivariate Gaussian with conjugate normal inverse-gamma prior on mean
 * and variance.
 */
class MultivariateNormalInverseGammaGaussian < Random<Real[_]> {
  /**
   * Mean.
   */
  μ:MultivariateNormalInverseGamma;

  /**
   * Variance.
   */
  σ2:InverseGamma;

  function initialize(μ:MultivariateNormalInverseGamma, σ2:InverseGamma) {
    //assert μ.σ2 == σ2;
    super.initialize(μ);
    this.μ <- μ;
    this.σ2 <- σ2;
  }
  
  function doCondition() {
    D:Integer <- length(μ.μ);
    μ_1:Real[_];
    Σ_1:Real[_,_];
    Λ_1:Real[_,_];
    α_1:Real;
    β_1:Real;
    
    if (μ.isRealized() && !σ2.isRealized()) {
      (α_1, β_1) <- update_multivariate_inverse_gamma_gaussian(value(), μ.value(), σ2.α, σ2.β);
      σ2.update(α_1, β_1);
    } else if (!μ.isRealized() && σ2.isRealized()) {
      μ_1 <- μ.μ;
      Σ_1 <- inv(μ.Λ)*σ2.value();
      (μ_1, Σ_1) <- update_multivariate_gaussian_gaussian(value(), μ_1, Σ_1, μ_1, Σ_1 + diagonal(σ2.value(), D));
      μ.update(μ_1, inv(Σ_1));
    } else {
      (μ_1, Λ_1, α_1, β_1) <- update_multivariate_normal_inverse_gamma_gaussian(value(), μ.μ, μ.Λ, σ2.α, σ2.β);
      μ.update(μ_1, Λ_1);
      σ2.update(α_1, β_1);
    }
  }

  function doRealize() {
    if (μ.isRealized() && σ2.isRealized()) {
      if (isMissing()) {
        set(simulate_multivariate_gaussian(μ.value(), σ2.value()));
      } else {
        setWeight(observe_multivariate_gaussian(value(), μ.value(), σ2.value()));
      }
    } else if (μ.isRealized() && !σ2.isRealized()) {
      if (isMissing()) {
        set(simulate_multivariate_inverse_gamma_gaussian(μ.value(), σ2.α, σ2.β));
      } else {
        setWeight(observe_multivariate_inverse_gamma_gaussian(value(), μ.value(), σ2.α, σ2.β));
      }
    } else if (!μ.isRealized() && σ2.isRealized()) {
      D:Integer <- length(μ.μ);
      Σ_1:Real[_,_] <- (inv(μ.Λ) + identity(D))*σ2.value();
      if (isMissing()) {
        set(simulate_multivariate_gaussian(μ.μ, Σ_1));
      } else {
        setWeight(observe_multivariate_gaussian(value(), μ.μ, Σ_1));
      }
    } else {
      if (isMissing()) {
        set(simulate_multivariate_normal_inverse_gamma_gaussian(μ.μ, μ.Λ, σ2.α, σ2.β));
      } else {
        setWeight(observe_multivariate_normal_inverse_gamma_gaussian(value(), μ.μ, μ.Λ, σ2.α, σ2.β));
      }
    }
  }
}

/**
 * Create Gaussian distribution.
 */
function Gaussian(μ:MultivariateNormalInverseGamma, σ2:InverseGamma) ->
    MultivariateNormalInverseGammaGaussian {
  ///@todo Check μ.σ2 == σ2
  x:MultivariateNormalInverseGammaGaussian;
  x.initialize(μ, σ2);
  return x;
}

/**
 * Create Gaussian distribution.
 */
function Gaussian(μ:Random<Real[_]>, σ2:Random<Real>) -> Random<Real[_]> {
  μ_1:MultivariateNormalInverseGamma? <- MultivariateNormalInverseGamma?(μ);
  σ2_1:InverseGamma? <- InverseGamma?(σ2);
  if (μ_1? && σ2_1?) {
    return Gaussian(μ_1!, σ2_1!);
  } else if (μ_1?) {
    return Gaussian(μ_1!, diagonal(σ2.value(), length(μ_1!.μ)));
  } else if (σ2_1?) {
    return Gaussian(μ.value(), σ2_1!);
  } else {
    return Gaussian(μ.value(), diagonal(σ2.value(), length(μ.value())));
  }
}
