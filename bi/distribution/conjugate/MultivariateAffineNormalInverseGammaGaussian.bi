/*
 * Multivariate Gaussian with multivariate normal inverse-gamma prior on mean
 * and covariance, where the normal on the mean is further modified with an
 * affine transformation.
 */
class MultivariateAffineNormalInverseGammaGaussian < Random<Real[_]> {
  /**
   * Scale.
   */
  A:Real[_,_];

  /**
   * Random variable.
   */
  x:MultivariateNormalInverseGamma;
  
  /**
   * Offset.
   */
  c:Real[_];

  /**
   * Variance.
   */
  σ2:InverseGamma;

  function initialize(A:Real[_,_], x:MultivariateNormalInverseGamma,
      c:Real[_], σ2:InverseGamma) {
    //assert x.σ2 == σ2;
    super.initialize(x);
    this.A <- A;
    this.x <- x;
    this.c <- c;
    this.σ2 <- σ2;
  }
  
  function doCondition() {
    D:Integer <- length(x.μ);
    μ_1:Real[_];
    Σ_1:Real[_,_];
    α_1:Real;
    β_1:Real;
    
    if (x.isRealized() && !σ2.isRealized()) {
      (α_1, β_1) <- update_multivariate_inverse_gamma_gaussian(value(), A*x.value() + c, σ2.α, σ2.β);
      σ2.update(α_1, β_1);
    } else if (!x.isRealized() && σ2.isRealized()) {
      μ_1 <- x.μ;
      Σ_1 <- x.Σ*σ2.value();
      (μ_1, Σ_1) <- update_multivariate_affine_gaussian_gaussian(value(), A, μ_1, Σ_1, A*μ_1 + c, A*Σ_1*trans(A) + diagonal(σ2.value(), D));
      x.update(μ_1, Σ_1);
    } else {
      (μ_1, Σ_1, α_1, β_1) <- update_multivariate_affine_normal_inverse_gamma_gaussian(A, value(), c, x.μ, x.Σ, σ2.α, σ2.β);
      x.update(μ_1, Σ_1);
      σ2.update(α_1, β_1);
    }
  }

  function doRealize() {
    D:Integer <- length(x.μ);
    μ_1:Real[_];
    Σ_1:Real[_,_];
    σ2_1:Real;
    
    if (x.isRealized() && σ2.isRealized()) {
      μ_1 <- A*x.value() + c;
      σ2_1 <- σ2.value();
      if (isMissing()) {
        set(simulate_multivariate_gaussian(μ_1, σ2_1));
      } else {
        setWeight(observe_multivariate_gaussian(value(), μ_1, σ2));
      }
    } else if (x.isRealized() && !σ2.isRealized()) {
      μ_1 <- A*x.value() + c;
      if (isMissing()) {
        set(simulate_multivariate_inverse_gamma_gaussian(μ_1, σ2.α, σ2.β));
      } else {
        setWeight(observe_multivariate_inverse_gamma_gaussian(value(), μ_1, σ2.α, σ2.β));
      }
    } else if (!x.isRealized() && σ2.isRealized()) {
      μ_1 <- A*x.μ + c;
      Σ_1 <- (A*x.Σ*trans(A) + identity(D))*σ2.value();
      if (isMissing()) {
        set(simulate_multivariate_gaussian(μ_1, Σ_1));
      } else {
        setWeight(observe_multivariate_gaussian(value(), μ_1, Σ_1));
      }
    } else {
      if (isMissing()) {
        set(simulate_multivariate_affine_normal_inverse_gamma_gaussian(A, x.μ, c, x.Σ, σ2.α, σ2.β));
      } else {
        setWeight(observe_multivariate_affine_normal_inverse_gamma_gaussian(value(), A, x.μ, c, x.Σ, σ2.α, σ2.β));
      }
    }
  }
}

/**
 * Create Gaussian distribution.
 */
function Gaussian(μ:MultivariateAffineExpression, σ2:InverseGamma) -> Random<Real[_]> {
  x:MultivariateNormalInverseGamma? <- MultivariateNormalInverseGamma?(μ.x);
  if (x?) {  // and σ2 match
    y:MultivariateAffineNormalInverseGammaGaussian;
    y.initialize(μ.A, x!, μ.c, σ2);
    return y;
  } else {
    return Gaussian(μ.value(), σ2);
  }
}

/**
 * Create Gaussian distribution.
 */
function Gaussian(μ:MultivariateAffineExpression, σ2:Random<Real>) -> Random<Real[_]> {
  σ2_1:InverseGamma? <- InverseGamma?(σ2);
  if (σ2_1?) {
    return Gaussian(μ, σ2_1!);
  } else {
    return Gaussian(μ, diagonal(σ2.value(), length(μ.c)));
  }
}
