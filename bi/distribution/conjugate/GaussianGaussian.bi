/*
 * Gaussian with another Gaussian as its mean.
 */
class GaussianGaussian < Gaussian {
  /**
   * Mean.
   */
  μ:Gaussian;

  /**
   * Variance.
   */
  σ2:Real;
  
  /**
   * Marginal mean.
   */
  μ_m:Real;
  
  /**
   * Marginal variance.
   */
  σ2_m:Real;

  function initialize(μ:Gaussian, σ2:Real) {
    super.initialize(μ);
    this.μ <- μ;
    this.σ2 <- σ2;
  }
  
  function doMarginalize() {
    μ_m <- μ.μ;
    σ2_m <- μ.σ2 + σ2;
    update(μ_m, σ2_m);
  }

  function doForward() {
    μ_m <- μ.value();
    σ2_m <- σ2;
    update(μ_m, σ2_m);
  }
  
  function doCondition() {
    μ_1:Real;
    σ2_1:Real;
    (μ_1, σ2_1) <- update_gaussian_gaussian(value(), μ.μ, μ.σ2, μ_m, σ2_m);
    μ.update(μ_1, σ2_1);
  }
}

/**
 * Create Gaussian distribution.
 */
function Gaussian(μ:Gaussian, σ2:Real) -> Gaussian {
  x:GaussianGaussian;
  x.initialize(μ, σ2);
  return x;
}

/**
 * Create Gaussian distribution.
 */
function Gaussian(μ:Random<Real>, σ2:Real) -> Gaussian {
  μ1:Gaussian? <- Gaussian?(μ);
  if (μ1?) {
    return Gaussian(μ1!, σ2);
  } else {
    return Gaussian(μ.value(), σ2);
  }
}

/**
 * Create Gaussian distribution.
 */
function Normal(μ:Gaussian, σ2:Real) -> Gaussian {
  return Gaussian(μ, σ2);
}

/**
 * Create Gaussian distribution.
 */
function Normal(μ:Random<Real>, σ2:Real) -> Gaussian {
  return Gaussian(μ, σ2);
}
