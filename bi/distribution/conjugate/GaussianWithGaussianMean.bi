/**
 * Gaussian with conjugate prior on mean.
 */
class GaussianWithGaussianMean < Gaussian {
  /**
   * Mean.
   */
  μ:Gaussian;

  /**
   * Variance.
   */
  σ2:Real;
  
  /**
   * Prior (marginalized) mean.
   */
  μ_0:Real;
  
  /**
   * Prior (marginalized) variance.
   */
  σ2_0:Real;

  function initialize(μ:Gaussian, σ2:Real) {
    super.initialize(μ);
    this.μ <- μ;
    this.σ2 <- σ2;
  }
  
  function doMarginalize() {
    μ_0 <- μ.μ;
    σ2_0 <- μ.σ2 + σ2;
    update(μ_0, σ2_0);
  }

  function doForward() {
    μ_0 <- μ.x;
    σ2_0 <- σ2;
    update(μ_0, σ2_0);
  }
  
  function doCondition() {
    k:Real <- μ.σ2/σ2_0;
    μ.update(μ.μ + k*(x - μ_0), μ.σ2 - k*μ.σ2);
  }
}

function Gaussian(μ:Gaussian, σ2:Real) -> Gaussian {
  x:GaussianWithGaussianMean;
  x.initialize(μ, σ2);
  return x;
}
