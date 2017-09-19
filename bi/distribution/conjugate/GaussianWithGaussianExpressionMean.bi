/**
 * Gaussian with conjugate prior on mean.
 */
class GaussianWithGaussianExpressionMean < Gaussian {
  /**
   * Mean.
   */
  μ:GaussianExpression;

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

  function initialize(μ:GaussianExpression, σ2:Real) {
    super.initialize(μ.x);
    this.μ <- μ;
    this.σ2 <- σ2;
  }
  
  function doMarginalize() {
    μ_0 <- μ.a*μ.x.μ + μ.c;
    σ2_0 <- μ.a*μ.a*μ.x.σ2 + σ2;
    update(μ_0, σ2_0);
  }

  function doForward() {
    μ_0 <- μ.a*μ.x.μ + μ.c;
    σ2_0 <- σ2;
    update(μ_0, σ2_0);
  }
  
  function doCondition() {
    k:Real <- μ.x.σ2*μ.a/σ2_0;
    μ.x.update(μ.x.μ + k*(x - μ_0), μ.x.σ2 - k*μ.a*μ.x.σ2);
  }
}

function Gaussian(μ:GaussianExpression, σ2:Real) -> Gaussian {
  x:GaussianWithGaussianExpressionMean;
  x.initialize(μ, σ2);
  return x;
}
