/**
 * Gaussian with affine transformation of another Gaussian as its mean.
 */
class GaussianWithAffineGaussianMean < Gaussian {
  /**
   * Scale.
   */
  a:Real;
  
  /**
   * Random variable.
   */
  x:Gaussian;
  
  /**
   * Offset.
   */
  c:Real;

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

  function initialize(a:Real, x:Gaussian, c:Real, σ2:Real) {
    super.initialize(x);
    this.a <- a;
    this.x <- x;
    this.c <- c;
    this.σ2 <- σ2;
  }
  
  function doMarginalize() {
    μ_0 <- a*x.μ + c;
    σ2_0 <- a*a*x.σ2 + σ2;
    update(μ_0, σ2_0);
  }

  function doForward() {
    μ_0 <- a*x.value() + c;
    σ2_0 <- σ2;
    update(μ_0, σ2_0);
  }
  
  function doCondition() {
    k:Real <- x.σ2*a/σ2_0;
    x.update(x.μ + k*(value() - μ_0), x.σ2 - k*a*x.σ2);
  }
}

/**
 * Create Gaussian distribution.
 */
function Gaussian(μ:AffineExpression, σ2:Real) -> Gaussian {
  x:Gaussian? <- Gaussian?(μ.x);
  if (x?) {
    y:GaussianWithAffineGaussianMean;
    y.initialize(μ.a, x!, μ.c, σ2);
    return y;
  } else {
    return Gaussian(μ.value(), σ2);
  }
}

/**
 * Create Gaussian distribution.
 */
function Normal(μ:AffineExpression, σ2:Real) -> Gaussian {
  return Gaussian(μ, σ2);
}
