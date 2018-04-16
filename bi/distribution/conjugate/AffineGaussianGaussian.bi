/*
 * Gaussian with affine transformation of another Gaussian as its mean.
 */
class AffineGaussianGaussian < Gaussian {
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
   * Marginal mean.
   */
  μ_m:Real;
  
  /**
   * Marginal variance.
   */
  σ2_m:Real;

  function initialize(a:Real, x:Gaussian, c:Real, σ2:Real) {
    super.initialize(x);
    this.a <- a;
    this.x <- x;
    this.c <- c;
    this.σ2 <- σ2;
  }
  
  function doMarginalize() {
    if (x.isRealized()) {
      μ_m <- a*x.value() + c;
      σ2_m <- σ2;
    } else {
      μ_m <- a*x.μ + c;
      σ2_m <- a*a*x.σ2 + σ2;
    }
    update(μ_m, σ2_m);
  }
  
  function doCondition() {
    μ_1:Real;
    σ2_1:Real;
    (μ_1, σ2_1) <- update_affine_gaussian_gaussian(value(), a, x.μ, x.σ2, μ_m, σ2_m);
    x.update(μ_1, σ2_1);
  }
}

/**
 * Create Gaussian distribution.
 */
function Gaussian(μ:AffineExpression, σ2:Real) -> Gaussian {
  x:Gaussian? <- Gaussian?(μ.x);
  if (x?) {
    y:AffineGaussianGaussian;
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
