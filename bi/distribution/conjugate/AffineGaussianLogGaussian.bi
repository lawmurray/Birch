/*
 * Log-Gaussian with affine transformation of Gaussian as mean.
 */
class AffineGaussianLogGaussian < LogGaussian {
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
    if (x.isRealized()) {
      μ_0 <- a*x.value() + c;
      σ2_0 <- σ2;
    } else {
      μ_0 <- a*x.μ + c;
      σ2_0 <- a*a*x.σ2 + σ2;
    }
    update(μ_0, σ2_0);
  }
  
  function doCondition() {
    k:Real <- x.σ2*a/σ2_0;
    x.update(x.μ + k*(log(value()) - μ_0), x.σ2 - k*a*x.σ2);
  }
}

/**
 * Create log-Gaussian distribution.
 */
function LogGaussian(μ:AffineExpression, σ2:Real) -> LogGaussian {
  x:Gaussian? <- Gaussian?(μ.x);
  if (x?) {
    y:AffineGaussianLogGaussian;
    y.initialize(μ.a, x!, μ.c, σ2);
    return y;
  } else {
    return LogGaussian(μ.value(), σ2);
  }
}

/**
 * Create log-Gaussian distribution.
 */
function LogNormal(μ:AffineExpression, σ2:Real) -> LogGaussian {
  return LogGaussian(μ, σ2);
}
