import distribution.Gaussian;
import delay.AffineGaussianExpression;
import math;

/**
 * Gaussian that has a mean which is an affine transformation of another
 * Gaussian.
 */
class AffineGaussian < Gaussian {
  /**
   * Multiplicative scalar of affine transformation.
   */
  a:Real;

  /**
   * Mean.
   */
  μ:Gaussian;
  
  /**
   * Additive scalar of affine transformation.
   */
  c:Real;

  /**
   * Variance.
   */
  q:Real;
  
  /**
   * Marginalized prior mean.
   */
  y:Real;
  
  /**
   * Marginalized prior variance.
   */
  s:Real;

  function initialize(a:Real, μ:Gaussian, c:Real, q:Real) {
    super.initialize(μ);
    this.a <- a;
    this.μ <- μ;
    this.c <- c;
    this.q <- q;
  }
  
  function doMarginalize() {
    y <- a*μ.μ + c;
    s <- pow(a, 2.0)*μ.σ2 + q;
    update(y, s);
  }

  function doForward() {
    y <- a*μ.x + c;
    s <- q;
    update(y, s);
  }
  
  function doCondition() {
    k:Real <- μ.σ2*a/s;
    μ.update(μ.μ + k*(x - y), μ.σ2 - k*a*μ.σ2);
  }
}

function Gaussian(μ:Gaussian, q:Real) -> Gaussian {
  if (μ.isRealized()) {
    v:Gaussian;
    v.initialize(μ.value(), q);
    return v;
  } else {
    v:AffineGaussian;
    v.initialize(1.0, μ, 0.0, q);
    return v;
  }
}

function Gaussian(μ:AffineGaussianExpression, q:Real) -> Gaussian {
  if (μ.u.isRealized()) {
    v:Gaussian;
    v.initialize(μ.a*μ.u.value() + μ.c, q);
    return v;
  } else {
    v:AffineGaussian;
    v.initialize(μ.a, μ.u, μ.c, q);
    return v;
  }
}
