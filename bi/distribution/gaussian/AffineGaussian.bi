import distribution.Gaussian;
import distribution.gaussian.AffineGaussianExpression;
import math;
import assert;

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
   * Marginalised prior mean.
   */
  y:Real;
  
  /**
   * Marginalised prior variance.
   */
  s:Real;

  function initialise(a:Real, μ:Gaussian, c:Real, q:Real) {
    super.initialise(μ);
    this.a <- a;
    this.μ <- μ;
    this.c <- c;
    this.q <- q;
  }
  
  function doMarginalise() {
    this.y <- a*μ.μ + c;
    this.s <- pow(a, 2.0)*μ.σ2 + q;
    update(y, s);
  }

  function doForward() {
    this.y <- a*μ.x + c;
    this.s <- q;
    update(y, s);
  }
  
  function doCondition() {
    k:Real <- μ.σ2*a/s;
    μ.update(μ.μ + k*(x - y), μ.σ2 - k*a*μ.σ2);
  }

  function copy(o:AffineGaussian) {
    super.copy(o);
    this.a <- o.a;
    this.μ.copy(o.μ);
    this.c <- o.c;
    this.q <- o.q;
    this.y <- o.y;
    this.s <- o.s;
    
    /* update graph edges */
    setParent(μ);
    if (isMarginalised()) {
      μ.setChild(this);
    }
  }
}

function Gaussian(μ:Gaussian, q:Real) -> Gaussian {
  v:AffineGaussian;
  v.initialise(1.0, μ, 0.0, q);
  return v;
}

function Gaussian(μ:AffineGaussianExpression, q:Real) -> Gaussian {
  v:AffineGaussian;
  v.initialise(μ.a, μ.u, μ.c, q);
  return v;
}
