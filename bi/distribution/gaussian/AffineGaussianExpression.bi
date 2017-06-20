import distribution.Gaussian;
import math;
import assert;

/**
 * Expression used to accumulate affine transformations of Gaussians.
 */
class AffineGaussianExpression {
  /**
   * Multiplicative scalar of affine transformation.
   */
  a:Real;
  
  /**
   * Parent.
   */
  u:Gaussian;

  /**
   * Additive scalar of affine transformation.
   */
  c:Real;
  
  /**
   * Constructor.
   */
  function initialize(a:Real, u:Gaussian, c:Real) {
    this.a <- a;
    this.u <- u;
    this.c <- c;
  }
}

operator u:Gaussian + c:Real -> AffineGaussianExpression {
  v:AffineGaussianExpression;
  v.initialize(1.0, u, c);
  return v;
}

operator c:Real + u:Gaussian -> AffineGaussianExpression {
  return u + c;
}

operator u:Gaussian - c:Real -> AffineGaussianExpression {
  v:AffineGaussianExpression;
  v.initialize(1.0, u, -c);
  return v;
}

operator c:Real - u:Gaussian -> AffineGaussianExpression {
  v:AffineGaussianExpression;
  v.initialize(-1.0, u, c);
  return v;
}

operator a:Real*u:Gaussian -> AffineGaussianExpression {
  v:AffineGaussianExpression;
  v.initialize(1.0, u, 0.0);
  return v;
}

operator u:AffineGaussianExpression + c:Real -> AffineGaussianExpression {
  v:AffineGaussianExpression;
  v.initialize(u.a, u.u, u.c + c);
  return v;
}

operator c:Real + u:AffineGaussianExpression -> AffineGaussianExpression {
  return u + c;
}

operator u:AffineGaussianExpression - c:Real -> AffineGaussianExpression {
  v:AffineGaussianExpression;
  v.initialize(u.a, u.u, u.c - c);
  return v;
}

operator c:Real - u:AffineGaussianExpression -> AffineGaussianExpression {
  v:AffineGaussianExpression;
  v.initialize(-u.a, u.u, c - u.c);
  return v;
}

operator a:Real*u:AffineGaussianExpression -> AffineGaussianExpression {
  v:AffineGaussianExpression;
  v.initialize(a*u.a, u.u, a*u.c);
  return v;
}
