import distribution.Gaussian;
import math;

/**
 * Expression used to accumulate affine transformations of Gaussians.
 *
 *   - `a` Multiplicative scalar of affine transformation.
 *   - `u` Parent.
 *   - `c` Additive scalar of affine transformation.
 */
class AffineGaussianExpression(a:Real, u:Gaussian, c:Real) {  
  /**
   * Value conversion.
   */
  operator -> Real {
    return a*u.value() + c;
  }
}

operator +u:Gaussian -> Gaussian {
  return u;
}

operator -u:Gaussian -> AffineGaussianExpression {
  v:AffineGaussianExpression(-1.0, u, 0.0);
  return v;
}

operator u:Gaussian + c:Real -> AffineGaussianExpression {
  v:AffineGaussianExpression(1.0, u, c);
  return v;
}

operator c:Real + u:Gaussian -> AffineGaussianExpression {
  return u + c;
}

operator u:Gaussian - c:Real -> AffineGaussianExpression {
  v:AffineGaussianExpression(1.0, u, -c);
  return v;
}

operator c:Real - u:Gaussian -> AffineGaussianExpression {
  v:AffineGaussianExpression(-1.0, u, c);
  return v;
}

operator a:Real*u:Gaussian -> AffineGaussianExpression {
  v:AffineGaussianExpression(1.0, u, 0.0);
  return v;
}

operator +u:AffineGaussianExpression -> AffineGaussianExpression {
  return u;
}

operator -u:AffineGaussianExpression -> AffineGaussianExpression {
  v:AffineGaussianExpression(-u.a, u.u, -u.c);
  return v;
}

operator u:AffineGaussianExpression + c:Real -> AffineGaussianExpression {
  v:AffineGaussianExpression(u.a, u.u, u.c + c);
  return v;
}

operator c:Real + u:AffineGaussianExpression -> AffineGaussianExpression {
  return u + c;
}

operator u:AffineGaussianExpression - c:Real -> AffineGaussianExpression {
  v:AffineGaussianExpression(u.a, u.u, u.c - c);
  return v;
}

operator c:Real - u:AffineGaussianExpression -> AffineGaussianExpression {
  v:AffineGaussianExpression(-u.a, u.u, c - u.c);
  return v;
}

operator a:Real*u:AffineGaussianExpression -> AffineGaussianExpression {
  v:AffineGaussianExpression(a*u.a, u.u, a*u.c);
  return v;
}
