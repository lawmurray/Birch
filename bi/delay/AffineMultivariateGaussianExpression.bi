import distribution.MultivariateGaussian;
import math;

/**
 * Expression used to accumulate affine transformations of multivariate
 * Gaussians.
 *
 *   - `A` Matrix of affine transformation.
 *   - `u` Parent.
 *   - `c` Vector of affine transformation.
 */
class AffineMultivariateGaussianExpression(A:Real[_,_],
    u:MultivariateGaussian, c:Real[_]) {

  /**
   * Value conversion.
   */
  operator -> Real[_] {
    return A*u.value() + c;
  }
}

operator u:MultivariateGaussian + c:Real[_] -> AffineMultivariateGaussianExpression {
  assert u.D == length(c);
  v:AffineMultivariateGaussianExpression(identity(u.D, u.D), u, c);
  return v;
}

operator c:Real[_] + u:MultivariateGaussian -> AffineMultivariateGaussianExpression {
  return u + c;
}

operator u:MultivariateGaussian - c:Real[_] -> AffineMultivariateGaussianExpression {
  assert u.D == length(c);
  v:AffineMultivariateGaussianExpression(identity(u.D, u.D), u, -c);
  return v;
}

operator c:Real[_] - u:MultivariateGaussian -> AffineMultivariateGaussianExpression {
  assert u.D == length(c);
  v:AffineMultivariateGaussianExpression(-identity(u.D, u.D), u, c);
  return v;
}

operator A:Real[_,_]*u:MultivariateGaussian -> AffineMultivariateGaussianExpression {
  assert columns(A) == u.D;
  v:AffineMultivariateGaussianExpression(A, u, vector(0.0, rows(A)));
  return v;
}

operator u:AffineMultivariateGaussianExpression + c:Real[_] -> AffineMultivariateGaussianExpression {
  v:AffineMultivariateGaussianExpression(u.A, u.u, u.c + c);
  return v;
}

operator c:Real[_] + u:AffineMultivariateGaussianExpression -> AffineMultivariateGaussianExpression {
  return u + c;
}

operator u:AffineMultivariateGaussianExpression - c:Real[_] -> AffineMultivariateGaussianExpression {
  v:AffineMultivariateGaussianExpression(u.A, u.u, u.c - c);
  return v;
}

operator c:Real[_] - u:AffineMultivariateGaussianExpression -> AffineMultivariateGaussianExpression {
  v:AffineMultivariateGaussianExpression(-u.A, u.u, c - u.c);
  return v;
}

operator A:Real[_,_]*u:AffineMultivariateGaussianExpression -> AffineMultivariateGaussianExpression {
  v:AffineMultivariateGaussianExpression(A*u.A, u.u, A*u.c);
  return v;
}
