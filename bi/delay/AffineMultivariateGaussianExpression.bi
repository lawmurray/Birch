import distribution.MultivariateGaussian;
import math;

/**
 * Expression used to accumulate affine transformations of multivariate
 * Gaussians.
 *
 * `R` Number of rows in transformation.
 * `C` Number of columns in transformation.
 */
class AffineMultivariateGaussianExpression(R:Integer, C:Integer) {
  /**
   * Matrix of affine transformation.
   */
  A:Real[R,C];
  
  /**
   * Parent.
   */
  u:MultivariateGaussian;

  /**
   * Vector of affine transformation.
   */
  c:Real[R];
    
  /**
   * Initialize.
   */
  function initialize(A:Real[_,_], u:MultivariateGaussian, c:Real[_]) {
    assert rows(A) == R && columns(A) == C;
    assert length(c) == R;
    assert u.D == C;
  
    this.A <- A;
    this.u <- u;
    this.c <- c;
  }
}

operator u:MultivariateGaussian + c:Real[_] -> AffineMultivariateGaussianExpression {
  assert u.D == length(c);
  v:AffineMultivariateGaussianExpression(u.D, u.D);
  v.initialize(identity(u.D, u.D), u, c);
  return v;
}

operator c:Real[_] + u:MultivariateGaussian -> AffineMultivariateGaussianExpression {
  return u + c;
}

operator u:MultivariateGaussian - c:Real[_] -> AffineMultivariateGaussianExpression {
  assert u.D == length(c);
  v:AffineMultivariateGaussianExpression(u.D, u.D);
  v.initialize(identity(u.D, u.D), u, -c);
  return v;
}

operator c:Real[_] - u:MultivariateGaussian -> AffineMultivariateGaussianExpression {
  assert u.D == length(c);
  v:AffineMultivariateGaussianExpression(u.D, u.D);
  v.initialize(-identity(u.D, u.D), u, c);
  return v;
}

operator A:Real[_,_]*u:MultivariateGaussian -> AffineMultivariateGaussianExpression {
  assert columns(A) == u.D;
  R:Integer <- rows(A);
  C:Integer <- columns(A);
  v:AffineMultivariateGaussianExpression(R, C);
  v.initialize(A, u, vector(0.0, R));
  return v;
}

operator u:AffineMultivariateGaussianExpression + c:Real[_] -> AffineMultivariateGaussianExpression {
  v:AffineMultivariateGaussianExpression(u.R, u.C);
  v.initialize(u.A, u.u, u.c + c);
  return v;
}

operator c:Real[_] + u:AffineMultivariateGaussianExpression -> AffineMultivariateGaussianExpression {
  return u + c;
}

operator u:AffineMultivariateGaussianExpression - c:Real[_] -> AffineMultivariateGaussianExpression {
  v:AffineMultivariateGaussianExpression(u.R, u.C);
  v.initialize(u.A, u.u, u.c - c);
  return v;
}

operator c:Real[_] - u:AffineMultivariateGaussianExpression -> AffineMultivariateGaussianExpression {
  v:AffineMultivariateGaussianExpression(u.R, u.C);
  v.initialize(-u.A, u.u, c - u.c);
  return v;
}

operator A:Real[_,_]*u:AffineMultivariateGaussianExpression -> AffineMultivariateGaussianExpression {
  v:AffineMultivariateGaussianExpression(rows(A), u.C);
  v.initialize(A*u.A, u.u, A*u.c);
  return v;
}
