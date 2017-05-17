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
  function initialise(a:Real, u:Gaussian, c:Real) {
    this.a <- a;
    this.u <- u;
    this.c <- c;
  }
}

function u:Gaussian + c:Real -> AffineGaussianExpression {
  v:AffineGaussianExpression;
  v.initialise(1.0, u, c);
  return v;
}

function c:Real + u:Gaussian -> AffineGaussianExpression {
  return u + c;
}

function u:Gaussian - c:Real -> AffineGaussianExpression {
  v:AffineGaussianExpression;
  v.initialise(1.0, u, -c);
  return v;
}

function c:Real - u:Gaussian -> AffineGaussianExpression {
  v:AffineGaussianExpression;
  v.initialise(-1.0, u, c);
  return v;
}

function a:Real*u:Gaussian -> AffineGaussianExpression {
  v:AffineGaussianExpression;
  v.initialise(1.0, u, 0.0);
  return v;
}

function u:AffineGaussianExpression + c:Real -> AffineGaussianExpression {
  v:AffineGaussianExpression;
  v.initialise(u.a, u.u, u.c + c);
  return v;
}

function c:Real + u:AffineGaussianExpression -> AffineGaussianExpression {
  return u + c;
}

function u:AffineGaussianExpression - c:Real -> AffineGaussianExpression {
  v:AffineGaussianExpression;
  v.initialise(u.a, u.u, u.c - c);
  return v;
}

function c:Real - u:AffineGaussianExpression -> AffineGaussianExpression {
  v:AffineGaussianExpression;
  v.initialise(-u.a, u.u, c - u.c);
  return v;
}

function a:Real*u:AffineGaussianExpression -> AffineGaussianExpression {
  v:AffineGaussianExpression;
  v.initialise(a*u.a, u.u, a*u.c);
  return v;
}
