import distribution.MultivariateGaussian;
import math;
import assert;

/**
 * Expression used to accumulate affine transformations of multivariate
 * Gaussians.
 */
class AffineMultivariateGaussianExpression(R1:Integer, C1:Integer) {
  /**
   * Number of rows in matrix.
   */
  R:Integer <- R1;
  
  /**
   * Number of columns in matrix.
   */
  C:Integer <- C1;
  
  /**
   * Matrix of affine transformation.
   */
  A:Real[R,C];
  
  /**
   * Parent.
   */
  u:MultivariateGaussian;

  /**
   * Additive ector of affine transformation.
   */
  c:Real[R];
  
  /**
   * Constructor.
   */
  function initialise(A:Real[_,_], u:MultivariateGaussian, c:Real[_]) {
    assert(rows(A) == R && columns(A) == C);
    assert(length(c) == R);
    assert(u.D == C);
  
    this.A <- A;
    this.u <- u;
    this.c <- c;
  }
}

function u:MultivariateGaussian + c:Real[_] -> AffineMultivariateGaussianExpression {
  assert(u.D == length(c));
  v:AffineMultivariateGaussianExpression(u.D, u.D);
  v.initialise(identity(u.D, u.D), u, c);
  return v;
}

function c:Real[_] + u:MultivariateGaussian -> AffineMultivariateGaussianExpression {
  return u + c;
}

function u:MultivariateGaussian - c:Real[_] -> AffineMultivariateGaussianExpression {
  assert(u.D == length(c));
  v:AffineMultivariateGaussianExpression(u.D, u.D);
  v.initialise(identity(u.D, u.D), u, -c);
  return v;
}

function c:Real[_] - u:MultivariateGaussian -> AffineMultivariateGaussianExpression {
  assert(u.D == length(c));
  v:AffineMultivariateGaussianExpression(u.D, u.D);
  v.initialise(-identity(u.D, u.D), u, c);
  return v;
}

function A:Real[_,_]*u:MultivariateGaussian -> AffineMultivariateGaussianExpression {
  assert(columns(A) == u.D);
  R:Integer <- rows(A);
  C:Integer <- columns(A);
  v:AffineMultivariateGaussianExpression(R, C);
  v.initialise(A, u, vector(0.0, R));
  return v;
}

function u:AffineMultivariateGaussianExpression + c:Real[_] -> AffineMultivariateGaussianExpression {
  v:AffineMultivariateGaussianExpression(u.R, u.C);
  v.initialise(u.A, u.u, u.c + c);
  return v;
}

function c:Real[_] + u:AffineMultivariateGaussianExpression -> AffineMultivariateGaussianExpression {
  return u + c;
}

function u:AffineMultivariateGaussianExpression - c:Real[_] -> AffineMultivariateGaussianExpression {
  v:AffineMultivariateGaussianExpression(u.R, u.C);
  v.initialise(u.A, u.u, u.c - c);
  return v;
}

function c:Real[_] - u:AffineMultivariateGaussianExpression -> AffineMultivariateGaussianExpression {
  v:AffineMultivariateGaussianExpression(u.R, u.C);
  v.initialise(-u.A, u.u, c - u.c);
  return v;
}

function A:Real[_,_]*u:AffineMultivariateGaussianExpression -> AffineMultivariateGaussianExpression {
  v:AffineMultivariateGaussianExpression(rows(A), u.C);
  v.initialise(A*u.A, u.u, A*u.c);
  return v;
}
