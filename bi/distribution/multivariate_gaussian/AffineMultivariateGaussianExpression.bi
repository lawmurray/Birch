import distribution.MultivariateGaussian;
import math;
import assert;
import io;

/**
 * Multivariate Gaussian that has a mean which is an affine transformation of
 * another multivariate Gaussian.
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
   * Vector of affine transformation.
   */
  c:Real[R];
  
  /**
   * Parent.
   */
  u:MultivariateGaussian;
  
  /**
   * Constructor.
   */
  function initialise(A:Real[_,_], c:Real[_], u:MultivariateGaussian) {
    assert(rows(A) == R && columns(A) == C);
    assert(length(c) == R);
    assert(u.D == C);
  
    this.A <- A;
    this.c <- c;
    this.u <- u;
  }
}

function u:MultivariateGaussian + c:Real[_] -> AffineMultivariateGaussianExpression {
    print(u.D); print("\n");
  assert(u.D == length(c));
  v:AffineMultivariateGaussianExpression(u.D, u.D);
  v.initialise(identity(u.D, u.D), c, u);
  return v;
}

function c:Real[_] + u:MultivariateGaussian -> AffineMultivariateGaussianExpression {
  return u + c;
}

function u:MultivariateGaussian - c:Real[_] -> AffineMultivariateGaussianExpression {
  assert(u.D == length(c));
  v:AffineMultivariateGaussianExpression(u.D, u.D);
  v.initialise(identity(u.D, u.D), -c, u);
  return v;
}

function c:Real[_] - u:MultivariateGaussian -> AffineMultivariateGaussianExpression {
  assert(u.D == length(c));
  v:AffineMultivariateGaussianExpression(u.D, u.D);
  v.initialise(-identity(u.D, u.D), c, u);
  return v;
}


function A:Real[_,_]*u:MultivariateGaussian -> AffineMultivariateGaussianExpression {
  assert(columns(A) == u.D);
  R:Integer <- rows(A);
  C:Integer <- columns(A);
  v:AffineMultivariateGaussianExpression(R, C);
  v.initialise(A, vector(0.0, R), u);
  return v;
}

function u:AffineMultivariateGaussianExpression + c:Real[_] -> AffineMultivariateGaussianExpression {
  v:AffineMultivariateGaussianExpression(u.R, u.C);
  v.initialise(u.A, u.c + c, u.u);
  return v;
}

function c:Real[_] + u:AffineMultivariateGaussianExpression -> AffineMultivariateGaussianExpression {
  return u + c;
}

function u:AffineMultivariateGaussianExpression - c:Real[_] -> AffineMultivariateGaussianExpression {
  v:AffineMultivariateGaussianExpression(u.R, u.C);
  v.initialise(u.A, u.c - c, u.u);
  return v;
}

function c:Real[_] - u:AffineMultivariateGaussianExpression -> AffineMultivariateGaussianExpression {
  v:AffineMultivariateGaussianExpression(u.R, u.C);
  v.initialise(-u.A, c - u.c, u.u);
  return v;
}


function A:Real[_,_]*u:AffineMultivariateGaussianExpression -> AffineMultivariateGaussianExpression {
  v:AffineMultivariateGaussianExpression(rows(A), u.C);
  v.initialise(A*u.A, A*u.c, u.u);
  return v;
}
