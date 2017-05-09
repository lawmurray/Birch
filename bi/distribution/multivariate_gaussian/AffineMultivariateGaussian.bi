import distribution.MultivariateGaussian;
import distribution.multivariate_gaussian.AffineMultivariateGaussianExpression;
import math;
import assert;

/**
 * Multivariate Gaussian that has a mean which is an affine transformation of
 * another multivariate Gaussian.
 */
class AffineMultivariateGaussian(R1:Integer,C1:Integer) < MultivariateGaussian(R1) {
  /**
   * Number of rows in matrix.
   */
  R:Integer <- R1;
  
  /**
   * Number of columns in matrix.
   */
  C:Integer <- C1;

  /**
   * Parent.
   */
  u:MultivariateGaussian;
  
  /**
   * Standard deviation.
   */
  L1:Real[R,R];
  
  /**
   * Matrix of affine transformation.
   */
  A:Real[R,C];
  
  /**
   * Vector of affine transformation.
   */
  c:Real[R];

  function initialise(μ:MultivariateGaussian, L:Real[_,_], A:Real[_,_], c:Real[_]) {
    super.initialise(μ);
    this.u <- μ;
    this.L1 <- L;
    this.A <- A;
    this.c <- c;
  }
  
  function doMarginalise() {
    this.μ <- A*u.μ + c;
    this.L <- llt(A*u.L*transpose(A*u.L) + L1*transpose(L1));
  }

  function doForward() {
    this.μ <- A*u.x + c;
    this.L <- L1;
  }
  
  function doCondition() {
    
  }
}

function Gaussian(μ:MultivariateGaussian, L:Real[_,_]) -> MultivariateGaussian {
  D:Integer <- μ.D;
  v:AffineMultivariateGaussian(D, D);
  v.initialise(μ, L, identity(D, D), vector(0.0, D));
  return v;
}

function Gaussian(μ:AffineMultivariateGaussianExpression, L:Real[_,_]) -> MultivariateGaussian {
  R:Integer <- μ.R;
  C:Integer <- μ.C;
  v:AffineMultivariateGaussian(R, C);
  v.initialise(μ.u, L, μ.A, μ.c);
  return v;
}
