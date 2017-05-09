import distribution.MultivariateGaussian;
import math;
import assert;

/**
 * Variate that is matrix multiple of a MultivariateGaussian variate.
 */
class MultivariateGaussianMatrixLeftMultiply(D1:Integer) < MultivariateGaussian(D1) {
  /**
   * Scalar operand.
   */
  A:Real[D1,D1];
  
  /**
   * MultivariateGaussian operand.
   */
  u:MultivariateGaussian;

  function initialise(A:Real[_,_], u:MultivariateGaussian) {
    assert(rows(A) == u.D && columns(A) == u.D);
  
    super.initialise(u);
    this.A <- A;
    this.u <- u;
  }

  function isDeterministic() -> Boolean {
    return true;
  }
  
  function doMarginalise() {
    this.μ <- A*u.μ;
    
    X:Real[D,D];
    X <- A*u.L;
    this.L <- llt(X*transpose(X));
  }
  
  function doForward() {
    set(A*u.x);
  }
  
  function doCondition() {
    u.set(solve(A, x));
  }
}

function A:Real[_,_]*u:MultivariateGaussian -> MultivariateGaussian {
  assert(rows(A) == u.D && columns(A) == u.D);

  v:MultivariateGaussianMatrixLeftMultiply(u.D);
  v.initialise(A, u);
  return v;
}
