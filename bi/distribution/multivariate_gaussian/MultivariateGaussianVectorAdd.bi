import distribution.MultivariateGaussian;
import distribution.multivariate_gaussian.MultivariateGaussianScalarMultiply;

/**
 * Variate that is scalar multiple of a MultivariateGaussian variate.
 */
class MultivariateGaussianVectorAdd(D1:Integer) < MultivariateGaussian(D1) {
  /**
   * Scalar operand.
   */
  a:Real[D1];
  
  /**
   * MultivariateGaussian operand.
   */
  u:MultivariateGaussian;

  function initialise(a:Real[_], u:MultivariateGaussian) {
    super.initialise(u);
    this.a <- a;
    this.u <- u;
  }

  function isDeterministic() -> Boolean {
    return true;
  }
  
  function doMarginalise() {
    this.μ <- a + u.μ;
    this.L <- u.L;
  }
  
  function doForward() {
    set(a + u.x);
  }
  
  function doCondition() {
    u.set(x - a);
  }
}

function a:Real[_] + u:MultivariateGaussian -> MultivariateGaussian {
  v:MultivariateGaussianVectorAdd(u.D);
  v.initialise(a, u);
  return v;
}

function u:MultivariateGaussian + a:Real[_] -> MultivariateGaussian {
  v:MultivariateGaussianVectorAdd(u.D);
  v.initialise(a, u);
  return v;
}

function a:Real[_] - u:MultivariateGaussian -> MultivariateGaussian {
  v:MultivariateGaussianVectorAdd(u.D);
  v.initialise(a, -1.0*u);
  return v;
}

function u:MultivariateGaussian - a:Real[_] -> MultivariateGaussian {
  v:MultivariateGaussianVectorAdd(u.D);
  v.initialise(-a, u);
  return v;
}
