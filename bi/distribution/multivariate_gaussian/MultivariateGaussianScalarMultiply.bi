import distribution.MultivariateGaussian;

/**
 * Variate that is scalar multiple of a MultivariateGaussian variate.
 */
class MultivariateGaussianScalarMultiply(D1:Integer) < MultivariateGaussian(D1) {
  /**
   * Scalar operand.
   */
  a:Real;
  
  /**
   * MultivariateGaussian operand.
   */
  u:MultivariateGaussian;

  function initialise(a:Real, u:MultivariateGaussian) {
    super.initialise(u);
    this.a <- a;
    this.u <- u;
  }

  function isDeterministic() -> Boolean {
    return true;
  }
  
  function doMarginalise() {
    this.μ <- a*u.μ;
    this.L <- abs(a)*u.L;
  }
  
  function doForward() {
    set(a*u.x);
  }
  
  function doCondition() {
    u.set(x/a);
  }
}

function a:Real*u:MultivariateGaussian -> MultivariateGaussian {
  v:MultivariateGaussianScalarMultiply(u.D);
  v.initialise(a, u);
  return v;
}

function u:MultivariateGaussian*a:Real -> MultivariateGaussian {
  v:MultivariateGaussianScalarMultiply(u.D);
  v.initialise(a, u);
  return v;
}

function u:MultivariateGaussian/a:Real -> MultivariateGaussian {
  v:MultivariateGaussianScalarMultiply(u.D);
  v.initialise(1.0/a, u);
  return v;
}
