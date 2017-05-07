import distribution.Gaussian;

/**
 * Variate that is scalar multiple of a Gaussian variate.
 */
class GaussianMultiply < Gaussian {
  /**
   * Scalar operand.
   */
  a:Real;
  
  /**
   * Gaussian operand.
   */
  u:Gaussian;

  function isDeterministic() -> Boolean {
    return true;
  }

  function construct(a:Real, u:Gaussian) {
    super.construct();
    this.a <- a;
    this.u <- u;
  }
  
  function doMarginalise() {
    this.μ <- a*u.μ;
    this.σ <- abs(a*u.σ);
  }
  
  function doForward() {
    set(a*u.x);
  }
  
  function doCondition() {
    u.set(x/a);
  }
}

function a:Real*u:Gaussian -> Gaussian {
  v:GaussianMultiply;
  v.construct(a, u);
  v.initialise(u);
  return v;
}

function u:Gaussian*a:Real -> Gaussian {
  v:GaussianMultiply;
  v.construct(a, u);
  v.initialise(u);
  return v;
}
