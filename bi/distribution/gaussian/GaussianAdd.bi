import distribution.Gaussian;
import distribution.gaussian.GaussianMultiply;

/**
 * Variate that is scalar multiple of a Gaussian variate.
 */
class GaussianAdd < Gaussian {
  /**
   * Scalar operand.
   */
  a:Real;
  
  /**
   * Gaussian operand.
   */
  u:Gaussian;

  function initialise(a:Real, u:Gaussian) {
    super.initialise(u);
    this.a <- a;
    this.u <- u;
  }

  function isDeterministic() -> Boolean {
    return true;
  }
  
  function doMarginalise() {
    this.μ <- a + u.μ;
    this.σ <- u.σ;
  }
  
  function doForward() {
    set(a + u.x);
  }
  
  function doCondition() {
    u.set(x - a);
  }
}

function a:Real + u:Gaussian -> Gaussian {
  v:GaussianAdd;
  v.initialise(a, u);
  return v;
}

function u:Gaussian + a:Real -> Gaussian {
  v:GaussianAdd;
  v.initialise(a, u);
  return v;
}

function a:Real - u:Gaussian -> Gaussian {
  v:GaussianAdd;
  v.initialise(a, -1.0*u);
  return v;
}

function u:Gaussian - a:Real -> Gaussian {
  v:GaussianAdd;
  v.initialise(-a, u);
  return v;
}
