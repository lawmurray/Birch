import delay.DelayRealVector;
import distribution.Gaussian;
import math;
import random;

/**
 * Multivariate Gaussian distribution.
 *
 * `D` Number of dimensions.
 */
class MultivariateGaussian(D:Integer) < DelayRealVector(D) {
  /**
   * Mean.
   */
  μ:Real[D];
  
  /**
   * Covariance matrix.
   */
  Σ:Real[D,D];

  function initialize(u:MultivariateGaussian) {
    super.initialize(u);
  }

  function initialize(μ:Real[_], Σ:Real[_,_]) {
    super.initialize();
    update(μ, Σ);
  }

  function update(μ:Real[_], Σ:Real[_,_]) {
    this.μ <- μ;
    this.Σ <- Σ;
  }

  function doRealize() {
    if (missing) {
      d:Integer;
      for (d in 1..D) {
        x[d] <~ Gaussian(0.0, 1.0);
      }
      set(μ + llt(Σ)*x);
    } else {
      L:Real[D,D];
      L <- llt(Σ);
      setWeight(-0.5*squaredNorm(solve(L, x - μ)) - log(determinant(L)) - 0.5*Real(D)*log(2.0*π));
    }
  }
}

/**
 * Create.
 */
function Gaussian(μ:Real[_], Σ:Real[_,_]) -> MultivariateGaussian {
  D:Integer <- length(μ);
  assert rows(Σ) == D;
  assert columns(Σ) == D;
  m:MultivariateGaussian(D);
  m.initialize(μ, Σ);
  return m;
}
