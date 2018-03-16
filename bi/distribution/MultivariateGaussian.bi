/**
 * Multivariate Gaussian distribution.
 *
 *   - D: Number of dimensions.
 */
class MultivariateGaussian(D:Integer) < Random<Real[_]> {
  /**
   * Mean.
   */
  μ:Real[D];
  
  /**
   * Covariance.
   */
  Σ:Real[D,D];

  function size() -> Integer {
    return length(μ);
  }

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
    if (isMissing()) {
      set(simulate_multivariate_gaussian(μ, Σ));
    } else {
      setWeight(observe_multivariate_gaussian(value(), μ, Σ));
    }
  }
}

/**
 * Synonym for MultivariateGaussian.
 */
class MultivariateNormal = MultivariateGaussian;

/**
 * Create multivariate Gaussian distribution.
 */
function Gaussian(μ:Real[_], Σ:Real[_,_]) -> MultivariateGaussian {
  D:Integer <- length(μ);
  assert D > 0;
  assert rows(Σ) == D;
  assert columns(Σ) == D;
  m:MultivariateGaussian(D);
  m.initialize(μ, Σ);
  return m;
}

/**
 * Create multivariate Gaussian distribution.
 */
function Normal(μ:Real[_], Σ:Real[_,_]) -> MultivariateGaussian {
  return Gaussian(μ, Σ);
}
