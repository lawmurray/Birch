/*
 * Multivariate Gaussian with another multivariate Gaussian as mean.
 *
 * - D: Number of dimensions.
 */
class MultivariateGaussianGaussian(D:Integer) < MultivariateGaussian(D) {
  /**
   * Mean.
   */
  μ:MultivariateGaussian(D);
  
  /**
   * Covariance.
   */
  Σ:Real[D,D];
  
  /**
   * Marginal mean.
   */
  μ_0:Real[D];
  
  /**
   * Marginal covariance.
   */
  Σ_0:Real[D,D];

  function initialize(μ:MultivariateGaussian, Σ:Real[_,_]) {
    super.initialize(μ);
    this.μ <- μ;
    this.Σ <- Σ;
  }
  
  function doMarginalize() {
    μ_0 <- μ.μ;
    Σ_0 <- μ.Σ + Σ;
    update(μ_0, Σ_0);
  }

  function doForward() {
    μ_0 <- μ.value();
    Σ_0 <- Σ;
    update(μ_0, Σ_0);
  }
  
  function doCondition() {
    K:Real[_,_] <- μ.Σ*inv(Σ_0);
    μ.update(μ.μ + K*(value() - μ_0), μ.Σ - K*μ.Σ);
  }
}

/**
 * Create Gaussian distribution.
 */
function Gaussian(μ:MultivariateGaussian, Σ:Real[_,_]) ->
    MultivariateGaussian {
  x:MultivariateGaussianGaussian(μ.size());
  x.initialize(μ, Σ);
  return x;
}

/**
 * Create Gaussian distribution.
 */
function Gaussian(μ:Random<Real[_]>, Σ:Real[_,_]) -> MultivariateGaussian {
  μ1:MultivariateGaussian? <- MultivariateGaussian?(μ);
  if (μ1?) {
    return Gaussian(μ1!, Σ);
  } else {
    return Gaussian(μ.value(), Σ);
  }
}

/**
 * Create Gaussian distribution.
 */
function Normal(μ:MultivariateGaussian, Σ:Real[_,_]) -> MultivariateGaussian {
  return Gaussian(μ, Σ);
}

/**
 * Create Gaussian distribution.
 */
function Normal(μ:Random<Real[_]>, Σ:Real[_,_]) -> MultivariateGaussian {
  return Gaussian(μ, Σ);
}
