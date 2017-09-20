/**
 * Multivariate Gaussian with conjugate prior on mean.
 *
 * `D` Number of dimensions.
 */
class MultivariateGaussianWithMultivariateGaussianMean(D:Integer)
    < MultivariateGaussian(D) {
  /**
   * Mean.
   */
  μ:MultivariateGaussian(D);
  
  /**
   * Covariance.
   */
  Σ:Real[D,D];
  
  /**
   * Marginalized prior mean.
   */
  μ_0:Real[D];
  
  /**
   * Marginalized prior covariance.
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
    μ_0 <- μ.x;
    Σ_0 <- Σ;
    update(μ_0, Σ_0);
  }
  
  function doCondition() {
    K:Real[D,D] <- μ.Σ*inverse(Σ_0);
    μ.update(μ.μ + K*(x - μ_0), μ.Σ - K*μ.Σ);
  }
}

function Gaussian(μ:MultivariateGaussian, Σ:Real[_,_]) ->
    MultivariateGaussian {
  x:MultivariateGaussianWithMultivariateGaussianMean(μ.D);
  x.initialize(μ, Σ);
  return x;
}

function Normal(μ:MultivariateGaussian, Σ:Real[_,_]) ->
    MultivariateGaussian {
  return Gaussian(μ, Σ);
}
