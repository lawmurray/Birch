/**
 * Multivariate Gaussian with conjugate prior on mean.
 *
 * `D` Number of dimensions.
 */
class MultivariateGaussianWithMultivariateGaussianExpressionMean(
    D:Integer) < MultivariateGaussian(D) {
  /**
   * Mean.
   */
  μ:MultivariateGaussianExpression?;
  
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

  function initialize(μ:MultivariateGaussianExpression, Σ:Real[_,_]) {
    super.initialize(μ.x);
    this.μ <- μ;
    this.Σ <- Σ;
  }
  
  function doMarginalize() {
    assert μ?;
    m:MultivariateGaussianExpression <- μ!;
    
    μ_0 <- m.A*m.x.μ + m.c;
    Σ_0 <- m.A*m.x.Σ*transpose(m.A) + Σ;
    update(μ_0, Σ_0);
  }

  function doForward() {
    assert μ?;
    m:MultivariateGaussianExpression <- μ!;
    
    μ_0 <- m.A*m.x.x + m.c;
    Σ_0 <- Σ;
    update(μ_0, Σ_0);
  }
  
  function doCondition() {
    assert μ?;
    m:MultivariateGaussianExpression <- μ!;

    K:Real[m.x.D,D] <- m.x.Σ*transpose(m.A)*inverse(Σ_0);    
    m.x.update(m.x.μ + K*(x - μ_0), m.x.Σ - K*m.A*m.x.Σ);
  }
}

/**
 * Create Gaussian distribution.
 */
function Gaussian(μ:MultivariateGaussianExpression, Σ:Real[_,_]) ->
    MultivariateGaussian {
  x:MultivariateGaussianWithMultivariateGaussianExpressionMean(μ.R);
  x.initialize(μ, Σ);
  return x;
}

/**
 * Create Gaussian distribution.
 */
function Normal(μ:MultivariateGaussianExpression, Σ:Real[_,_]) ->
    MultivariateGaussian {
  return Gaussian(μ, Σ);
}
