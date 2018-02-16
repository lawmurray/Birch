/*
 * Multivariate Gaussian with affine transformation of another multivariate
 * Gaussian as mean.
 *
 * - D: Number of dimensions.
 */
class MultivariateGaussianWithAffineGaussianMean(D:Integer) <
    MultivariateGaussian(D) {
  /**
   * Scale.
   */
  A:Real[_,_];
  
  /**
   * Random variable.
   */
  x:MultivariateGaussian?;
  
  /**
   * Offset.
   */
  c:Real[_];
  
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

  function initialize(A:Real[_,_], x:MultivariateGaussian, c:Real[_],
      Σ:Real[_,_]) {
    super.initialize(x);
    this.A <- A;
    this.x <- x;
    this.c <- c;
    this.Σ <- Σ;
  }
  
  function doMarginalize() {
    assert x?;
    μ_0 <- A*x!.μ + c;
    Σ_0 <- A*x!.Σ*transpose(A) + Σ;
    update(μ_0, Σ_0);
  }

  function doForward() {
    assert x?;
    μ_0 <- A*x!.value() + c;
    Σ_0 <- Σ;
    update(μ_0, Σ_0);
  }
  
  function doCondition() {
    assert x?;
    K:Real[_,_] <- x!.Σ*transpose(A)*inverse(Σ_0);    
    x!.update(x!.μ + K*(value() - μ_0), x!.Σ - K*A*x!.Σ);
  }
}

/**
 * Create Gaussian distribution.
 */
 function Gaussian(μ:MultivariateAffineExpression, Σ:Real[_,_]) ->
    MultivariateGaussian {
  x:MultivariateGaussian? <- MultivariateGaussian?(μ.x);
  if (x?) {
    y:MultivariateGaussianWithAffineGaussianMean(rows(μ.A));
    y.initialize(μ.A, x!, μ.c, Σ);
    return y;
  } else {
    return Gaussian(μ.value(), Σ);
  }
}

/**
 * Create Gaussian distribution.
 */
function Normal(μ:MultivariateAffineExpression, Σ:Real[_,_]) ->
    MultivariateGaussian {
  return Gaussian(μ, Σ);
}
