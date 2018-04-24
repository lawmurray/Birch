/**
 * Multivariate Gaussian distribution.
 */
class MultivariateGaussian(μ:Expression<Real[_]>, Σ:Expression<Real[_,_]>) < Random<Real[_]> {
  /**
   * Mean.
   */
  μ:Expression<Real[_]> <- μ;
  
  /**
   * Covariance.
   */
  Σ:Expression<Real[_,_]> <- Σ;
  
  function isMultivariateGaussian() -> Boolean {
    return isMissing();
  }

  function getMultivariateGaussian() -> DelayMultivariateGaussian {
    assert isMultivariateGaussian();
    return DelayMultivariateGaussian?(delay)!;
  }

  function isMultivariateAffineGaussian() -> Boolean {
    return isMultivariateGaussian();
  }

  function getMultivariateAffineGaussian() -> (Real[_,_], DelayMultivariateGaussian, Real[_]) {
    D:Integer <- length(μ);
    return (identity(D), getMultivariateGaussian(), vector(0.0, D));
  }

  /*function isMultivariateNormalInverseGamma(σ2:Expression<Real>) -> Boolean {
    return σ2.isScaledInverseGamma(σ2);
  }
  
  function getMultivariateNormalInverseGamma(σ2:Expression<Real>) -> DelayMultivariateNormalInverseGamma {
    return DelayMultivariateNormalInverseGamma?(delay)!;
  }

  function isMultivariateAffineNormalInverseGamma(σ2:Expression<Real>) -> Boolean {
    return isMultivariateNormalInverseGamma(σ2);
  }
  
  function getMultivariateAffineNormalInverseGamma(σ2:Expression<Real>) -> (Real, DelayMultivariateNormalInverseGamma, Real) {
    return (1.0, getMultivariateNormalInverseGamma(σ2), 0.0);
  }*/
  
  function graft() {
    /*if (μ.isMultivariateNormalInverseGamma(σ2)) {
      m:DelayMultivariateNormalInverseGammaGaussian(this, μ.getMultivariateNormalInverseGamma(σ2));
      m.graft();
      delay <- m;
    } else if (μ.isMultivariateAffineNormalInverseGamma(σ2)) {
      A:Real[_,_];
      μ_0:DelayMultivariateNormalInverseGamma?;
      c:Real[_];
      (A, μ_0, c) <- μ.getMultivariateAffineNormalInverseGamma(σ2);
      m:DelayMultivariateAffineNormalInverseGammaGaussian(this, a, μ_0!, c);
      m.graft();
      delay <- m;
    } else if (σ2.isInverseGamma()) {
      m:DelayInverseGammaGaussian(this, μ.value(), σ2.getInverseGamma());
      m.graft();
      delay <- m;
    } else */if (μ.isMultivariateGaussian()) {
      m:DelayMultivariateGaussianGaussian(this, μ.getMultivariateGaussian(), Σ.value());
      m.graft();
      delay <- m;
    } else if (μ.isMultivariateAffineGaussian()) {
      A:Real[_,_];
      μ_0:DelayMultivariateGaussian?;
      c:Real[_];
      (A, μ_0, c) <- μ.getMultivariateAffineGaussian();
      m:DelayMultivariateAffineGaussianGaussian(this, A, μ_0!, c, Σ.value());
      m.graft();
      delay <- m;
    } else {
      m:DelayMultivariateGaussian(this, μ.value(), Σ.value());
      m.graft();
      delay <- m;
    }
  }
}

/**
 * Create multivariate Gaussian distribution.
 */
function Gaussian(μ:Expression<Real[_]>, Σ:Expression<Real[_,_]>) -> MultivariateGaussian {
  m:MultivariateGaussian(μ, Σ);
  return m;
}

/**
 * Create multivariate Gaussian distribution.
 */
function Gaussian(μ:Expression<Real[_]>, Σ:Real[_,_]) -> MultivariateGaussian {
  return Gaussian(μ, Literal(Σ));
}

/**
 * Create multivariate Gaussian distribution.
 */
function Gaussian(μ:Real[_], Σ:Expression<Real[_,_]>) -> MultivariateGaussian {
  return Gaussian(Literal(μ), Σ);
}

/**
 * Create multivariate Gaussian distribution.
 */
function Gaussian(μ:Real[_], Σ:Real[_,_]) -> MultivariateGaussian {
  return Gaussian(Literal(μ), Literal(Σ));
}
