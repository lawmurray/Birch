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

  function isMultivariateNormalInverseGamma(σ2:Expression<Real>) -> Boolean {
    return Σ.isScaledInverseGamma(σ2);
  }
  
  function getMultivariateNormalInverseGamma(σ2:Expression<Real>) -> DelayMultivariateNormalInverseGamma {
    A:Real[_,_];
    s2:DelayInverseGamma?;
    (A, s2) <- σ2.getMultivariateScaledInverseGamma(σ2);
    m:DelayMultivariateNormalInverseGamma(this, μ.value(), inv(A), s2!);
    return m;
  }
  
  function graft() {
    if (μ.isMultivariateGaussian()) {
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
