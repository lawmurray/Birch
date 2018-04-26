/**
 * Multivariate Gaussian distribution with independent components of
 * identical variance.
 */
class MultivariateIndependentGaussian(μ:Expression<Real[_]>,
    σ2:Expression<Real>) < Random<Real[_]> {
  /**
   * Mean.
   */
  μ:Expression<Real[_]> <- μ;
  
  /**
   * Variance.
   */
  σ2:Expression<Real> <- σ2;
  
  function isMultivariateGaussian() -> Boolean {
    return isMissing();
  }

  function getMultivariateGaussian() -> DelayMultivariateGaussian {
    assert isMultivariateGaussian();
    return DelayMultivariateGaussian?(delay)!;
  }

  function isMultivariateNormalInverseGamma(σ2:Expression<Real>) -> Boolean {
    return σ2.isScaledInverseGamma(σ2);
  }
  
  function getMultivariateNormalInverseGamma(σ2:Expression<Real>) -> DelayMultivariateNormalInverseGamma {
    A:Real[_,_];
    s2:DelayInverseGamma?;
    (A, s2) <- σ2.getMultivariateScaledInverseGamma(σ2);
    m:DelayMultivariateNormalInverseGamma(this, μ.value(), inv(A), s2!);
    return m;
  }
  
  function graft() {
    if (μ.isMultivariateNormalInverseGamma(σ2)) {
      m:DelayMultivariateNormalInverseGammaGaussian(this, μ.getMultivariateNormalInverseGamma(σ2));
      m.graft();
      delay <- m;
    } else if (μ.isMultivariateAffineNormalInverseGamma(σ2)) {
      A:Real[_,_];
      μ_0:DelayMultivariateNormalInverseGamma?;
      c:Real[_];
      (A, μ_0, c) <- μ.getMultivariateAffineNormalInverseGamma(σ2);
      m:DelayMultivariateAffineNormalInverseGammaGaussian(this, A, μ_0!, c);
      m.graft();
      delay <- m;
    } else if (σ2.isInverseGamma()) {
      m:DelayMultivariateInverseGammaGaussian(this, μ.value(), σ2.getInverseGamma());
      m.graft();
      delay <- m;
    } else if (μ.isMultivariateGaussian()) {
      μ_0:DelayMultivariateGaussian <- μ.getMultivariateGaussian();
      m:DelayMultivariateGaussianGaussian(this, μ_0, diagonal(σ2.value(), μ_0.size()));
      m.graft();
      delay <- m;
    } else if (μ.isMultivariateAffineGaussian()) {
      A:Real[_,_];
      μ_0:DelayMultivariateGaussian?;
      c:Real[_];
      (A, μ_0, c) <- μ.getMultivariateAffineGaussian();
      m:DelayMultivariateAffineGaussianGaussian(this, A, μ_0!, c, diagonal(σ2.value(), μ_0!.size()));
      m.graft();
      delay <- m;
    } else {
      μ_0:Real[_] <- μ.value();
      m:DelayMultivariateGaussian(this, μ_0, diagonal(σ2.value(), length(μ_0)));
      m.graft();
      delay <- m;
    }
  }
}

/**
 * Create multivariate Gaussian distribution.
 */
function Gaussian(μ:Expression<Real[_]>, σ2:Expression<Real>) -> MultivariateIndependentGaussian {
  m:MultivariateIndependentGaussian(μ, σ2);
  return m;
}

/**
 * Create multivariate Gaussian distribution.
 */
function Gaussian(μ:Expression<Real[_]>, σ2:Real) -> MultivariateIndependentGaussian {
  return Gaussian(μ, Boxed(σ2));
}

/**
 * Create multivariate Gaussian distribution.
 */
function Gaussian(μ:Real[_], σ2:Expression<Real>) -> MultivariateIndependentGaussian {
  return Gaussian(Boxed(μ), σ2);
}

/**
 * Create multivariate Gaussian distribution.
 */
function Gaussian(μ:Real[_], σ2:Real) -> MultivariateIndependentGaussian {
  return Gaussian(Boxed(μ), Boxed(σ2));
}
