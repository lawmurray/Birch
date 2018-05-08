/**
 * Multivariate Gaussian distribution with independent components of
 * identical variance.
 */
class MultivariateIndependentGaussian(μ:Expression<Real[_]>,
    σ2:Expression<Real>) < Distribution<Real[_]> {
  /**
   * Mean.
   */
  μ:Expression<Real[_]> <- μ;
  
  /**
   * Variance.
   */
  σ2:Expression<Real> <- σ2;
  
  function graft() -> DelayValue<Real[_]> {
    s2:DelayInverseGamma?;
    m1:TransformMultivariateLinearNormalInverseGamma?;
    m2:DelayMultivariateNormalInverseGamma?;
    m3:TransformMultivariateLinearGaussian?;
    m4:DelayMultivariateGaussian?;

    if (s2 <- σ2.graftInverseGamma())? {
      if (m1 <- μ.graftMultivariateLinearNormalInverseGamma(σ2))? {
        return DelayMultivariateLinearNormalInverseGammaGaussian(m1!.A,
            m1!.x, m1!.c);
      } else if (m2 <- μ.graftMultivariateNormalInverseGamma(σ2))? {
        return DelayMultivariateNormalInverseGammaGaussian(m2!);
      } else {
        return DelayMultivariateInverseGammaGaussian(μ, s2!);
      }
    } else if (m3 <- μ.graftMultivariateLinearGaussian())? {
      return DelayMultivariateLinearGaussianGaussian(m3!.A, m3!.x, m3!.c,
          diagonal(σ2.value(), m3!.size()));
    } else if (m4 <- μ.graftMultivariateGaussian())? {
      return DelayMultivariateGaussianGaussian(m4!, diagonal(σ2, m4!.size()));
    } else {
      μ1:Real[_] <- μ.value();
      return DelayMultivariateGaussian(μ1, diagonal(σ2, length(μ1)));
    }
  }

  function graftMultivariateGaussian() -> DelayMultivariateGaussian? {
    m1:TransformMultivariateLinearGaussian?;
    m2:DelayMultivariateGaussian?;

    if (m1 <- μ.graftMultivariateLinearGaussian())? {
      return DelayMultivariateLinearGaussianGaussian(m1!.A, m1!.x, m1!.c,
          diagonal(σ2, length(m1!.c)));
    } else if (m2 <- μ.graftMultivariateGaussian())? {
      return DelayMultivariateGaussianGaussian(m2!, diagonal(σ2, length(m2!.μ)));
    } else {
      μ1:Real[_] <- μ.value();
      return DelayMultivariateGaussian(μ1, diagonal(σ2, length(μ1)));
    }
  }

  function graftMultivariateNormalInverseGamma(σ2:Expression<Real>) ->
      DelayMultivariateNormalInverseGamma? {
    s1:TransformScaledInverseGamma?;
    s2:DelayInverseGamma?;
    if (s1 <- this.σ2.graftScaledInverseGamma(σ2))? {
      μ1:Real[_] <- μ.value();
      D:Integer <- length(μ1);
      return DelayMultivariateNormalInverseGamma(μ1, diagonal(s1!.a2, D),
          s1!.σ2);
    } else if Object(this.σ2) == σ2 && (s2 <- this.σ2.graftInverseGamma())? {
      μ1:Real[_] <- μ.value();
      D:Integer <- length(μ1);
      return DelayMultivariateNormalInverseGamma(μ1, identity(D), s2!);
    } else {
      return nil;
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
