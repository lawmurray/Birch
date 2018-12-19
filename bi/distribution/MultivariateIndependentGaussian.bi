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
  
  function graft() {
    if delay? {
      delay!.prune();
    } else {
      s2:DelayInverseGamma?;
      m1:TransformMultivariateLinearNormalInverseGamma?;
      m2:DelayMultivariateNormalInverseGamma?;
      m3:TransformMultivariateLinearGaussian?;
      m4:DelayMultivariateGaussian?;

      if (s2 <- σ2.graftInverseGamma())? {
        if (m1 <- μ.graftMultivariateLinearNormalInverseGamma(σ2))? {
          delay <- DelayMultivariateLinearNormalInverseGammaGaussian(x, m1!.A,
              m1!.x, m1!.c);
        } else if (m2 <- μ.graftMultivariateNormalInverseGamma(σ2))? {
          delay <- DelayMultivariateNormalInverseGammaGaussian(x, m2!);
        } else {
          delay <- DelayMultivariateInverseGammaGaussian(x, μ, s2!);
        }
      } else if (m3 <- μ.graftMultivariateLinearGaussian())? {
        delay <- DelayMultivariateLinearGaussianGaussian(x, m3!.A, m3!.x, m3!.c,
            diagonal(σ2.value(), m3!.size()));
      } else if (m4 <- μ.graftMultivariateGaussian())? {
        delay <- DelayMultivariateGaussianGaussian(x, m4!, diagonal(σ2, m4!.size()));
      } else {
        μ1:Real[_] <- μ.value();
        delay <- DelayMultivariateGaussian(x, μ1, diagonal(σ2, length(μ1)));
      }
    }
  }

  function graftMultivariateGaussian() -> DelayMultivariateGaussian? {
    if delay? {
      delay!.prune();
    } else {
      m1:TransformMultivariateLinearGaussian?;
      m2:DelayMultivariateGaussian?;
      if (m1 <- μ.graftMultivariateLinearGaussian())? {
        delay <- DelayMultivariateLinearGaussianGaussian(x, m1!.A, m1!.x,
            m1!.c, diagonal(σ2, length(m1!.c)));
      } else if (m2 <- μ.graftMultivariateGaussian())? {
        delay <- DelayMultivariateGaussianGaussian(x, m2!, diagonal(σ2,
            length(m2!.μ)));
      } else {
        μ1:Real[_] <- μ.value();
        delay <- DelayMultivariateGaussian(x, μ1, diagonal(σ2, length(μ1)));
      }
    }
    return DelayMultivariateGaussian?(delay);
  }

  function graftMultivariateNormalInverseGamma(σ2:Expression<Real>) ->
      DelayMultivariateNormalInverseGamma? {
    if delay? {
      delay!.prune();
      
      m:DelayMultivariateNormalInverseGamma?;
      s2:DelayInverseGamma?;
      if (m <- DelayMultivariateNormalInverseGamma?(delay))? &&
         (s2 <- σ2.graftInverseGamma())? && m!.σ2! == s2! {
        return m;
      } else {
        return nil;
      }
    } else {
      s1:TransformScaledInverseGamma?;
      s2:DelayInverseGamma?;
      if (s1 <- this.σ2.graftScaledInverseGamma(σ2))? {
        μ1:Real[_] <- μ.value();
        D:Integer <- length(μ1);
        delay <- DelayMultivariateNormalInverseGamma(x, μ1,
            diagonal(s1!.a2, D), s1!.σ2);
      } else if Object(this.σ2) == σ2 && (s2 <- this.σ2.graftInverseGamma())? {
        μ1:Real[_] <- μ.value();
        D:Integer <- length(μ1);
        delay <- DelayMultivariateNormalInverseGamma(x, μ1, identity(D), s2!);
      }
      return DelayMultivariateNormalInverseGamma?(delay);
    }
  }
}

/**
 * Create multivariate Gaussian distribution.
 */
function Gaussian(μ:Expression<Real[_]>, σ2:Expression<Real>) ->
    MultivariateIndependentGaussian {
  m:MultivariateIndependentGaussian(μ, σ2);
  return m;
}

/**
 * Create multivariate Gaussian distribution.
 */
function Gaussian(μ:Expression<Real[_]>, σ2:Real) ->
    MultivariateIndependentGaussian {
  return Gaussian(μ, Boxed(σ2));
}

/**
 * Create multivariate Gaussian distribution.
 */
function Gaussian(μ:Real[_], σ2:Expression<Real>) ->
    MultivariateIndependentGaussian {
  return Gaussian(Boxed(μ), σ2);
}

/**
 * Create multivariate Gaussian distribution.
 */
function Gaussian(μ:Real[_], σ2:Real) -> MultivariateIndependentGaussian {
  return Gaussian(Boxed(μ), Boxed(σ2));
}
