/**
 * Gaussian distribution.
 */
class Gaussian(μ:Expression<Real>, σ2:Expression<Real>) < Distribution<Real> {
  /**
   * Mean.
   */
  μ:Expression<Real> <- μ;
  
  /**
   * Variance.
   */
  σ2:Expression<Real> <- σ2;

  function graft() {
    if delay? {
      delay!.prune();
    } else {
      s2:DelayInverseGamma?;
      m1:TransformLinearNormalInverseGamma?;
      m2:DelayNormalInverseGamma?;
      m3:TransformLinearGaussian?;
      m4:DelayGaussian?;

      if (s2 <- σ2.graftInverseGamma())? {
        if (m1 <- μ.graftLinearNormalInverseGamma(σ2))? {
          delay <- DelayLinearNormalInverseGammaGaussian(x, m1!.a, m1!.x,
              m1!.c);
        } else if (m2 <- μ.graftNormalInverseGamma(σ2))? {
          delay <- DelayNormalInverseGammaGaussian(x, m2!);
        } else {
          delay <- DelayInverseGammaGaussian(x, μ, s2!);
        }
      } else if (m3 <- μ.graftLinearGaussian())? {
        delay <- DelayLinearGaussianGaussian(x, m3!.a, m3!.x, m3!.c, σ2);
      } else if (m4 <- μ.graftGaussian())? {
        delay <- DelayGaussianGaussian(x, m4!, σ2);
      } else {
        delay <- DelayGaussian(x, μ, σ2);
      }
    }
  }

  function graftGaussian() -> DelayGaussian? {
    if delay? {
      delay!.prune();
    } else {
      m1:TransformLinearGaussian?;
      m2:DelayGaussian?;
      if (m1 <- μ.graftLinearGaussian())? {
        delay <- DelayLinearGaussianGaussian(x, m1!.a, m1!.x, m1!.c, σ2);
      } else if (m2 <- μ.graftGaussian())? {
        delay <- DelayGaussianGaussian(x, m2!, σ2);
      } else {
        delay <- DelayGaussian(x, μ, σ2);
      }
    }
    return DelayGaussian?(delay);
  }

  function graftNormalInverseGamma(σ2:Expression<Real>) ->
      DelayNormalInverseGamma? {
    if delay? {
      delay!.prune();
      
      m:DelayNormalInverseGamma?;
      s2:DelayInverseGamma?;
      if (m <- DelayNormalInverseGamma?(delay))? &&
          (s2 <- σ2.graftInverseGamma())? && m!.σ2! == s2! {
        return m;
      } else {
        return nil;
      }
    } else {
      s2:TransformScaledInverseGamma?;
      t2:DelayInverseGamma?;
      if (s2 <- this.σ2.graftScaledInverseGamma(σ2))? {
        delay <- DelayNormalInverseGamma(x, μ, s2!.a2, s2!.σ2);
      } else if Object(this.σ2) == σ2 && (t2 <- this.σ2.graftInverseGamma())? {
        delay <- DelayNormalInverseGamma(x, μ, 1.0, t2!);
      }
      return DelayNormalInverseGamma?(delay);
    }
  }
}

/**
 * Create Gaussian distribution.
 */
function Gaussian(μ:Expression<Real>, σ2:Expression<Real>) -> Gaussian {
  m:Gaussian(μ, σ2);
  return m;
}

/**
 * Create Gaussian distribution.
 */
function Gaussian(μ:Expression<Real>, σ2:Real) -> Gaussian {
  return Gaussian(μ, Boxed(σ2));
}

/**
 * Create Gaussian distribution.
 */
function Gaussian(μ:Real, σ2:Expression<Real>) -> Gaussian {
  return Gaussian(Boxed(μ), σ2);
}

/**
 * Create Gaussian distribution.
 */
function Gaussian(μ:Real, σ2:Real) -> Gaussian {
  return Gaussian(Boxed(μ), Boxed(σ2));
}
