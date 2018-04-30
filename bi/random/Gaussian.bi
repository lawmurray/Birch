/**
 * Gaussian distribution.
 */
class Gaussian(μ:Expression<Real>, σ2:Expression<Real>) < Random<Real> {
  /**
   * Mean.
   */
  μ:Expression<Real> <- μ;
  
  /**
   * Variance.
   */
  σ2:Expression<Real> <- σ2;

  function doGraft() -> DelayValue<Real>? {
    s2:DelayInverseGamma?;
    m1:TransformAffineNormalInverseGamma?;
    m2:DelayNormalInverseGamma?;
    m3:TransformAffineGaussian?;
    m4:DelayGaussian?;
        
    if (s2 <- σ2.graftInverseGamma())? {
      if (m1 <- μ.graftAffineNormalInverseGamma(σ2))? {
        return DelayAffineNormalInverseGammaGaussian(this, m1!.a, m1!.x, m1!.c);
      } else if (m2 <- μ.graftNormalInverseGamma(σ2))? {
        return DelayNormalInverseGammaGaussian(this, m2!);
      } else {
        return DelayInverseGammaGaussian(this, μ, s2!);
      }
    } else if (m3 <- μ.graftAffineGaussian())? {
      return DelayAffineGaussianGaussian(this, m3!.a, m3!.x, m3!.c, σ2);
    } else if (m4 <- μ.graftGaussian())? {
      return DelayGaussianGaussian(this, m4!, σ2);
    } else {
      return DelayGaussian(this, μ, σ2);
    }
  }

  function doGraftGaussian() -> DelayGaussian? {
    return DelayGaussian(this, μ, σ2);
  }

  function doGraftNormalInverseGamma(σ2:Expression<Real>) ->
      DelayNormalInverseGamma? {
    s2:TransformScaledInverseGamma?;
    t2:DelayInverseGamma?;
    if (s2 <- this.σ2.graftScaledInverseGamma(σ2))? {
      return DelayNormalInverseGamma(this, μ, s2!.a2, s2!.σ2);
    } else if Object(this.σ2) == σ2 && (t2 <- this.σ2.graftInverseGamma())? {
      return DelayNormalInverseGamma(this, μ, 1.0, t2!);
    } else {
      return nil;
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
