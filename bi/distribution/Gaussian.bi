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

  function simulate() -> Real {
    return simulate_gaussian(μ.value(), σ2.value());
  }
  
  function logpdf(x:Real) -> Real {
    return logpdf_gaussian(x, μ.value(), σ2.value());
  }

  function simulateLazy() -> Real? {
    return simulate_gaussian(μ.pilot(), σ2.pilot());
  }

  function logpdfLazy(x:Expression<Real>) -> Expression<Real>? {
    return logpdf_lazy_gaussian(x, μ, σ2);
  }
  
  function cdf(x:Real) -> Real? {
    return cdf_gaussian(x, μ.value(), σ2.value());
  }

  function quantile(P:Real) -> Real? {
    return quantile_gaussian(P, μ.value(), σ2.value());
  }

  function graft() -> Distribution<Real> {
    prune();
    m1:TransformLinear<NormalInverseGamma>?;
    m3:NormalInverseGamma?;
    m4:TransformLinear<Gaussian>?;
    m5:TransformDot<MultivariateGaussian>?;
    m6:Gaussian?;
    s2:InverseGamma?;
    r:Distribution<Real>?;

    /* match a template */
    auto compare <- σ2.distribution();
    if compare? && (m1 <- μ.graftLinearNormalInverseGamma(compare!))? {
      r <- LinearNormalInverseGammaGaussian(m1!.a, m1!.x, m1!.c);
    } else if compare? && (m3 <- μ.graftNormalInverseGamma(compare!))? {
      r <- NormalInverseGammaGaussian(m3!);
    } else if (m4 <- μ.graftLinearGaussian())? {
      r <- LinearGaussianGaussian(m4!.a, m4!.x, m4!.c, σ2);
    } else if (m5 <- μ.graftDotGaussian())? {
      r <- LinearMultivariateGaussianGaussian(m5!.a, m5!.x, m5!.c, σ2);
    } else if (m6 <- μ.graftGaussian())? {
      r <- GaussianGaussian(m6!, σ2);
    } else if (s2 <- σ2.graftInverseGamma())? {
      r <- NormalInverseGamma(μ, Boxed(1.0), s2!);
    }
    
    /* finalize, and if not valid, use default template */
    if !r? || !r!.graftFinalize() {
      r <- GraftedGaussian(μ, σ2);
      r!.graftFinalize();
    }
    return r!;
  }

  function graftGaussian() -> Gaussian? {
    prune();
    m1:TransformLinear<Gaussian>?;
    m2:TransformDot<MultivariateGaussian>?;
    m3:Gaussian?;
    r:Gaussian?;
    
    /* match a template */
    if (m1 <- μ.graftLinearGaussian())? {
      r <- LinearGaussianGaussian(m1!.a, m1!.x, m1!.c, σ2);
    } else if (m2 <- μ.graftDotGaussian())? {
      r <- LinearMultivariateGaussianGaussian(m2!.a, m2!.x, m2!.c, σ2);
    } else if (m3 <- μ.graftGaussian())? {
      r <- GaussianGaussian(m3!, σ2);
    }
    
    /* finalize, and if not valid, use default template */
    if !r? || !r!.graftFinalize() {
      r <- GraftedGaussian(μ, σ2);
      r!.graftFinalize();
    }
    return r;
  }

  function graftNormalInverseGamma(compare:Distribution<Real>) ->
      NormalInverseGamma? {
    prune();
    s1:InverseGamma?;
    r:NormalInverseGamma?;
    
    /* match a template */
    if (s1 <- σ2.graftInverseGamma())? && s1! == compare {
      r <- NormalInverseGamma(μ, Boxed(1.0), s1!);
    }
    
    /* finalize, and if not valid, return nil */
    if !r? || !r!.graftFinalize() {
      r <- nil;
    }
    return r;
  }

  function graftFinalize() -> Boolean {
    assert false;  // should have been replaced during graft
    return false;
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Gaussian");
    buffer.set("μ", μ.value());
    buffer.set("σ2", σ2.value());
  }
}

/**
 * Create Gaussian distribution.
 */
function Gaussian(μ:Expression<Real>, σ2:Expression<Real>) -> Gaussian {
  o:Gaussian(μ, σ2);
  return o;
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
