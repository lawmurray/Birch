/*
 * ed Gaussian random variate.
 */
class Gaussian(future:Real?, futureUpdate:Boolean, μ:Expression<Real>,
    σ2:Expression<Real>) < Moveable<Real>(future, futureUpdate) {
  /**
   * Mean.
   */
  μ:Expression<Real> <- μ;

  /**
   * Variance.
   */
  σ2:Expression<Real> <- σ2;

  function simulate() -> Real {
    return simulate_gaussian(μ.pilot(), σ2.pilot());
  }
  
  function logpdf(x:Real) -> Real {
    return logpdf_gaussian(x, μ.pilot(), σ2.pilot());
  }
  
  function cdf(x:Real) -> Real? {
    return cdf_gaussian(x, μ.pilot(), σ2.pilot());
  }

  function quantile(P:Real) -> Real? {
    return quantile_gaussian(P, μ.pilot(), σ2.pilot());
  }

  function lazy(x:Expression<Real>) -> Expression<Real>? {
    return lazy_gaussian(x, μ, σ2);
  }

  function graft() -> Distribution<Real> {
    prune();
    m1:TransformLinear<NormalInverseGamma>?;
    m3:NormalInverseGamma?;
    m4:TransformLinear<Gaussian>?;
    m5:TransformDot<MultivariateGaussian>?;
    m6:Gaussian?;
    s2:InverseGamma?;

    if (m1 <- μ.graftLinearNormalInverseGamma())? && m1!.x.σ2 == σ2.distribution() {
      return LinearNormalInverseGammaGaussian(future, futureUpdate, m1!.a, m1!.x, m1!.c);
    } else if (m3 <- μ.graftNormalInverseGamma())? && m3!.σ2 == σ2.distribution() {
      return NormalInverseGammaGaussian(future, futureUpdate, m3!);
    } else if (m4 <- μ.graftLinearGaussian())? {
      return LinearGaussianGaussian(future, futureUpdate, m4!.a, m4!.x, m4!.c, σ2);
    } else if (m5 <- μ.graftDotGaussian())? {
      return LinearMultivariateGaussianGaussian(future, futureUpdate, m5!.a, m5!.x, m5!.c, σ2);
    } else if (m6 <- μ.graftGaussian())? {
      return GaussianGaussian(future, futureUpdate, m6!, σ2);
    } else if (s2 <- σ2.graftInverseGamma())? {
      return NormalInverseGamma(future, futureUpdate, μ, 1.0, s2!);
    } else {
      return this;
    }
  }

  function graftGaussian() -> Gaussian? {
    prune();
    m1:TransformLinear<Gaussian>?;
    m2:TransformDot<MultivariateGaussian>?;
    m3:Gaussian?;
    if (m1 <- μ.graftLinearGaussian())? {
      return LinearGaussianGaussian(future, futureUpdate, m1!.a, m1!.x, m1!.c, σ2);
    } else if (m2 <- μ.graftDotGaussian())? {
      return LinearMultivariateGaussianGaussian(future, futureUpdate, m2!.a, m2!.x, m2!.c, σ2);
    } else if (m3 <- μ.graftGaussian())? {
      return GaussianGaussian(future, futureUpdate, m3!, σ2);
    } else {
      return this;
    }
  }

  function graftNormalInverseGamma() -> NormalInverseGamma? {
    prune();
    s1:InverseGamma?;
    if (s1 <- σ2.graftInverseGamma())? {
      return NormalInverseGamma(future, futureUpdate, μ, 1.0, s1!);
    }
    return nil;
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Gaussian");
    buffer.set("μ", μ.value());
    buffer.set("σ2", σ2.value());
  }
}

function Gaussian(future:Real?, futureUpdate:Boolean,
    μ:Expression<Real>, σ2:Expression<Real>) -> Gaussian {
  o:Gaussian(future, futureUpdate, μ, σ2);
  return o;
}

/**
 * Create Gaussian distribution.
 */
function Gaussian(μ:Expression<Real>, σ2:Expression<Real>) -> Gaussian {
  m:Gaussian(nil, true, μ, σ2);
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
