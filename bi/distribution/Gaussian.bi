/*
 * ed Gaussian random variate.
 */
class Gaussian(future:Real?, futureUpdate:Boolean, μ:Expression<Real>,
    σ2:Expression<Real>) < DelayMove<Distribution<Real>>(future, futureUpdate) {
  /**
   * Mean.
   */
  μ:Expression<Real> <- μ;

  /**
   * Precision.
   */
  λ:Expression<Real> <- 1.0/σ2;

  function simulate() -> Real {
    return simulate_gaussian(μ.value(), 1.0/λ.value());
  }

  function simulatePilot() -> Real {
    return simulate_gaussian(μ.pilot(), 1.0/λ.pilot());
  }

  function simulatePropose() -> Real {
    return simulate_gaussian(μ.propose(), 1.0/λ.propose());
  }
  
  function logpdf(x:Real) -> Real {
    return logpdf_gaussian(x, μ.value(), 1.0/λ.value());
  }

  function lazy(x:Expression<Real>) -> Expression<Real> {
    return lazy_gaussian(x, μ, 1.0/λ);
  }
  
  function cdf(x:Real) -> Real? {
    return cdf_gaussian(x, μ.value(), 1.0/λ.value());
  }

  function quantile(p:Real) -> Real? {
    return quantile_gaussian(p, μ.value(), 1.0/λ.value());
  }

  function graft() {
    if delay? {
      delay!.prune();
    } else {
      m1:TransformLinear<NormalInverseGamma>?;
      m3:NormalInverseGamma?;
      m4:TransformLinear<Gaussian>?;
      m5:TransformDot<MultivariateGaussian>?;
      m6:Gaussian?;
      s2:InverseGamma?;

      if (m1 <- μ.graftLinearNormalInverseGamma())? && m1!.x.σ2 == σ2.get() {
        delay <- LinearNormalInverseGammaGaussian(future, futureUpdate, m1!.a, m1!.x, m1!.c);
      } else if (m3 <- μ.graftNormalInverseGamma())? && m3!.σ2 == σ2.get() {
        delay <- NormalInverseGammaGaussian(future, futureUpdate, m3!);
      } else if (m4 <- μ.graftLinearGaussian())? {
        delay <- LinearGaussianGaussian(future, futureUpdate, m4!.a, m4!.x, m4!.c, σ2);
      } else if (m5 <- μ.graftDotGaussian())? {
        delay <- LinearMultivariateGaussianGaussian(future, futureUpdate, m5!.a, m5!.x, m5!.c, σ2);
      } else if (m6 <- μ.graftGaussian())? {
        delay <- GaussianGaussian(future, futureUpdate, m6!, σ2);
      } else if (s2 <- σ2.graftInverseGamma())? {
        delay <- NormalInverseGamma(future, futureUpdate, μ, 1.0, s2!);
      } else {
        delay <- Gaussian(future, futureUpdate, μ, σ2);
      }
    }
  }

  function graftGaussian() -> Gaussian? {
    if delay? {
      delay!.prune();
    } else {
      m1:TransformLinear<Gaussian>?;
      m2:TransformDot<MultivariateGaussian>?;
      m3:Gaussian?;
      if (m1 <- μ.graftLinearGaussian())? {
        delay <- LinearGaussianGaussian(future, futureUpdate, m1!.a, m1!.x, m1!.c, σ2);
      } else if (m2 <- μ.graftDotGaussian())? {
        delay <- LinearMultivariateGaussianGaussian(future, futureUpdate, m2!.a, m2!.x, m2!.c, σ2);
      } else if (m3 <- μ.graftGaussian())? {
        delay <- GaussianGaussian(future, futureUpdate, m3!, σ2);
      } else {
        delay <- Gaussian(future, futureUpdate, μ, σ2);
      }
    }
    return Gaussian?(delay);
  }

  function graftNormalInverseGamma() -> NormalInverseGamma? {
    if delay? {
      delay!.prune();
    } else {
      s1:InverseGamma?;
      if (s1 <- σ2.graftInverseGamma())? {
        delay <- NormalInverseGamma(future, futureUpdate, μ, 1.0, s1!);
      }
    }
    return NormalInverseGamma?(delay);
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Gaussian");
    buffer.set("μ", μ.value());
    buffer.set("σ2", 1.0/λ.value());
  }
}

function Gaussian(future:Real?, futureUpdate:Boolean,
    μ:Expression<Real>, σ2:Expression<Real>) -> Gaussian {
  o:Gaussian(future, futureUpdate, μ.graft(), σ2.graft());
  return o;
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
