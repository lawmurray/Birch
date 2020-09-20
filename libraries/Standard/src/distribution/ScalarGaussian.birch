/**
 * Gaussian distribution where the variance is given as a product of two
 * scalars.
 */
final class ScalarGaussian(μ:Expression<Real>, σ2:Expression<Real>,
    τ2:Expression<Real>) < Distribution<Real> {
  /**
   * Mean.
   */
  μ:Expression<Real> <- μ;
  
  /**
   * Variance operand.
   */
  σ2:Expression<Real> <- σ2;

  /**
   * Variance operand.
   */
  τ2:Expression<Real> <- τ2;

  function supportsLazy() -> Boolean {
    return true;
  }

  function simulate() -> Real {
    return simulate_gaussian(μ.value(), σ2.value()*τ2.value());
  }

  function simulateLazy() -> Real? {
    return simulate_gaussian(μ.get(), σ2.get()*τ2.get());
  }
  
  function logpdf(x:Real) -> Real {
    return logpdf_gaussian(x, μ.value(), σ2.value()*τ2.value());
  }

  function logpdfLazy(x:Expression<Real>) -> Expression<Real>? {
    return logpdf_lazy_gaussian(x, μ, σ2*τ2);
  }

  function cdf(x:Real) -> Real? {
    return cdf_gaussian(x, μ.value(), σ2.value()*τ2.value());
  }

  function quantile(P:Real) -> Real? {
    return quantile_gaussian(P, μ.value(), σ2.value()*τ2.value());
  }
  
  function graft() -> Distribution<Real> {
    prune();
    s1:InverseGamma?;
    r:Distribution<Real> <- this;
    
    /* match a template */
    if (s1 <- σ2.graftInverseGamma())? {
      r <- NormalInverseGamma(μ, τ2, s1!);
    } else if (s1 <- τ2.graftInverseGamma())? {
      r <- NormalInverseGamma(μ, σ2, s1!);
    }
    
    return r;
  }

  function graftGaussian() -> Gaussian? {
    prune();
    return Gaussian(μ, σ2*τ2);
  }

  function graftNormalInverseGamma(compare:Distribution<Real>) ->
      NormalInverseGamma? {
    prune();
    s1:InverseGamma?;
    r:NormalInverseGamma?;
    
    /* match a template */
    if (s1 <- σ2.graftInverseGamma())? && s1! == compare {
      r <- NormalInverseGamma(μ, τ2, s1!);
    } else if (s1 <- τ2.graftInverseGamma())? && s1! == compare {
      r <- NormalInverseGamma(μ, σ2, s1!);
    }

    return r;
  }
}

/**
 * Create Gaussian distribution where the variance is given as a product of
 * two scalars. This is usually used for establishing a normal-inverse-gamma
 * distribution, where one of the arguments is inverse-gamma distributed.
 */
function Gaussian(μ:Expression<Real>, σ2:Expression<Real>,
    τ2:Expression<Real>) -> ScalarGaussian {
  return construct<ScalarGaussian>(μ, σ2, τ2);
}

/**
 * Create Gaussian distribution where the variance is given as a product of
 * two scalars. This is usually used for establishing a normal-inverse-gamma
 * distribution, where one of the arguments is inverse-gamma distributed.
 */
function Gaussian(μ:Expression<Real>, σ2:Expression<Real>,
    τ2:Real) -> ScalarGaussian {
  return Gaussian(μ, σ2, box(τ2));
}

/**
 * Create Gaussian distribution where the variance is given as a product of
 * two scalars. This is usually used for establishing a normal-inverse-gamma
 * distribution, where one of the arguments is inverse-gamma distributed.
 */
function Gaussian(μ:Expression<Real>, σ2:Real, τ2:Expression<Real>) ->
    ScalarGaussian {
  return Gaussian(μ, box(σ2), τ2);
}

/**
 * Create Gaussian distribution where the variance is given as a product of
 * two scalars. This is usually used for establishing a normal-inverse-gamma
 * distribution, where one of the arguments is inverse-gamma distributed.
 */
function Gaussian(μ:Expression<Real>, σ2:Real, τ2:Real) ->
      ScalarGaussian {
  return Gaussian(μ, box(σ2), box(τ2));
}

/**
 * Create Gaussian distribution where the variance is given as a product of
 * two scalars. This is usually used for establishing a normal-inverse-gamma
 * distribution, where one of the arguments is inverse-gamma distributed.
 */
function Gaussian(μ:Real, σ2:Expression<Real>, τ2:Expression<Real>) ->
    ScalarGaussian {
  return Gaussian(box(μ), σ2, τ2);
}

/**
 * Create Gaussian distribution where the variance is given as a product of
 * two scalars. This is usually used for establishing a normal-inverse-gamma
 * distribution, where one of the arguments is inverse-gamma distributed.
 */
function Gaussian(μ:Real, σ2:Expression<Real>, τ2:Real) ->
    ScalarGaussian {
  return Gaussian(box(μ), σ2, box(τ2));
}

/**
 * Create Gaussian distribution where the variance is given as a product of
 * two scalars. This is usually used for establishing a normal-inverse-gamma
 * distribution, where one of the arguments is inverse-gamma distributed.
 */
function Gaussian(μ:Real, σ2:Real, τ2:Expression<Real>) ->
    ScalarGaussian {
  return Gaussian(box(μ), box(σ2), τ2);
}

/**
 * Create Gaussian distribution where the variance is given as a product of
 * two scalars. This is usually used for establishing a normal-inverse-gamma
 * distribution, where one of the arguments is inverse-gamma distributed.
 */
function Gaussian(μ:Real, σ2:Real, τ2:Real) -> ScalarGaussian {
  return Gaussian(box(μ), box(σ2), box(τ2));
}
