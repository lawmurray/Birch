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

  function simulate() -> Real {
    return simulate_gaussian(μ.value(), σ2.value()*τ2.value());
  }
  
  function logpdf(x:Real) -> Real {
    return logpdf_gaussian(x, μ.value(), σ2.value()*τ2.value());
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
    if (s1 <- σ2.graftInverseGamma())? {
      return NormalInverseGamma(μ, τ2, s1!);
    } else if (s1 <- τ2.graftInverseGamma())? {
      return NormalInverseGamma(μ, σ2, s1!);
    } else {
      return Gaussian(μ, σ2*τ2);
    }
  }

  function graftGaussian() -> Gaussian? {
    prune();
    return Gaussian(μ, σ2*τ2);
  }

  function graftNormalInverseGamma() -> NormalInverseGamma? {
    prune();
    s1:InverseGamma?;
    if (s1 <- σ2.graftInverseGamma())? {
      return NormalInverseGamma(μ, τ2, s1!);
    } else if (s1 <- τ2.graftInverseGamma())? {
      return NormalInverseGamma(μ, σ2, s1!);
    }
    return nil;
  }
}

/**
 * Create Gaussian distribution where the variance is given as a product of
 * two scalars. This is usually used for establishing a normal-inverse-gamma
 * distribution, where one of the arguments is inverse-gamma distributed.
 */
function Gaussian(μ:Expression<Real>, σ2:Expression<Real>,
    τ2:Expression<Real>) -> ScalarGaussian {
  m:ScalarGaussian(μ, σ2, τ2);
  return m;
}

/**
 * Create Gaussian distribution where the variance is given as a product of
 * two scalars. This is usually used for establishing a normal-inverse-gamma
 * distribution, where one of the arguments is inverse-gamma distributed.
 */
function Gaussian(μ:Expression<Real>, σ2:Expression<Real>,
    τ2:Real) -> ScalarGaussian {
  return Gaussian(μ, σ2, Boxed(τ2));
}

/**
 * Create Gaussian distribution where the variance is given as a product of
 * two scalars. This is usually used for establishing a normal-inverse-gamma
 * distribution, where one of the arguments is inverse-gamma distributed.
 */
function Gaussian(μ:Expression<Real>, σ2:Real, τ2:Expression<Real>) ->
    ScalarGaussian {
  return Gaussian(μ, Boxed(σ2), τ2);
}

/**
 * Create Gaussian distribution where the variance is given as a product of
 * two scalars. This is usually used for establishing a normal-inverse-gamma
 * distribution, where one of the arguments is inverse-gamma distributed.
 */
function Gaussian(μ:Expression<Real>, σ2:Real, τ2:Real) ->
      ScalarGaussian {
  return Gaussian(μ, Boxed(σ2), Boxed(τ2));
}

/**
 * Create Gaussian distribution where the variance is given as a product of
 * two scalars. This is usually used for establishing a normal-inverse-gamma
 * distribution, where one of the arguments is inverse-gamma distributed.
 */
function Gaussian(μ:Real, σ2:Expression<Real>, τ2:Expression<Real>) ->
    ScalarGaussian {
  return Gaussian(Boxed(μ), σ2, τ2);
}

/**
 * Create Gaussian distribution where the variance is given as a product of
 * two scalars. This is usually used for establishing a normal-inverse-gamma
 * distribution, where one of the arguments is inverse-gamma distributed.
 */
function Gaussian(μ:Real, σ2:Expression<Real>, τ2:Real) ->
    ScalarGaussian {
  return Gaussian(Boxed(μ), σ2, Boxed(τ2));
}

/**
 * Create Gaussian distribution where the variance is given as a product of
 * two scalars. This is usually used for establishing a normal-inverse-gamma
 * distribution, where one of the arguments is inverse-gamma distributed.
 */
function Gaussian(μ:Real, σ2:Real, τ2:Expression<Real>) ->
    ScalarGaussian {
  return Gaussian(Boxed(μ), Boxed(σ2), τ2);
}

/**
 * Create Gaussian distribution where the variance is given as a product of
 * two scalars. This is usually used for establishing a normal-inverse-gamma
 * distribution, where one of the arguments is inverse-gamma distributed.
 */
function Gaussian(μ:Real, σ2:Real, τ2:Real) -> ScalarGaussian {
  return Gaussian(Boxed(μ), Boxed(σ2), Boxed(τ2));
}
