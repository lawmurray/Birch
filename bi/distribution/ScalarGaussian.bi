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
    r:Distribution<Real>?;
    
    /* match a template */
    if (s1 <- σ2.graftInverseGamma())? {
      r <- NormalInverseGamma(μ, τ2, s1!);
    } else if (s1 <- τ2.graftInverseGamma())? {
      r <- NormalInverseGamma(μ, σ2, s1!);
    }
    
    /* finalize, and if not valid, use default template */
    if !r? || !r!.graftFinalize() {
      r <- GraftedGaussian(μ, σ2*τ2);
      r!.graftFinalize();
    }
    return r!;
  }

  function graftGaussian() -> Gaussian? {
    prune();
    auto r <- GraftedGaussian(μ, σ2*τ2);
    r!.graftFinalize();
    return r;
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
