/**
 * Gaussian distribution where the variance is given as a product of two
 * scalars.
 */
final class ScalarGaussian(μ:Expression<Real>, σ2:Expression<Real>,
    a:Expression<Real>) < Distribution<Real> {
  /**
   * Mean.
   */
  μ:Expression<Real> <- μ;
  
  /**
   * Covariance.
   */
  σ2:Expression<Real> <- σ2;

  /**
   * Covariance scale.
   */
  a:Expression<Real> <- a;
  
  function valueForward() -> Real {
    assert !delay?;
    return simulate_gaussian(μ, σ2*a);
  }

  function observeForward(x:Real) -> Real {
    assert !delay?;
    return logpdf_gaussian(x, μ, σ2*a);
  }
  
  function graft(force:Boolean) {
    if delay? {
      delay!.prune();
    } else if force {
      delay <- DelayGaussian(future, futureUpdate, μ, σ2*a);
    }
  }

  function graftGaussian() -> DelayGaussian? {
    if delay? {
      delay!.prune();
    } else {
      delay <- DelayGaussian(future, futureUpdate, μ, σ2*a);
    }
    return DelayGaussian?(delay);
  }
}

/**
 * Create Gaussian distribution where the covariance is given
 * as a matrix multiplied by a scalar.
 */
function Gaussian(μ:Expression<Real>, σ2:Expression<Real>,
    a:Expression<Real>) -> ScalarGaussian {
  m:ScalarGaussian(μ, σ2, a);
  return m;
}

/**
 * Create matrix Gaussian distribution.
 */
function Gaussian(μ:Expression<Real>, σ2:Expression<Real>,
    a:Real) -> ScalarGaussian {
  return Gaussian(μ, σ2, Boxed(a));
}

/**
 * Create μθλτι Gaussian distribution.
 */
function Gaussian(μ:Expression<Real>, σ2:Real,
    a:Expression<Real>) -> ScalarGaussian {
  return Gaussian(μ, Boxed(σ2), a);
}

/**
 * Create matrix Gaussian distribution.
 */
function Gaussian(μ:Expression<Real>, σ2:Real, a:Real) ->
      ScalarGaussian {
  return Gaussian(μ, Boxed(σ2), Boxed(a));
}

/**
 * Create matrix Gaussian distribution.
 */
function Gaussian(μ:Real, σ2:Expression<Real>,
    a:Expression<Real>) -> ScalarGaussian {
  return Gaussian(Boxed(μ), σ2, a);
}

/**
 * Create matrix Gaussian distribution.
 */
function Gaussian(μ:Real, σ2:Expression<Real>, a:Real) ->
    ScalarGaussian {
  return Gaussian(Boxed(μ), σ2, Boxed(a));
}

/**
 * Create matrix Gaussian distribution.
 */
function Gaussian(μ:Real, σ2:Real, a:Real) -> ScalarGaussian {
  return Gaussian(Boxed(μ), Boxed(σ2), Boxed(a));
}
