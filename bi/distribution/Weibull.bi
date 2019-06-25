/**
 * Weibull distribution.
 */
final class Weibull(k:Expression<Real>, λ:Expression<Real>) < Distribution<Real> {
  /**
   * Shape.
   */
  k:Expression<Real> <- k;

  /**
   * Scale.
   */
  λ:Expression<Real> <- λ;

  function valueForward() -> Real {
    assert !delay?;
    return simulate_weibull(k, λ);
  }

  function observeForward(x:Real) -> Real {
    assert !delay?;
    return logpdf_weibull(x, k, λ);
  }

  function graft(force:Boolean) {
    if delay? {
      delay!.prune();
    } else if force {
      delay <- DelayWeibull(future, futureUpdate, k, λ);
    }
  }

  function write(buffer:Buffer) {
    if delay? {
      delay!.write(buffer);
    } else {
      buffer.set("class", "Weibull");
      buffer.set("k", k.value());
      buffer.set("λ", λ.value());
    }
  }
}

/**
 * Create Weibull distribution.
 */
function Weibull(k:Expression<Real>, λ:Expression<Real>) -> Weibull {
  m:Weibull(k, λ);
  return m;
}

/**
 * Create Weibull distribution.
 */
function Weibull(k:Expression<Real>, λ:Real) -> Weibull {
  return Weibull(k, Boxed(λ));
}

/**
 * Create Weibull distribution.
 */
function Weibull(k:Real, λ:Expression<Real>) -> Weibull {
  return Weibull(Boxed(k), λ);
}

/**
 * Create Weibull distribution.
 */
function Weibull(k:Real, λ:Real) -> Weibull {
  return Weibull(Boxed(k), Boxed(λ));
}
