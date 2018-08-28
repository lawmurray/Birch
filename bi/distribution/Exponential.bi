/**
 * Exponential distribution.
 */
class Exponential(λ:Expression<Real>) < Distribution<Real> {
  /**
   * Rate.
   */
  λ:Expression<Real> <- λ;

  function graft() {
    if delay? {
      delay!.prune();
    } else {
      m:DelayGamma?;
      if (m <- λ.graftGamma())? {
        delay <- DelayGammaExponential(x, m!);
      } else {
        delay <- DelayExponential(x, λ);
      }
    }
  }
}

/**
 * Create Exponential distribution.
 */
function Exponential(λ:Expression<Real>) -> Exponential {
  m:Exponential(λ);
  return m;
}

/**
 * Create Exponential distribution.
 */
function Exponential(λ:Real) -> Exponential {
  return Exponential(Boxed(λ));
}
