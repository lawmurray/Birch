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
      m1:TransformScaledGamma?;
      m2:DelayGamma?;

      if (m1 <- λ.graftScaledGamma())? {
        delay <- DelayScaledGammaExponential(x, m1!.a, m1!.x);
      } else if (m2 <- λ.graftGamma())? {
        delay <- DelayGammaExponential(x, m2!);
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
