/**
 * Exponential distribution.
 */
final class Exponential(λ:Expression<Real>) < Distribution<Real> {
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
        delay <- DelayScaledGammaExponential(future, futureUpdate, m1!.a, m1!.x);
      } else if (m2 <- λ.graftGamma())? {
        delay <- DelayGammaExponential(future, futureUpdate, m2!);
      } else {
        delay <- DelayExponential(future, futureUpdate, λ);
      }
    }
  }

  function write(buffer:Buffer) {
    if delay? {
      delay!.write(buffer);
    } else {
      buffer.set("class", "Exponential");
      buffer.set("λ", λ.value());
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
