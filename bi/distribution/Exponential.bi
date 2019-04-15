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
      m:DelayGamma?;
      if (m <- λ.graftGamma())? {
        delay <- DelayGammaExponential(x, m!);
      } else {
        delay <- DelayExponential(x, λ);
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
