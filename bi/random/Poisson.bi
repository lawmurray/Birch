/**
 * Poisson distribution.
 */
class Poisson(λ:Expression<Real>) < Random<Integer> {
  /**
   * Rate.
   */
  λ:Expression<Real> <- λ;

  function doGraft() -> DelayValue<Integer>? {
    m:DelayGamma?;
    if (m <- λ.graftGamma())? {
      return DelayGammaPoisson(this, m!);
    } else {
      return DelayPoisson(this, λ.value());
    }
  }
}

/**
 * Create Poisson distribution.
 */
function Poisson(λ:Expression<Real>) -> Poisson {
  m:Poisson(λ);
  return m;
}

/**
 * Create Poisson distribution.
 */
function Poisson(λ:Real) -> Poisson {
  return Poisson(Boxed(λ));
}
