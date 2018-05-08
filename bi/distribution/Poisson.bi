/**
 * Poisson distribution.
 */
class Poisson(λ:Expression<Real>) < Distribution<Integer> {
  /**
   * Rate.
   */
  λ:Expression<Real> <- λ;

  function graft() -> DelayValue<Integer> {
    m:DelayGamma?;
    if (m <- λ.graftGamma())? {
      return DelayGammaPoisson(m!);
    } else {
      return DelayPoisson(λ);
    }
  }

  function graftDiscrete() -> DelayValue<Integer>? {
    return DelayPoisson(λ);
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
