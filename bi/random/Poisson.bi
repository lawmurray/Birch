/**
 * Poisson distribution.
 */
class Poisson(λ:Expression<Real>) < Random<Integer> {
  /**
   * Rate.
   */
  λ:Expression<Real> <- λ;

  function graft() -> Delay? {
    if (delay?) {
      return delay;
    } else {
      m:DelayGamma?;
      if (m <- λ.graftGamma())? {
        return DelayGammaPoisson(this, m!);
      } else {
        return DelayPoisson(this, λ.value());
      }
    }
  }

  function graftDiscrete() -> DelayValue<Integer>? {
    if (delay?) {
      return DelayValue<Integer>?(delay);
    } else {
      return DelayPoisson(this, λ);
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
