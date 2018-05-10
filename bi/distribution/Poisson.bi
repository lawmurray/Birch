/**
 * Poisson distribution.
 */
class Poisson(λ:Expression<Real>) < Distribution<Integer> {
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
        delay <- DelayGammaPoisson(x, m!);
      } else {
        delay <- DelayPoisson(x, λ);
      }
    }
  }

  function graftDiscrete() -> DelayValue<Integer>? {
    if delay? {
      delay!.prune();
    } else {
      delay <- DelayPoisson(x, λ);
    }
    return DelayValue<Integer>?(delay);
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
