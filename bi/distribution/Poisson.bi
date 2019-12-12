/**
 * Poisson distribution.
 */
final class Poisson(λ:Expression<Real>) < Distribution<Integer> {
  /**
   * Rate.
   */
  λ:Expression<Real> <- λ;

  function graft() {
    if delay? {
      delay!.prune();
    } else {
      m1:TransformLinear<DelayGamma>?;
      m2:DelayGamma?;
      
      if (m1 <- λ.graftScaledGamma())? {
        delay <- DelayScaledGammaPoisson(future, futureUpdate, m1!.a, m1!.x);
      } else if (m2 <- λ.graftGamma())? {
        delay <- DelayGammaPoisson(future, futureUpdate, m2!);
      } else {
        delay <- DelayPoisson(future, futureUpdate, λ);
      }
    }
  }

  function graftDiscrete() -> DelayDiscrete? {
    graft();
    return DelayDiscrete?(delay);
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
