/**
 * Poisson distribution.
 */
class Poisson(λ:Expression<Real>) < Random<Integer> {
  /**
   * Rate.
   */
  λ:Expression<Real> <- λ;

  function doParent() -> Delay? {
    if (λ.isGamma()) {
      return λ;
    } else {
      return nil;
    }
  }

  function doSimulate() -> Integer {
    if (λ.isGamma()) {
      k:Real;
      θ:Real;
      (k, θ) <- λ.getGamma();
      return simulate_gamma_poisson(k, θ);
    } else {
      return simulate_poisson(λ.value());
    }
  }
  
  function doObserve(x:Integer) -> Real {
    if (λ.isGamma()) {
      k:Real;
      θ:Real;
      (k, θ) <- λ.getGamma();
      return observe_gamma_poisson(x, k, θ);
    } else {
      return observe_poisson(x, λ.value());
    }
  }
}

/**
 * Create Poisson distribution.
 */
function Poisson(λ:Expression<Real>) -> Poisson {
  m:Poisson(λ);
  m.initialize();
  return m;
}

/**
 * Create Poisson distribution.
 */
function Poisson(λ:Real) -> Poisson {
  return Poisson(Literal(λ));
}
