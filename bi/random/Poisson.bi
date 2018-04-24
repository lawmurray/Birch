/**
 * Poisson distribution.
 */
class Poisson(λ:Expression<Real>) < Random<Integer> {
  /**
   * Rate.
   */
  λ:Expression<Real> <- λ;

  function graft() {
    if (λ.isGamma()) {
      m:DelayGammaPoisson(this, λ.getGamma());
      m.graft();
      delay <- m;
    } else {
      m:DelayPoisson(this, λ.value());
      m.graft();
      delay <- m;
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
  return Poisson(Literal(λ));
}
