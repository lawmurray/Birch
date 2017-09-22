/**
 * Poisson with conjugate prior on rate. When the rate is known, this is
 * simply a Poisson distribution. When the rate is gamma distributed, this is
 * a negative binomial distribution.
 */
class GammaPoisson < DelayInteger {
  /**
   * Rate.
   */
  λ:Gamma;

  /*
   * Parameters of negative binomial when λ is marginalized out. See
   * NegativeBinomial.
   */
  k:Integer;
  ρ:Real;

  function initialize(λ:Gamma) {
    super.initialize(λ);
    this.λ <- λ;
  }
  
  function doMarginalize() {
    this.k <- λ.k;
    this.ρ <- 1.0/(λ.θ + 1.0);
  }
  
  function doCondition() {
    λ.update(λ.k + x, λ.θ/(λ.θ + 1.0));
  }

  function doRealize() {
    if (λ.isRealized()) {
      if (isMissing()) {
        set(simulate_poisson(λ));
      } else {
        setWeight(observe_poisson(x, λ));
      }
    } else {
      if (isMissing()) {
        set(simulate_negative_binomial(k, ρ));
      } else {
        setWeight(observe_negative_binomial(x, k, ρ));
      }
    }
  }

  function tildeLeft() -> GammaPoisson {
    simulate();
    return this;
  }
  
  function tildeRight(left:GammaPoisson) -> GammaPoisson {
    set(left.value());
    observe();
    return this;
  }
}

/**
 * Create gamma-Poisson distribution.
 */
function Poisson(λ:Gamma) -> GammaPoisson {
  /* the shape parameter of the rate must be an integer for the marginal
   * to make sense as a negative binomial distribution; instantiate the
   * rate if this is not the case */
  if (λ.k != Real(Integer(λ.k))) {
    λ.value();
  }
  
  x:GammaPoisson;
  x.initialize(λ);
  return x;
}
