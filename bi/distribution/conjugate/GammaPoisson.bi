/**
 * Poisson with conjugate prior on rate. When the rate is known, this is
 * simply a Poisson distribution. When the rate is gamma distributed, this is
 * a negative binomial distribution.
 */
class GammaPoisson < Random<Integer> {
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
    assert round(λ.k) == λ.k;  ///@todo Polya distribution with real value?
    this.k <- Integer(λ.k);
    this.ρ <- 1.0/(λ.θ + 1.0);
  }
  
  function doCondition() {
    λ.update(λ.k + value(), λ.θ/(λ.θ + 1.0));
  }

  function doRealize() {
    if (λ.isRealized()) {
      if (isMissing()) {
        set(simulate_poisson(λ));
      } else {
        setWeight(observe_poisson(value(), λ));
      }
    } else {
      if (isMissing()) {
        set(simulate_negative_binomial(k, ρ));
      } else {
        setWeight(observe_negative_binomial(value(), k, ρ));
      }
    }
  }
}

/**
 * Create Poisson distribution.
 */
function Poisson(λ:Gamma) -> GammaPoisson {
  /* the shape parameter of the rate must be an integer for the marginal
   * to make sense as a negative binomial distribution; instantiate the
   * rate if this is not the case */
  if (!λ.isRealized() && λ.k != Real(Integer(λ.k))) {
    λ.value();
  }
  
  x:GammaPoisson;
  x.initialize(λ);
  return x;
}

/**
 * Create Poisson distribution.
 */
function Poisson(λ:Random<Real>) -> Random<Integer> {
  λ1:Gamma? <- Gamma?(λ);
  if (λ1?) {
    return Poisson(λ1!);
  } else {
    return Poisson(λ.value());
  }
}
