/*
 * Poisson with conjugate gamma prior on rate.
 */
class GammaPoisson < Random<Integer> {
  /**
   * Rate.
   */
  λ:Gamma;

  function initialize(λ:Gamma) {
    super.initialize(λ);
    this.λ <- λ;
  }
  
  function doMarginalize() {
    //
  }
  
  function doCondition() {
    λ.update(λ.k + value(), λ.θ/(λ.θ + 1.0));
  }

  function doRealize() {
    if (λ.isRealized() || λ.k != floor(λ.k)) {
      // ^ must have integer shape for conjugacy
      if (isMissing()) {
        set(simulate_poisson(λ.value()));
      } else {
        setWeight(observe_poisson(value(), λ.value()));
      }
    } else {
      if (isMissing()) {
        set(simulate_gamma_poisson(Integer(λ.k), λ.θ));
      } else {
        setWeight(observe_gamma_poisson(value(), Integer(λ.k), λ.θ));
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
