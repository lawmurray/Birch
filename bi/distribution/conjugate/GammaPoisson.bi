/*
 * Poisson with conjugate gamma prior on rate.
 */
class GammaPoisson < Random<Integer> {
  /**
   * Rate.
   */
  λ:Expression<Real>;

  function initialize(λ:Expression<Real>) {
    super.initialize(λ);
    this.λ <- λ;
  }
  
  function doMarginalize() {
    //
  }
  
  function doCondition() {
    λ1:Gamma? <- Gamma?(λ);
    if (λ1?) {
      λ1!.update(λ1!.k + value(), λ1!.θ/(λ1!.θ + 1.0));
    }
  }

  function doRealize() {
    λ1:Gamma? <- Gamma?(λ);
    if (λ1? && !λ1!.isRealized() && λ1!.k == Integer(λ1!.k)) {
      /* ^ the shape parameter of the rate must be an integer for the marginal
       *   to make sense as a negative binomial distribution */
      if (isMissing()) {
        set(simulate_gamma_poisson(Integer(λ1!.k), λ1!.θ));
      } else {
        setWeight(observe_gamma_poisson(value(), Integer(λ1!.k), λ1!.θ));
      }
    } else {
      if (isMissing()) {
        set(simulate_poisson(λ.value()));
      } else {
        setWeight(observe_poisson(value(), λ.value()));
      }
    }
  }
}

/**
 * Create Poisson distribution.
 */
function Poisson(λ:Expression<Real>) -> GammaPoisson {
  x:GammaPoisson;
  x.initialize(λ);
  return x;
}
