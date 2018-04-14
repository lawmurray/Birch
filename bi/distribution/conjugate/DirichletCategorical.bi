/*
 * Categorical distribution with conjugate prior on category probabilities.
 */
class DirichletCategorical < Random<Integer> {
  /**
   * Category probabilities.
   */
  ρ:Random<Real[_]>;

  function initialize(ρ:Random<Real[_]>) {
    super.initialize(ρ);
    this.ρ <- ρ;
  }
  
  function doMarginalize() {
    //
  }
  
  function doCondition() {
    ρ1:Dirichlet? <- Dirichlet?(ρ);
    if (ρ1?) {
      ρ1!.update(value());
    }
  }

  function doRealize() {
    ρ1:Dirichlet? <- Dirichlet?(ρ);
    ρ2:RestaurantProcess? <- RestaurantProcess?(ρ);
    if (ρ1? && !ρ1!.isRealized()) {
      if (isMissing()) {
        set(simulate_dirichlet_categorical(ρ1!.α));
      } else {
        setWeight(observe_dirichlet_categorical(value(), ρ1!.α));
      }
    } else if (ρ2? && !ρ2!.isRealized()) {
      if (isMissing()) {
        set(simulate_crp_categorical(ρ2!.α, ρ2!.θ, ρ2!.n[1..ρ2!.K], ρ2!.N));
      } else {
        setWeight(observe_crp_categorical(value(), ρ2!.α, ρ2!.θ,
            ρ2!.n[1..ρ2!.K], ρ2!.N));
      }
    } else {
      if (isMissing()) {
        set(simulate_categorical(ρ.value()));
      } else {
        setWeight(observe_categorical(value(), ρ.value()));
      }
    }
  }
}

/**
 * Create categorical distribution.
 */
function Categorical(ρ:Random<Real[_]>) -> DirichletCategorical {
  x:DirichletCategorical;
  x.initialize(ρ);
  return x;
}
