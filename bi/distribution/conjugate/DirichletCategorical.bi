/**
 * Categorical distribution with conjugate prior on category probabilities.
 */
class DirichletCategorical < Random<Integer> {
  /**
   * Category probabilities.
   */
  ρ:Dirichlet;

  function initialize(ρ:Dirichlet) {
    super.initialize(ρ);
    this.ρ <- ρ;
  }
  
  function doMarginalize() {
    //
  }
  
  function doCondition() {
    ρ.update(x);
  }

  function doRealize() {
    if (ρ.isRealized()) {
      if (isMissing()) {
        set(simulate_categorical(ρ));
      } else {
        setWeight(observe_categorical(x, ρ));
      }
    } else {
      if (isMissing()) {
        set(simulate_dirichlet_categorical(ρ.α));
      } else {
        setWeight(observe_dirichlet_categorical(x, ρ.α));
      }
    }
  }

  function tildeLeft() -> DirichletCategorical {
    simulate();
    return this;
  }
}

/**
 * Create Dirichlet-categorical distribution.
 */
function Categorical(ρ:Dirichlet) -> DirichletCategorical {
  x:DirichletCategorical;
  x.initialize(ρ);
  return x;
}
