/**
 * Multinomial with conjugate prior on category probabilities.
 */
class DirichletMultinomial < Random<Integer[_]> {
  /**
   * Number of trials.
   */
  n:Integer;

  /**
   * Category probabilities.
   */
  ρ:Dirichlet;

  function initialize(n:Integer, ρ:Dirichlet) {
    assert 0 <= n;
  
    super.initialize(ρ);
    this.n <- n;
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
        set(simulate_multinomial(n, ρ));
      } else {
        setWeight(observe_multinomial(x, n, ρ));
      }
    } else {
      if (isMissing()) {
        set(simulate_dirichlet_multinomial(n, ρ.α));
      } else {
        setWeight(observe_dirichlet_multinomial(x, n, ρ.α));
      }
    }
  }

  function tildeLeft() -> DirichletMultinomial {
    simulate();
    return this;
  }
}

/**
 * Create Dirichlet-multinomial distribution.
 */
function Multinomial(n:Integer, ρ:Dirichlet) -> DirichletMultinomial {
  x:DirichletMultinomial;
  x.initialize(n, ρ);
  return x;
}
