/*
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
  ρ:Random<Real[_]>;

  function initialize(n:Integer, ρ:Random<Real[_]>) {
    assert 0 <= n;
  
    super.initialize(ρ);
    this.n <- n;
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
    if (ρ1? && !ρ1!.isRealized()) {
      if (isMissing()) {
        set(simulate_dirichlet_multinomial(n, ρ1!.α));
      } else {
        setWeight(observe_dirichlet_multinomial(value(), n, ρ1!.α));
      }
    } else {
      if (isMissing()) {
        set(simulate_multinomial(n, ρ));
      } else {
        setWeight(observe_multinomial(value(), n, ρ));
      }
    }
  }
}

/**
 * Create multinomial distribution.
 */
function Multinomial(n:Integer, ρ:Random<Real[_]>) -> DirichletMultinomial {
  x:DirichletMultinomial;
  x.initialize(n, ρ);
  return x;
}
