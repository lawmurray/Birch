/*
 * Binomial with conjugate beta prior on success probability.
 */
class BetaBinomial < Random<Integer> {
  /**
   * Number of trials.
   */
  n:Integer;

  /**
   * Probability of a true result.
   */
  ρ:Expression<Real>;

  function initialize(n:Integer, ρ:Expression<Real>) {
    assert 0 <= n;
  
    super.initialize(ρ);
    this.n <- n;
    this.ρ <- ρ;
  }
  
  function doMarginalize() {
    //
  }
  
  function doCondition() {
    ρ1:Beta? <- Beta?(ρ);
    if (ρ1?) {
      ρ1!.update(ρ1!.α + value(), ρ1!.β + n - value());
    }
  }

  function doRealize() {
    ρ1:Beta? <- Beta?(ρ);
    if (ρ1? && !ρ1!.isRealized()) {
      if (isMissing()) {
        set(simulate_beta_binomial(n, ρ1!.α, ρ1!.β));
      } else {
        setWeight(observe_beta_binomial(value(), n, ρ1!.α, ρ1!.β));
      }
    } else {
      if (isMissing()) {
        set(simulate_binomial(n, ρ.value()));
      } else {
        setWeight(observe_binomial(value(), n, ρ.value()));
      }
    }
  }
}

/**
 * Create binomial distribution.
 */
function Binomial(n:Integer, ρ:Expression<Real>) -> BetaBinomial {
  x:BetaBinomial;
  x.initialize(n, ρ);
  return x;
}
