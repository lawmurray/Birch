/**
 * Binomial with conjugate prior on success probability. When the success
 * probability is known, this is simply a Binomial distribution.
 */
class BetaBinomial < Random<Integer> {
  /**
   * Number of trials.
   */
  n:Integer;

  /**
   * Probability of a true result.
   */
  ρ:Beta;

  function initialize(n:Integer, ρ:Beta) {
    assert 0 <= n;
  
    super.initialize(ρ);
    this.n <- n;
    this.ρ <- ρ;
  }
  
  function doMarginalize() {
    //
  }
  
  function doCondition() {
    ρ.update(ρ.α + x, ρ.β + n - x);
  }

  function doRealize() {
    if (ρ.isRealized()) {
      if (isMissing()) {
        set(simulate_binomial(n, ρ));
      } else {
        setWeight(observe_binomial(x, n, ρ));
      }
    } else {
      if (isMissing()) {
        set(simulate_beta_binomial(n, ρ.α, ρ.β));
      } else {
        setWeight(observe_beta_binomial(x, n, ρ.α, ρ.β));
      }
    }
  }
}

/**
 * Create binomial distribution.
 */
function Binomial(n:Integer, ρ:Beta) -> BetaBinomial {
  x:BetaBinomial;
  x.initialize(n, ρ);
  return x;
}

/**
 * Create binomial distribution.
 */
function Binomial(n:Integer, ρ:Random<Real>) -> Random<Integer> {
  ρ1:Beta? <- Beta?(ρ);
  if (ρ1?) {
    return Binomial(n, ρ1!);
  } else {
    return Binomial(n, ρ.value());
  }
}
