import delay.DelayInteger;
import math;

/**
 * Binomial distribution.
 */
class Binomial < DelayInteger {
  /**
   * Number of trials.
   */
  n:Integer;

  /**
   * Probability of a true result.
   */
  ρ:Real;

  function initialize(n:Integer, ρ:Real) {
    super.initialize();
    update(n, ρ);
  }

  function update(n:Integer, ρ:Real) {
    assert 0 <= n;
    assert 0.0 <= ρ && ρ <= 1.0;
  
    this.n <- n;
    this.ρ <- ρ;
  }

  function doRealize() {
    if (isMissing()) {
      set(simulate_binomial(n, ρ));
    } else {
      setWeight(observe_binomial(x, n, ρ));
    }
  }

  function tildeLeft() -> Binomial {
    simulate();
    return this;
  }
  
  function tildeRight(left:Binomial) -> Binomial {
    set(left.value());
    observe();
    return this;
  }
}

/**
 * Create.
 */
function Binomial(n:Integer, ρ:Real) -> Binomial {
  m:Binomial;
  m.initialize(n, ρ);
  return m;
}
