import delay.DelayBoolean;
import math;
import random;

/**
 * Bernoulli distribution.
 */
class Bernoulli < DelayBoolean {
  /**
   * Probability of a true result.
   */
  ρ:Real;

  function initialize(ρ:Real) {
    assert 0.0 <= ρ && ρ <= 1.0;
  
    super.initialize();
    this.ρ <- ρ;
  }

  function update(ρ:Real) {
    assert 0.0 <= ρ && ρ <= 1.0;
  
    super.update();
    this.ρ <- ρ;
  }

  function doRealize() {
    if (isMissing()) {
      set(random_bernoulli(ρ));
    } else {
      if (x) {
        setWeight(log(ρ));
       } else {
        setWeight(log(1.0 - ρ));
      }
    }
  }
}

/**
 * Create.
 */
function Bernoulli(ρ:Real) -> Bernoulli {
  m:Bernoulli;
  m.initialize(ρ);
  return m;
}
