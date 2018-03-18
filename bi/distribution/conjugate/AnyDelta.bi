/*
 * Delta with any other prior distribution.
 */
class AnyDelta < Random<Integer> {
  /**
   * Location.
   */
  μ:Random<Integer>;

  function initialize(μ:Random<Integer>) {
    super.initialize(μ);
    this.μ <- μ;
  }
  
  function doMarginalize() {
    //
  }
  
  function doCondition() {
    //
  }

  function doRealize() {
    if (isMissing()) {
      set(μ.value());
    } else {
      if (μ.isMissing()) {
        setWeight(μ.observe(value()));
      } else {
        setWeight(observe_delta(value(), μ.value()));
      }
    }
  }
}

/**
 * Create delta distribution.
 */
function Delta(μ:Random<Integer>) -> AnyDelta {
  x:AnyDelta;
  x.initialize(μ);
  return x;
}
