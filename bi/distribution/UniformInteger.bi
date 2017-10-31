/**
 * Uniform distribution over integers.
 */
class UniformInteger < Random<Integer> {
  /**
   * Lower bound.
   */
  l:Integer;
  
  /**
   * Upper bound.
   */
  u:Integer;

  function initialize(l:Integer, u:Integer) {
    super.initialize();
    update(l, u);
  }

  function update(l:Integer, u:Integer) {
    assert l <= u;
  
    this.l <- l;
    this.u <- u;
  }

  function doRealize() {
    if (isMissing()) {
      set(simulate_int_uniform(l, u));
    } else {
      setWeight(observe_int_uniform(x, l, u));
    }
  }

  function tildeLeft() -> UniformInteger {
    simulate();
    return this;
  }
}

/**
 * Create a uniform distribution over integers.
 */
function Uniform(l:Integer, u:Integer) -> UniformInteger {
  m:UniformInteger;
  m.initialize(l, u);
  return m;
}
