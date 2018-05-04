/**
 * Uniform distribution over integers.
 */
class UniformInteger(l:Expression<Integer>, u:Expression<Integer>) <
    Random<Integer> {
  /**
   * Lower bound.
   */
  l:Expression<Integer> <- l;
  
  /**
   * Upper bound.
   */
  u:Expression<Integer> <- u;

  function graft() -> Delay? {
    if (delay?) {
      return delay;
    } else {
      return DelayUniformInteger(this, l, u);
    }
  }

  function graftDiscrete() -> DelayValue<Integer>? {
    if (delay?) {
      return DelayValue<Integer>?(delay);
    } else {
      return DelayUniformInteger(this, l, u);
    }
  }
}

/**
 * Create uniform distribution over integers.
 */
function Uniform(l:Expression<Integer>, u:Expression<Integer>) -> UniformInteger {
  m:UniformInteger(l, u);
  return m;
}

/**
 * Create uniform distribution over integers.
 */
function Uniform(l:Expression<Integer>, u:Integer) -> UniformInteger {
  return Uniform(l, Boxed(u));
}

/**
 * Create uniform distribution over integers.
 */
function Uniform(l:Integer, u:Expression<Integer>) -> UniformInteger {
  return Uniform(Boxed(l), u);
}

/**
 * Create uniform distribution over integers.
 */
function Uniform(l:Integer, u:Integer) -> UniformInteger {
  return Uniform(Boxed(l), Boxed(u));
}
