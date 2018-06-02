/**
 * Uniform distribution over integers.
 */
class UniformInteger(l:Expression<Integer>, u:Expression<Integer>) <
    Distribution<Integer> {
  /**
   * Lower bound.
   */
  l:Expression<Integer> <- l;
  
  /**
   * Upper bound.
   */
  u:Expression<Integer> <- u;

  function graft() {
    if delay? {
      delay!.prune();
    } else {
      delay <- DelayUniformInteger(x, l, u);
    }
  }

  function graftDiscrete() -> DelayDiscrete? {
    if delay? {
      delay!.prune();
    } else {
      delay <- DelayUniformInteger(x, l, u);
    }
    return DelayDiscrete?(delay);
  }

  function graftBoundedDiscrete() -> DelayBoundedDiscrete? {
    if delay? {
      delay!.prune();
    } else {
      delay <- DelayUniformInteger(x, l, u);
    }
    return DelayBoundedDiscrete?(delay);
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
