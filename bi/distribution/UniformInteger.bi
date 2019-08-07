/**
 * Uniform distribution over integers.
 */
final class UniformInteger(l:Expression<Integer>, u:Expression<Integer>) <
    Distribution<Integer> {
  /**
   * Lower bound.
   */
  l:Expression<Integer> <- l;
  
  /**
   * Upper bound.
   */
  u:Expression<Integer> <- u;

  function valueForward() -> Integer {
    assert !delay?;
    return simulate_uniform_int(l, u);
  }

  function observeForward(x:Integer) -> Real {
    assert !delay?;
    return logpdf_uniform_int(x, l, u);
  }

  function graft(force:Boolean) {
    if delay? {
      delay!.prune();
    } else if force {
      delay <- DelayUniformInteger(future, futureUpdate, l, u);
    }
  }

  function graftDiscrete() -> DelayDiscrete? {
    return graftBoundedDiscrete();
  }

  function graftBoundedDiscrete() -> DelayBoundedDiscrete? {
    if delay? {
      delay!.prune();
    } else {
      delay <- DelayUniformInteger(future, futureUpdate, l, u);
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
