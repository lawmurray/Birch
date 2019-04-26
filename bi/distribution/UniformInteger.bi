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

  function graft() {
    if delay? {
      delay!.prune();
    } else {
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

  function write(buffer:Buffer) {
    if delay? {
      delay!.write(buffer);
    } else {
      buffer.set("class", "UniformInteger");
      buffer.set("l", l.value());
      buffer.set("u", u.value());
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
