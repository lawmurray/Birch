/*
 * ed uniform integer random variate.
 */
final class UniformInteger(future:Integer?, futureUpdate:Boolean,
    l:Expression<Integer>, u:Expression<Integer>) < BoundedDiscrete(future, futureUpdate, l, u) {
  function simulate() -> Integer {
    if value? {
      return value!;
    } else {
      return simulate_uniform_int(l, u);
    }
  }

  function logpdf(x:Integer) -> Real {
    return logpdf_uniform_int(x, l, u);
  }

  function cdf(x:Integer) -> Real? {
    return cdf_uniform_int(x, l, u);
  }

  function quantile(P:Real) -> Integer? {
    return quantile_uniform_int(P, l, u);
  }

  function graft() -> Distribution<Integer> {
    prune();
    return this;
  }

  function graftDiscrete() -> Discrete? {
    prune();
    return this;
  }

  function graftBoundedDiscrete() -> BoundedDiscrete? {
    prune();
    return this;
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "UniformInteger");
    buffer.set("l", l);
    buffer.set("u", u);
  }
}

function UniformInteger(future:Integer?, futureUpdate:Boolean,
    l:Expression<Integer>, u:Expression<Integer>) -> UniformInteger {
  m:UniformInteger(future, futureUpdate, l, u);
  return m;
}

/**
 * Create uniform distribution over integers.
 */
function Uniform(l:Expression<Integer>, u:Expression<Integer>) -> UniformInteger {
  m:UniformInteger(nil, true, l, u);
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
