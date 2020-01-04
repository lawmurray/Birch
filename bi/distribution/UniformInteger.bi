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

  function quantile(p:Real) -> Integer? {
    return quantile_uniform_int(p, l, u);
  }

  function graft() {
    if delay? {
      delay!.prune();
    } else {
      delay <- UniformInteger(future, futureUpdate, l, u);
    }
  }

  function graftDiscrete() -> Discrete? {
    return graftBoundedDiscrete();
  }

  function graftBoundedDiscrete() -> BoundedDiscrete? {
    if delay? {
      delay!.prune();
    } else {
      delay <- UniformInteger(future, futureUpdate, l, u);
    }
    return BoundedDiscrete?(delay);
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "UniformInteger");
    buffer.set("l", l);
    buffer.set("u", u);
  }
}

function UniformInteger(future:Integer?, futureUpdate:Boolean,
    l:Integer, u:Integer) -> UniformInteger {
  m:UniformInteger(future, futureUpdate, l, u);
  return m;
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
