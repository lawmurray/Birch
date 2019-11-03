/*
 * Delayed uniform integer random variate.
 */
final class DelayUniformInteger(future:Integer?, futureUpdate:Boolean,
    l:Integer, u:Integer) < DelayBoundedDiscrete(future, futureUpdate, l, u) {
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

  function update(x:Integer) {
    //
  }

  function downdate(x:Integer) {
    //
  }

  function cdf(x:Integer) -> Real {
    return cdf_uniform_int(x, l, u);
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "UniformInteger");
    buffer.set("l", l);
    buffer.set("u", u);
  }
}

function DelayUniformInteger(future:Integer?, futureUpdate:Boolean,
    l:Integer, u:Integer) -> DelayUniformInteger {
  m:DelayUniformInteger(future, futureUpdate, l, u);
  return m;
}
