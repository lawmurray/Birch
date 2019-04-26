/*
 * Delayed multivariate uniform random variable over integers.
 */
final class DelayMultivariateUniformInteger(future:Integer[_]?,
    futureUpdate:Boolean, l:Integer[_], u:Integer[_]) <
    DelayValue<Integer[_]>(future, futureUpdate) {
  /**
   * Lower bound.
   */
  l:Integer[_] <- l;

  /**
   * Upper bound.
   */
  u:Integer[_] <- u;

  function simulate() -> Integer[_] {
    return simulate_multivariate_uniform_int(l, u);
  }
  
  function observe(x:Integer[_]) -> Real {
    return observe_multivariate_uniform_int(x, l, u);
  }

  function update(x:Integer[_]) {
    //
  }

  function downdate(x:Integer[_]) {
    //
  }

  function pmf(x:Integer[_]) -> Real {
    return pmf_multivariate_uniform_int(x, l, u);
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "MultivariateUniformInteger");
    buffer.set("l", l);
    buffer.set("u", u);
  }
}

function DelayMultivariateUniformInteger(future:Integer[_]?,
    futureUpdate:Boolean, l:Integer[_], u:Integer[_]) ->
    DelayMultivariateUniformInteger {
  m:DelayMultivariateUniformInteger(future, futureUpdate, l, u);
  return m;
}
