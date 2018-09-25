/*
 * Delayed multivariate uniform random variable over integers.
 */
class DelayMultivariateUniformInteger(x:Random<Integer[_]>&, l:Integer[_],
    u:Integer[_]) < DelayValue<Integer[_]>(x) {
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

  function pmf(x:Integer[_]) -> Real {
    return pmf_multivariate_uniform_int(x, l, u);
  }
}

function DelayMultivariateUniformInteger(x:Random<Integer[_]>&, l:Integer[_],
    u:Integer[_]) -> DelayMultivariateUniformInteger {
  m:DelayMultivariateUniformInteger(x, l, u);
  return m;
}
