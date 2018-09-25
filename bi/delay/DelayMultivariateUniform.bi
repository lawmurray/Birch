/*
 * Delayed multivariate uniform random variable.
 */
class DelayMultivariateUniform(x:Random<Real[_]>&, l:Real[_], u:Real[_]) < DelayValue<Real[_]>(x) {
  /**
   * Lower bound.
   */
  l:Real[_] <- l;

  /**
   * Upper bound.
   */
  u:Real[_] <- u;

  function simulate() -> Real[_] {
    return simulate_multivariate_uniform(l, u);
  }
  
  function observe(x:Real[_]) -> Real {
    return observe_multivariate_uniform(x, l, u);
  }

  function pdf(x:Real[_]) -> Real {
    return pdf_multivariate_uniform(x, l, u);
  }
}

function DelayMultivariateUniform(x:Random<Real[_]>&, l:Real[_], u:Real[_]) -> DelayMultivariateUniform {
  m:DelayMultivariateUniform(x, l, u);
  return m;
}
