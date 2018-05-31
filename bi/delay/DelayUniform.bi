/*
 * Delayed uniform random variable.
 */
class DelayUniform(x:Random<Real>&, l:Real, u:Real) < DelayValue<Real>(x) {
  /**
   * Lower bound.
   */
  l:Real <- l;

  /**
   * Upper bound.
   */
  u:Real <- u;

  function simulate() -> Real {
    return simulate_uniform(l, u);
  }
  
  function observe(x:Real) -> Real {
    return observe_uniform(x, l, u);
  }

  function pdf(x:Real) -> Real {
    return pdf_uniform(x, l, u);
  }

  function cdf(x:Real) -> Real {
    return cdf_uniform(x, l, u);
  }

  function lower() -> Real? {
    return l;
  }
  
  function upper() -> Real? {
    return u;
  }
}

function DelayUniform(x:Random<Real>&, l:Real, u:Real) -> DelayUniform {
  m:DelayUniform(x, l, u);
  return m;
}
