/*
 * Delayed uniform random variable.
 */
class DelayUniform(l:Real, u:Real) < DelayValue<Real> {
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
}

function DelayUniform(l:Real, u:Real) -> DelayUniform {
  m:DelayUniform(l, u);
  return m;
}
