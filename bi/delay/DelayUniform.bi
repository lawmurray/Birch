/*
 * Delayed uniform random variable.
 */
final class DelayUniform(x:Random<Real>&, l:Real, u:Real) < DelayValue<Real>(x) {
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

  function update(x:Real) {
    //
  }

  function downdate(x:Real) {
    //
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

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Uniform");
    buffer.set("l", l);
    buffer.set("u", u);
  }
}

function DelayUniform(x:Random<Real>&, l:Real, u:Real) -> DelayUniform {
  m:DelayUniform(x, l, u);
  return m;
}
