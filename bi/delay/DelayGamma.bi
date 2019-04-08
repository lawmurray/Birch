/*
 * Delayed gamma random variate.
 */
class DelayGamma(x:Random<Real>&, k:Real, θ:Real) < DelayValue<Real>(x) {
  /**
   * Shape.
   */
  k:Real <- k;

  /**
   * Scale.
   */
  θ:Real <- θ;

  function simulate() -> Real {
    return simulate_gamma(k, θ);
  }
  
  function observe(x:Real) -> Real {
    return observe_gamma(x, k, θ);
  }

  function update(x:Real) {
    //
  }

  function downdate(x:Real) {
    //
  }

  function pdf(x:Real) -> Real {
    return pdf_gamma(x, k, θ);
  }

  function cdf(x:Real) -> Real {
    return cdf_gamma(x, k, θ);
  }

  function lower() -> Real? {
    return 0.0;
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Gamma");
    buffer.set("k", k);
    buffer.set("θ", θ);
  }
}

function DelayGamma(x:Random<Real>&, k:Real, θ:Real) -> DelayGamma {
  m:DelayGamma(x, k, θ);
  return m;
}
