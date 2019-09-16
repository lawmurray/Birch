/*
 * Delayed multivariate uniform random variable.
 */
final class DelayIndependentUniform(future:Real[_]?, futureUpdate:Boolean,
    l:Real[_], u:Real[_]) < DelayValue<Real[_]>(future, futureUpdate) {
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
  
  function logpdf(x:Real[_]) -> Real {
    return logpdf_multivariate_uniform(x, l, u);
  }

  function update(x:Real[_]) {
    //
  }

  function downdate(x:Real[_]) {
    //
  }

  function pdf(x:Real[_]) -> Real {
    return pdf_multivariate_uniform(x, l, u);
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "IndependentUniform");
    buffer.set("l", l);
    buffer.set("u", u);
  }
}

function DelayIndependentUniform(future:Real[_]?, futureUpdate:Boolean,
    l:Real[_], u:Real[_]) -> DelayIndependentUniform {
  m:DelayIndependentUniform(future, futureUpdate, l, u);
  return m;
}
