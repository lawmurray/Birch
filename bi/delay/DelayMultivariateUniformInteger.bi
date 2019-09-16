/*
 * Delayed multivariate uniform random variable over integers.
 */
final class DelayIndependentUniformInteger(future:Integer[_]?,
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
  
  function logpdf(x:Integer[_]) -> Real {
    return logpdf_multivariate_uniform_int(x, l, u);
  }

  function update(x:Integer[_]) {
    //
  }

  function downdate(x:Integer[_]) {
    //
  }

  function pdf(x:Integer[_]) -> Real {
    return pdf_multivariate_uniform_int(x, l, u);
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "IndependentUniformInteger");
    buffer.set("l", l);
    buffer.set("u", u);
  }
}

function DelayIndependentUniformInteger(future:Integer[_]?,
    futureUpdate:Boolean, l:Integer[_], u:Integer[_]) ->
    DelayIndependentUniformInteger {
  m:DelayIndependentUniformInteger(future, futureUpdate, l, u);
  return m;
}
