/*
 * Delayed multinomial random variate.
 */
final class DelayMultinomial(future:Integer[_]?, futureUpdate:Boolean,
    n:Integer, ρ:Real[_]) < DelayValue<Integer[_]>(future, futureUpdate) {
  /**
   * Number of trials.
   */
  n:Integer <- n;
   
  /**
   * Category probabilities.
   */
  ρ:Real[_] <- ρ;

  function simulate() -> Integer[_] {
    return simulate_multinomial(n, ρ);
  }
  
  function logpdf(x:Integer[_]) -> Real {
    return logpdf_multinomial(x, n, ρ);
  }

  function update(x:Integer[_]) {
    //
  }

  function downdate(x:Integer[_]) {
    //
  }

  function pdf(x:Integer[_]) -> Real {
    return pdf_multinomial(x, n, ρ);
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Multinomial");
    buffer.set("n", n);
    buffer.set("ρ", ρ);
  }
}

function DelayMultinomial(future:Integer[_]?, futureUpdate:Boolean, n:Integer,
    ρ:Real[_]) -> DelayMultinomial {
  m:DelayMultinomial(future, futureUpdate, n, ρ);
  return m;
}
