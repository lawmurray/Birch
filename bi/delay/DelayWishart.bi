/*
 * Delayed Wishart random variate.
 */
final class DelayWishart(future:Real[_,_]?, futureUpdate:Boolean,
    Ψ:Real[_,_], k:Real) < DelayValue<Real[_,_]>(future, futureUpdate) {
  /**
   * Scale.
   */
  Ψ:Real[_,_] <- Ψ;
  
  /**
   * Degrees of freedom.
   */
  k:Real <- k;

  function simulate() -> Real[_,_] {
    return simulate_wishart(Ψ, k);
  }
  
  function logpdf(X:Real[_,_]) -> Real {
    return logpdf_wishart(X, Ψ, k);
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Wishart");
    buffer.set("Ψ", Ψ);
    buffer.set("k", k);
  }
}

function DelayWishart(future:Real[_,_]?, futureUpdate:Boolean, Ψ:Real[_,_],
    k:Real) -> DelayWishart {
  m:DelayWishart(future, futureUpdate, Ψ, k);
  return m;
}
