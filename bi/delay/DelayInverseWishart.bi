/*
 * Delayed inverse-Wishart random variate.
 */
final class DelayInverseWishart(future:Real[_,_]?, futureUpdate:Boolean,
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
    return simulate_inverse_wishart(Ψ, k);
  }
  
  function logpdf(X:Real[_,_]) -> Real {
    return logpdf_inverse_wishart(X, Ψ, k);
  }

  function update(X:Real[_,_]) {
    //
  }

  function downdate(X:Real[_,_]) {
    //
  }

  function pdf(X:Real[_,_]) -> Real {
    return pdf_inverse_wishart(X, Ψ, k);
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "InverseWishart");
    buffer.set("Ψ", Ψ);
    buffer.set("k", k);
  }
}

function DelayInverseWishart(future:Real[_,_]?, futureUpdate:Boolean,
    Ψ:Real[_,_], k:Real) -> DelayInverseWishart {
  m:DelayInverseWishart(future, futureUpdate, Ψ, k);
  return m;
}
