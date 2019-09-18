/*
 * Delayed inverse-Wishart random variate.
 */
final class DelayInverseWishart(future:Real[_,_]?, futureUpdate:Boolean,
    Ψ:Real[_,_], ν:Real) < DelayValue<Real[_,_]>(future, futureUpdate) {
  /**
   * Scale.
   */
  Ψ:Real[_,_] <- Ψ;
  
  /**
   * Degrees of freedom.
   */
  ν:Real <- ν;

  function simulate() -> Real[_,_] {
    return simulate_inverse_wishart(Ψ, ν);
  }
  
  function logpdf(X:Real[_,_]) -> Real {
    return logpdf_inverse_wishart(X, Ψ, ν);
  }

  function update(X:Real[_,_]) {
    //
  }

  function downdate(X:Real[_,_]) {
    //
  }

  function pdf(X:Real[_,_]) -> Real {
    return pdf_inverse_wishart(X, Ψ, ν);
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "InverseWishart");
    buffer.set("Ψ", Ψ);
    buffer.set("ν", ν);
  }
}

function DelayInverseWishart(future:Real[_,_]?, futureUpdate:Boolean,
    Ψ:Real[_,_], ν:Real) -> DelayInverseWishart {
  m:DelayInverseWishart(future, futureUpdate, Ψ, ν);
  return m;
}
