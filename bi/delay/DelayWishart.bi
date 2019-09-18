/*
 * Delayed Wishart random variate.
 */
final class DelayWishart(future:Real[_,_]?, futureUpdate:Boolean,
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
    return simulate_wishart(Ψ, ν);
  }
  
  function logpdf(X:Real[_,_]) -> Real {
    return logpdf_wishart(X, Ψ, ν);
  }

  function update(X:Real[_,_]) {
    //
  }

  function downdate(X:Real[_,_]) {
    //
  }

  function pdf(X:Real[_,_]) -> Real {
    return pdf_wishart(X, Ψ, ν);
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Wishart");
    buffer.set("Ψ", Ψ);
    buffer.set("ν", ν);
  }
}

function DelayWishart(future:Real[_,_]?, futureUpdate:Boolean, Ψ:Real[_,_],
    ν:Real) -> DelayWishart {
  m:DelayWishart(future, futureUpdate, Ψ, ν);
  return m;
}
