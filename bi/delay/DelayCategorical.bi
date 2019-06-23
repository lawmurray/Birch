/*
 * Delayed Categorical random variate.
 */
final class DelayCategorical(future:Integer?, futureUpdate:Boolean, ρ:Real[_]) <
    DelayValue<Integer>(future, futureUpdate) {
  /**
   * Category probabilities.
   */
  ρ:Real[_] <- ρ;

  function simulate() -> Integer {
    return simulate_categorical(ρ);
  }
  
  function logpdf(x:Integer) -> Real {
    return logpdf_categorical(x, ρ);
  }

  function update(x:Integer) {
    //
  }

  function downdate(x:Integer) {
    //
  }

  function pdf(x:Integer) -> Real {
    return pdf_categorical(x, ρ);
  }

  function cdf(x:Integer) -> Real {
    return cdf_categorical(x, ρ);
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Categorical");
    buffer.set("ρ", ρ);
  }
}

function DelayCategorical(future:Integer?, futureUpdate:Boolean,
    ρ:Real[_]) -> DelayCategorical {
  m:DelayCategorical(future, futureUpdate, ρ);
  return m;
}
