/*
 * Delayed delta random variate.
 */
final class DelayDelta(future:Integer?, futureUpdate:Boolean, μ:Integer) <
    DelayDiscrete(future, futureUpdate) {
  /**
   * Location.
   */
  μ:Integer <- μ;

  function simulate() -> Integer {
    if value? {
      return value!;
    } else {
      return simulate_delta(μ);
    }
  }
  
  function logpdf(x:Integer) -> Real {
    return logpdf_delta(x, μ);
  }

  function lower() -> Integer? {
    return μ;
  }
  
  function upper() -> Integer? {
    return μ;
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Delta");
    buffer.set("μ", μ);
  }
}

function DelayDelta(future:Integer?, futureUpdate:Boolean, μ:Integer) ->
    DelayDelta {
  m:DelayDelta(future, futureUpdate, μ);
  return m;
}
