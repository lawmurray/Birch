/**
 * Delta distribution, representing a distribution on a discrete space with
 * all probability mass at one location.
 */
final class Delta(future:Integer?, futureUpdate:Boolean, μ:Expression<Integer>) <
    Discrete(future, futureUpdate) {
  /**
   * Location.
   */
  μ:Expression<Integer> <- μ;

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

  function graft() -> Distribution<Integer> {
    prune();
    m:Discrete?;
    if (m <- μ.graftDiscrete())? {
      return DiscreteDelta(future, futureUpdate, m!);
    } else {
      return this;
    }
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Delta");
    buffer.set("μ", μ);
  }
}

function Delta(future:Integer?, futureUpdate:Boolean,
    μ:Expression<Integer>) -> Delta {
  m:Delta(future, futureUpdate, μ);
  return m;
}

/**
 * Create delta distribution.
 *
 * - μ: Location.
 */
function Delta(μ:Expression<Integer>) -> Delta {
  m:Delta(nil, true, μ);
  return m;
}

/**
 * Create delta distribution.
 *
 * - μ: Location.
 */
function Delta(μ:Integer) -> Delta {
  return Delta(Boxed(μ));
}
