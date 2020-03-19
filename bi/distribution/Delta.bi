/**
 * Delta distribution, representing a distribution on a discrete space with
 * all probability mass at one location.
 */
class Delta(μ:Expression<Integer>) < Discrete {
  /**
   * Location.
   */
  μ:Expression<Integer> <- μ;

  function simulate() -> Integer {
    if value? {
      return value!;
    } else {
      return simulate_delta(μ.value());
    }
  }
  
  function logpdf(x:Integer) -> Real {
    return logpdf_delta(x, μ.value());
  }

  function lower() -> Integer? {
    return μ.value();
  }
  
  function upper() -> Integer? {
    return μ.value();
  }

  function graft() -> Distribution<Integer> {
    prune();
    m:Discrete?;
    r:Distribution<Integer>?;
    
    /* match a template */
    if (m <- μ.graftDiscrete())? {
      r <- DiscreteDelta(m!);
    }
    
    /* finalize, and if not valid, use default template */
    if !r? || !r!.graftFinalize() {
      r <- GraftedDelta(μ);
      r!.graftFinalize();
    }
    return r!;
  }

  function graftFinalize() -> Boolean {
    assert false;  // should have been replaced during graft
    return false;
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Delta");
    buffer.set("μ", μ);
  }
}

/**
 * Create delta distribution.
 *
 * - μ: Location.
 */
function Delta(μ:Expression<Integer>) -> Delta {
  m:Delta(μ);
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
