/**
 * Delta distribution, representing a distribution on a discrete space with
 * all probability mass at one location.
 */
class Delta(μ:Expression<Integer>) < Random<Integer> {
  /**
   * Location.
   */
  μ:Expression<Integer> <- μ;

  function graft() -> Delay? {
    if (delay?) {
      return delay;
    } else {
      m1:DelayBinomial?;
      m2:TransformLinearDiscrete?;
    
      if (m1 <- μ.graftBinomial())? {
        return DelayBinomialDelta(this, m1!);
      } else if (m2 <- μ.graftLinearDiscrete())? {
        return DelayLinearDiscreteDelta(this, m2!.a, m2!.x, m2!.c);
      } else {
        return DelayDelta(this, μ);
      }
    }
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
