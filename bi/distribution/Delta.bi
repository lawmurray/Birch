/**
 * Delta distribution, representing a distribution on a discrete space with
 * all probability mass at one location.
 */
class Delta(μ:Expression<Integer>) < Distribution<Integer> {
  /**
   * Location.
   */
  μ:Expression<Integer> <- μ;

  function graft() -> DelayValue<Integer> {
    m1:DelayValue<Integer>?;
    m2:TransformLinearDiscrete?;
    
    if (m1 <- μ.graftDiscrete())? {
      return DelayDiscreteDelta(m1!);
    } else if (m2 <- μ.graftLinearDiscrete())? {
      return DelayLinearDiscreteDelta(m2!.a, m2!.x, m2!.c);
    } else {
      return DelayDelta(μ);
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
