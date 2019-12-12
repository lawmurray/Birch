/**
 * Delta distribution, representing a distribution on a discrete space with
 * all probability mass at one location.
 */
final class Delta(μ:Expression<Integer>) < Distribution<Integer> {
  /**
   * Location.
   */
  μ:Expression<Integer> <- μ;

  function graft(child:Delay?) {
    if delay? {
      delay!.prune();
    } else {
      m:DelayDiscrete?;
      if (m <- μ.graftDiscrete(child))? {
        delay <- DelayDiscreteDelta(future, futureUpdate, m!);
      } else {
        delay <- DelayDelta(future, futureUpdate, μ);
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
