/**
 * Delta distribution, representing a distribution on a discrete space with
 * all probability mass at one location.
 */
final class Delta(μ:Expression<Integer>) < Distribution<Integer> {
  /**
   * Location.
   */
  μ:Expression<Integer> <- μ;

  function valueForward() -> Integer {
    assert !delay?;
    return simulate_delta(μ);
  }

  function observeForward(x:Integer) -> Real {
    assert !delay?;
    return logpdf_delta(x, μ);
  }

  function graft(force:Boolean) {
    if delay? {
      delay!.prune();
    } else {
      m:DelayDiscrete?;
      if (m <- μ.graftDiscrete())? {
        delay <- DelayDiscreteDelta(future, futureUpdate, m!);
      } else if force {
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
