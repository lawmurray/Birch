/**
 * Delta distribution, representing a distribution on a discrete space with
 * all probability mass at one location.
 */
class Delta(μ:Expression<Integer>) < Distribution<Integer> {
  /**
   * Location.
   */
  μ:Expression<Integer> <- μ;

  function graft() {
    if delay? {
      delay!.prune();
    } else {
      m1:DelayBoundedDiscrete?;
      m2:TransformLinearBoundedDiscrete?;
      m3:DelayDiscrete?;
      m4:TransformLinearDiscrete?;
      m5:TransformAddBoundedDiscrete?;
      m6:TransformSubtractBoundedDiscrete?;

      if (m1 <- μ.graftBoundedDiscrete())? {
        delay <- DelayDiscreteDelta(x, m1!);
      } else if (m2 <- μ.graftLinearBoundedDiscrete())? {
        delay <- DelayLinearDiscreteDelta(x, m2!.a, m2!.x, m2!.c);
      } else if (m3 <- μ.graftDiscrete())? {
        delay <- DelayDiscreteDelta(x, m3!);
      } else if (m4 <- μ.graftLinearDiscrete())? {
        delay <- DelayLinearDiscreteDelta(x, m4!.a, m4!.x, m4!.c);
      } else if (m5 <-  μ.graftAddBoundedDiscrete())? {
        delay <- DelayAddBoundedDiscreteDelta(x, m5!.x1, m5!.x2);
      } else if (m6 <-  μ.graftSubtractBoundedDiscrete())? {
        delay <- DelaySubtractBoundedDiscreteDelta(x, m6!.x1, m6!.x2);
      } else {
        delay <- DelayDelta(x, μ);
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
