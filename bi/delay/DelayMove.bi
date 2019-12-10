/*
 * Type-specific interface for delayed sampling $M$-path nodes with move
 * support.
 *
 * - Base: Base type.
 *
 * - future: Future value.
 * - futureUpdate: When realized, should the future value trigger an
 *   update? (Otherwise a downdate.)
 */
abstract class DelayMove<Base>(future:Value?, futureUpdate:Boolean) <
    Base(future, futureUpdate) {
  /**
   * Logpdf function.
   */
  p:Expression<Real>?;
    
  /**
   * When assigned, should the value trigger a move? Typically an observe
   * will trigger a move, while a simulation will not.
   */
  futureMove:Boolean <- false;

  /**
   * Attempt to move random variates upon which this delayed value depends.
   */
  function move(x:Value) {
    if p? {
      /* have a lazy expression on which we can attempt a move; first
       * evaluate the log-likelihood and its gradient at a pilot position */
      auto l <- p!.pilot();
      if p!.gradPilot(1.0) {
        /* at least one gradient; continue by evaluating the log-likelihood
         * and it gradient at a proposal position */
        auto l' <- p!.propose();
        if p!.gradPropose(1.0) {
          /* at least one gradient; continue by computing the acceptance
           * ratio for Metropolis--Hastings */
          auto α <- l' - l + p!.ratio();
          if log(simulate_uniform(0.0, 1.0)) <= α {
            /* accept the move */
            p!.accept();
          } else {
            /* reject the move */
            p!.reject();
          }
        } else {
          /* should not happen, as there were gradients the first time */
          assert false;
        }
      }
      p!.clamp();
      p <- nil;
    }
  }

  function observe(x:Value) -> Real {
    assert !this.x?;
    assert !this.future?;
    prune();
    this.x <- x;
    futureUpdate <- true;
    futureMove <- true;
    p <- lazy(Boxed(x));
    p!.graft(this);
    return p!.pilot();
  }

  function observeWithDowndate(x:Value) -> Real {
    assert !this.x?;
    assert !this.future?;
    prune();
    this.x <- x;
    futureUpdate <- false;
    futureMove <- true;
    p <- lazy(Boxed(x));
    p!.graft(this);
    return p!.pilot();
  }

  function realize() {
    super.realize();
    if futureMove {
      move(x!);
    }
    p <- nil;
  }
}
