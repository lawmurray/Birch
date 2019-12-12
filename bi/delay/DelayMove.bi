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
   * Piloted value.
   */
  x':Value?;
  
  /**
   * Gradient at the piloted value.
   */
  dfdx':Value?;
  
  /**
   * Proposed value.
   */
  x'':Value?;
  
  /**
   * Gradient at the proposed value.
   */
  dfdx'':Value?;
  
  /**
   * Contribution to the log acceptance ratio?
   */
  α:Real?;

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
   * Pilot value.
   */
  function pilot() -> Value {
    if x? {
      return x!;
    } else {
      assert !future?;
      if !x'? {
        x' <- simulatePilot();
      }
      return x'!;
    }
  }

  /**
   * Propose value.
   */
  function propose() -> Value {
    if x? {
      return x!;
    } else {
      assert !future?;
      if !x''? {
        x'' <- simulatePropose();
      }
      return x''!;
    }
  }

  function gradPilot(d:Value) -> Boolean {
    if x? {
      return false;
    } else {
      assert x'?;
      if !dfdx'? {
        /* first time this has been encountered in the gradient computation,
         * propagate into its prior */
        dfdx' <- d;
        if !p? {
          p <- lazy(DelayExpression<Value>(this));
        }
        α <- -p!.pilot();
        p!.gradPilot(1.0);
      } else {
        /* second or subsequent time this has been encountered in the gradient
         * computation; accumulate */
        dfdx' <- dfdx'! + d;
      }
      return dfdx'?;
    }
  }

  function gradPropose(d:Value) -> Boolean {
    if x? {
      return false;
    } else {
      assert x''?;
      if dfdx'? && !dfdx''? {
        /* first time this has been encountered in the gradient computation,
         * propagate into its prior */
        dfdx'' <- d;
        α <- α! + p!.propose();
        p!.gradPropose(1.0);
      } else {
        /* second or subsequent time this has been encountered in the gradient
         * computation; accumulate */
        dfdx'' <- dfdx''! + d;
      }
      return dfdx''?;
    }
  }
  
  function ratio() -> Real {
    if α? {
      if dfdx''? {
        α <- α! + logpdf_propose(x'!, x''!, dfdx''!);
      }
      auto result <- α!;
      α <- nil;
      return result;
    } else {
      return 0.0;
    }
  }
  
  function accept() {
    if x? {
      // nothing to do
    } else if x''? {
      x' <- x'';
      dfdx' <- dfdx'';
      x'' <- nil;
      dfdx'' <- nil;
      α <- nil;
      p!.accept();
    }
  }

  function reject() {
    if x? {
      // nothing to do
    } else if x''? {
      x'' <- nil;
      dfdx'' <- nil;
      α <- nil;
      p!.reject();
    }
  }

  function clamp() {
    if x? {
      // nothing to do
    } else {
      x <- x';
      x' <- nil;
      dfdx' <- nil;
      x'' <- nil;
      dfdx'' <- nil;
      α <- nil;
      p!.clamp();
    }
  }

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
