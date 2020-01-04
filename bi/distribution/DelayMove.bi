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

  function setChild(child:Delay) {
    if !x? {
      if !this.child? {
        this.child <- child;
      } else {
        assert this.child! == child;
      }
      if !p? {
        p <- lazy(Expression<Value>(this));
        p!.setChild(child);
      }
    }
  }

  /**
   * Lazy expression of logpdf.
   */
  abstract function lazy(x:Expression<Value>) -> Expression<Real>;

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
        if dfdx'? {
          x'' <- simulate_propose(x'!, dfdx'!);
        }
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
        p!.pilot();
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
        p!.propose();
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
    if !x? && !α? {
      α <- p!.propose() - p!.pilot();
      if dfdx'? && dfdx''? {
        α <- α! + logpdf_propose(x'!, x''!, dfdx''!) - logpdf_propose(x''!, x'!, dfdx'!);
      }
      return α! + p!.ratio();
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
      p <- nil;
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
        for i in 1..5 {
          auto l' <- p!.propose();
          if p!.gradPropose(1.0) {
            /* at least one gradient; continue by computing the acceptance
             * ratio for Metropolis--Hastings */
            assert !dfdx'?;
            assert !dfdx''?;
            if log(simulate_uniform(0.0, 1.0)) <= l' - l + p!.ratio() {
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
    p!.setChild(this);
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
    p!.setChild(this);
    return p!.pilot();
  }

  function realize() {
    prune();
    if !x? {
      if future? {
        x <- future!;
      } else {
        x <- simulate();
      }
    }
    if futureMove {
      move(x!);
    }
    p <- nil;
    if futureUpdate {
      update(x!);
    } else {
      downdate(x!);
    }
  }
}


function simulate_propose(x:Real, d:Real) -> Real {
  return simulate_gaussian(x + 0.03*d, 0.06);
}

function simulate_propose(x:Real[_], d:Real[_]) -> Real[_] {
  return simulate_multivariate_gaussian(x + d, 1.0);
}

function simulate_propose(x:Real[_,_], d:Real[_,_]) -> Real[_,_] {
  return simulate_matrix_gaussian(x + d, 1.0);
}

function simulate_propose(x:Integer, d:Integer) -> Integer {
  return x;
}

function simulate_propose(x:Integer[_], d:Integer[_]) -> Integer[_] {
  return x;
}

function simulate_propose(x:Boolean, d:Boolean) -> Boolean {
  return x;
}

function logpdf_propose(x':Real, x:Real, d:Real) -> Real {
  return logpdf_gaussian(x', x + 0.03*d, 0.06);
}

function logpdf_propose(x':Real[_], x:Real[_], d:Real[_]) -> Real {
  return logpdf_multivariate_gaussian(x', x + d, 1.0);
}

function logpdf_propose(x':Real[_,_], x:Real[_,_], d:Real[_,_]) -> Real {
  return logpdf_matrix_gaussian(x', x + d, 1.0);
}

function logpdf_propose(x':Integer, x:Integer, d:Integer) -> Real {
  return 0.0;
}

function logpdf_propose(x':Integer[_], x:Integer[_], d:Integer[_]) -> Real {
  return 0.0;
}

function logpdf_propose(x':Boolean, x:Boolean, d:Boolean) -> Real {
  return 0.0;
}
