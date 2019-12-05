/*
 * Type-specific interface for delayed sampling $M$-path nodes.
 *
 * - Value: Value type.
 *
 * - future: Future value.
 * - futureUpdate: When realized, should the future value trigger an
 *   update? (Otherwise a downdate.)
 */
abstract class DelayValue<Value>(future:Value?, futureUpdate:Boolean) < Delay {
  /**
   * Final value.
   */
  x:Value?;
  
  /**
   * Piloted value.
   */
  x':Value?;
  
  /**
   * Proposed value.
   */
  x'':Value?;
   
  /**
   * Future value. This is set for situations where delayed sampling
   * is used, but when ultimately realized, a particular value (this one)
   * should be assigned, and updates or downdates applied accordingly. It
   * is typically used when replaying traces.
   */
  future:Value? <- future;

  /**
   * When assigned, should the future value trigger an update? (Otherwise
   * a downdate.)
   */
  futureUpdate:Boolean <- futureUpdate;
  
  /**
   * When assigned, should the value trigger a move? Typically an observe
   * will trigger a move, while a simulation will not.
   */
  futureMove:Boolean <- false;

  /**
   * Does the node have a value?
   */
  function hasValue() -> Boolean {
    return x?;
  }

  /**
   * Realize a value for a random variate associated with this node.
   */
  function value() -> Value {
    if !x? {
      prune();
      if future? {
        x <- future!;
      } else {
        x <- simulate();
      }
    }
    return x!;
  }

  /**
   * Pilot value.
   */
  function pilot() -> Value {
    if x? {
      return x!;
    } else {
      assert !future?;
      if !x'? {
        prune();
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
        prune();
        x'' <- simulatePropose();
      }
      return x''!;
    }
  }

  /**
   * Set a value for a random variate associated with this node.
   */
  function set(x:Value) {
    assert !this.x?;
    assert !this.future?;
    prune();
    this.x <- x;
  }

  /**
   * Set a value for a random variate associated with this node,
   * downdating the delayed sampling graph accordingly.
   */
  function setWithDowndate(x:Value) {
    assert !this.x?;
    assert !this.future?;
    prune();
    this.x <- x;
    this.futureUpdate <- false;
  }
  
  /**
   * Observe a value for a random variate associated with this node,
   * updating (or downdating) the delayed sampling graph accordingly, and
   * returning a weight giving the log pdf (or pmf) of that variate under the
   * distribution.
   */
  function observe(x:Value) -> Real {
    assert !this.x?;
    assert !this.future?;

    prune();
    this.x <- x;
    this.futureUpdate <- true;
    this.futureMove <- true;
    return logpdfPilot(x);
  }

  /**
   * Observe a value for a random variate associated with this node,
   * updating (or downdating) the delayed sampling graph accordingly, and
   * returning a weight giving the log pdf (or pmf) of that variate under the
   * distribution.
   */
  function observeWithDowndate(x:Value) -> Real {
    assert !this.x?;
    assert !this.future?;

    prune();
    this.x <- x;
    this.futureUpdate <- false;
    this.futureMove <- true;
    return logpdfPilot(x);
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
    if futureUpdate {
      update(x!);
    } else {
      downdate(x!);
    }
    if futureMove {
      move(x!);
    }
  }
  
  /**
   * Simulate a value.
   *
   * Return: the value.
   */
  abstract function simulate() -> Value;

  /**
   * Simulate a pilot value.
   *
   * Return: the value.
   */
  function simulatePilot() -> Value {
    return simulate();
  }

  /**
   * Simulate a proposal value.
   *
   * Return: the value.
   */
  function simulatePropose() -> Value {
    return simulate();
  }

  /**
   * Log-pdf of a value.
   *
   * - x: The value.
   *
   * Return: The log likelihood.
   */
  abstract function logpdf(x:Value) -> Real;

  /**
   * Log-pdf of a value given piloted position.
   *
   * - x: The value.
   *
   * Return: The log likelihood.
   */
  function logpdfPilot(x:Value) -> Real {
    return logpdf(x);
  }

  /**
   * Log-pdf of a value given proposed position.
   *
   * - x: The value.
   *
   * Return: The log likelihood.
   */
  function logpdfPropose(x:Value) -> Real {
    return logpdf(x);
  }

  /**
   * Lazily observe a random variate, if supported.
   *
   * - x: The value.
   *
   * Return: The log likelihood.
   */
  function lazy(x:Expression<Value>) -> Expression<Real>? {
    return nil;
  }

  /**
   * Update the parent node on the $M$-path given the value of this node.
   *
   * - x: The value.
   */
  function update(x:Value) {
    //
  }

  /**
   * Downdate the parent node on the $M$-path given the value of this node.
   *
   * - x: The value.
   */
  function downdate(x:Value) {
    //
  }
  
  /**
   * Attempt to move random variates upon which this delayed value depends.
   */
  function move(x:Value) {
    auto p <- lazy(Boxed(x));
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
    }
  }
  
  /**
   * Evaluate the probability density (or mass) function, if it exists.
   *
   * - x: The value.
   *
   * Return: the probability density (or mass).
   */
  function pdf(x:Value) -> Real {
    return exp(logpdf(x));
  }

  /**
   * Evaluate the cumulative distribution function at a value.
   *
   * - x: The value.
   *
   * Return: the cumulative probability, if supported.
   */
  function cdf(x:Value) -> Real? {
    return nil;
  }

  /**
   * Evaluate the quantile function at a cumulative probability.
   *
   * - x: The cumulative probability.
   *
   * Return: the quantile, if supported.
   */
  function quantile(p:Real) -> Value? {
    return nil;
  }
  
  /**
   * Finite lower bound of the support of this node, if any.
   */
  function lower() -> Value? {
    return nil;
  }
  
  /**
   * Finite upper bound of the support of this node, if any.
   */
  function upper() -> Value? {
    return nil;
  }

  function write(buffer:Buffer) {
    buffer.set(value());
  }
}
