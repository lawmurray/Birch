/**
 * Event handler for MoveParticleFilter.
 *
 * - delayed: Enable delayed sampling?
 * - scale: Scale of moves.
 *
 * The Handler class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Handler.svg"></object>
 * </center>
 */
class MoveHandler(delayed:Boolean, scale:Real) < Handler {
  /**
   * Is delayed sampling enabled?
   */
  delayed:Boolean <- delayed;

  /**
   * Scale of moves.
   */
  scale:Real <- scale;
  
  /**
   * Deferred log-likelihood.
   */
  z:Expression<Real>?;

  final override function handle(event:Event) -> Real {
    /* double dispatch to one of the more specific handle() functions */
    return event.accept(this);
  }

  final override function handle(record:Record, event:Event) -> Real {
    /* double dispatch to one of the more specific handle() functions */
    return event.accept(record, this);
  }

  function handle<Value>(event:SimulateEvent<Value>) -> Real {
    if delayed {
      event.p <- event.p.graft();
    }
    event.x <- event.p.value();
    return 0.0;
  }

  function handle<Value>(event:ObserveEvent<Value>) -> Real {
    if delayed {
      event.p <- event.p.graft();
    }
    auto w <- event.p.observeLazy(Boxed(event.x));
    if w? {
      if z? {
        z <- z! + w!;
      } else {
        z <- w;
      }
      return 0.0;
    } else {
      /* lazy observe not supported for this distribution type */
      return event.p.observe(event.x);
    }
  }

  function handle<Value>(event:AssumeEvent<Value>) -> Real {
    if delayed {
      event.p <- event.p.graft();
    }
    if event.x.hasValue() {
      auto w <- event.p.observeLazy(event.x);
      if w? {
        if z? {
          z <- z! + w!;
        } else {
          z <- w;
        }
        return 0.0;
      } else {
        /* lazy observe not supported for this distribution type */
        return event.p.observe(event.x.value());
      }
    } else {
      event.x.assume(event.p);
      return 0.0;
    }
  }

  function handle(event:FactorEvent) -> Real {
    if z? {
      z <- z! + event.w;
    } else {
      z <- event.w;
    }
    return 0.0;
  }

  function handle<Value>(record:SimulateRecord<Value>,
      event:SimulateEvent<Value>) -> Real {
    if delayed {
      event.p <- event.p.graft();
    }
    event.x <- record.x;
    return 0.0;
  }

  function handle<Value>(record:ObserveRecord<Value>,
      event:ObserveEvent<Value>) -> Real {
    /* observe events are replayed in the same way they are played, it's
     * only necessary to check that the observed values actually match */
    assert record.x == event.x;
    return handle(event);
  }

  function handle<Value>(record:AssumeRecord<Value>,
      event:AssumeEvent<Value>) -> Real {
    if delayed {
      event.p <- event.p.graft();
    }
    if event.x.hasValue() {
      /* assume events with a value already assigned are replayed in the
       * same way they are played, it's only necessary to check that the
       * observed values actually match */
      assert record.x.hasValue() && record.x.value() == event.x.value();
      auto w <- event.p.observeLazy(event.x);
      if w? {
        if z? {
          z <- z! + w!;
        } else {
          z <- w;
        }
        return 0.0;
      } else {
        /* lazy observe not supported for this distribution type */
        return event.p.observe(event.x.value());
      }
    } else {
      event.x.assume(event.p);    
      if record.x.hasValue() {
        /* if the record has a value, we can set it now, even if its
         * simulation was delayed when originally played; such delays do not
         * change the distribution, only the way it is computed */
        event.x.setValue(record.x.value());
      }
      return 0.0;
    }
  }

  function handle(record:FactorRecord, event:FactorEvent) -> Real {
    /* factor events are replayed in the same way they are played */
    return handle(event);
  }
}

/**
 * Create a MoveHandler.
 */
function MoveHandler(delayed:Boolean, scale:Real) -> MoveHandler {
  o:MoveHandler(delayed, scale);
  return o;
}

function simulate_propose(x:Real, d:Real, τ:Real) -> Real {
  return simulate_gaussian(x + τ*d, 2.0*τ);
}

function simulate_propose(x:Real[_], d:Real[_], τ:Real) -> Real[_] {
  return simulate_multivariate_gaussian(x + τ*d, 2.0*τ);
}

function simulate_propose(x:Real[_,_], d:Real[_,_], τ:Real) -> Real[_,_] {
  return simulate_matrix_gaussian(x + τ*d, 2.0*τ);
}

function simulate_propose(x:Integer, d:Integer, τ:Real) -> Integer {
  return x;
}

function simulate_propose(x:Integer[_], d:Integer[_], τ:Real) -> Integer[_] {
  return x;
}

function simulate_propose(x:Boolean, d:Boolean, τ:Real) -> Boolean {
  return x;
}

function logpdf_propose(x':Real, x:Real, d:Real, τ:Real) -> Real {
  return logpdf_gaussian(x', x + τ*d, 2.0*τ);
}

function logpdf_propose(x':Real[_], x:Real[_], d:Real[_], τ:Real) -> Real {
  return logpdf_multivariate_gaussian(x', x + τ*d, 2.0*τ);
}

function logpdf_propose(x':Real[_,_], x:Real[_,_], d:Real[_,_],
    τ:Real) -> Real {
  return logpdf_matrix_gaussian(x', x + τ*d, 2.0*τ);
}

function logpdf_propose(x':Integer, x:Integer, d:Integer, τ:Real) -> Real {
  return 0.0;
}

function logpdf_propose(x':Integer[_], x:Integer[_], d:Integer[_],
    τ:Real) -> Real {
  return 0.0;
}

function logpdf_propose(x':Boolean, x:Boolean, d:Boolean, τ:Real) -> Real {
  return 0.0;
}

function ratio_propose(trace':Trace, trace:Trace, τ:Real) -> Real {
  auto α <- 0.0;
  auto r' <- trace'.walk();
  auto r <- trace.walk();
  while r'? && r? {
    α <- α + r'!.ratio(r!, τ);
  }
  assert !r'? && !r?;
  return α;
}
