/**
 * Event handler for MoveParticleFilter.
 *
 * - delayed: Enable delayed sampling?
 *
 * The Handler class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Handler.svg"></object>
 * </center>
 */
class MoveHandler(delayed:Boolean) < Handler {
  /**
   * Is delayed sampling enabled?
   */
  delayed:Boolean <- delayed;
  
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
    auto w <- event.p.observeLazy(box(event.x));
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
        event.x <- record.x.value();
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
function MoveHandler(delayed:Boolean) -> MoveHandler {
  o:MoveHandler(delayed);
  return o;
}
