/**
 * Standard event handler.
 *
 * - delayed: Enable delayed sampling?
 *
 * The Handler class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Handler.svg"></object>
 * </center>
 */
class PlayHandler(delayed:Boolean) < Handler {
  /**
   * Is delayed sampling enabled?
   */
  delayed:Boolean <- delayed;

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
    return event.p.observe(event.x);
  }

  function handle<Value>(event:AssumeEvent<Value>) -> Real {
    if delayed {
      event.p <- event.p.graft();
    }
    if event.x.hasValue() {
      return event.p.observe(event.x.value());
    } else {
      event.x.assume(event.p);
      return 0.0;
    }
  }

  function handle(event:FactorEvent) -> Real {
    return event.w;
  }

  function handle<Value>(record:SimulateRecord<Value>,
      event:SimulateEvent<Value>) -> Real {
    if delayed {
      event.p <- event.p.graft();
    }
    
    /* in some situations, e.g. replaying a trace with different parameters,
     * it is possible that a recorded value is now outside the support of
     * its prior distribution; handle this situation by treating the recorded
     * value as an observation at first, then returning a log-weight of -inf
     * if outside the support of the prior distribution, zero otherwise */
    auto w <- event.p.observe(record.x);
    if w > -inf {
      /* recorded value is within the support of the prior distribution */
      event.x <- record.x;
      w <- 0.0;
    }
    return w;
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
      return event.p.observe(event.x.value());
    } else if record.x.hasValue() {
      /* if the record has a value, we can set it now, even if its
       * simulation was delayed when originally played; such delays do not
       * change the distribution, only the way it is computed */
      auto w <- event.p.observe(record.x.value());
      if w > -inf {
        /* recorded value is within the support of the prior distribution */
        event.x <- record.x;
        w <- 0.0;
      }
      return w;
    } else {
      /* ...otherwise it can be eliminated again */
      event.x.assume(event.p);
      return 0.0;
    }
  }

  function handle(record:FactorRecord, event:FactorEvent) -> Real {
    /* factor events are replayed in the same way they are played */
    return event.w;
  }
}

/**
 * Create a PlayHandler.
 */
function PlayHandler(delayed:Boolean) -> PlayHandler {
  o:PlayHandler(delayed);
  return o;
}
