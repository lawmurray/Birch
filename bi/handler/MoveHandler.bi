/**
 * Event handler for gradient-based moves.
 *
 * - delayed: Enable delayed sampling?
 * - scale: Scale of moves.
 *
 * The Handler class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Handler.svg"></object>
 * </center>
 */
class MoveHandler(delayed:Boolean, scale:Real) < LazyHandler {
  /**
   * Is delayed sampling enabled?
   */
  delayed:Boolean <- delayed;

  /**
   * Scale of moves.
   */
  scale:Real <- scale;

  final override function handle(event:Event) -> Expression<Real>? {
    /* double dispatch to one of the more specific handle() functions */
    return event.accept(this);
  }

  final override function handle(record:Record, event:Event) ->
      Expression<Real>? {
    /* double dispatch to one of the more specific handle() functions */
    return event.accept(record, this);
  }

  function handle<Value>(event:SimulateEvent<Value>) -> Expression<Real>? {
    if delayed {
      event.p <- event.p.graft();
    }
    event.x <- event.p.value();
    return nil;
  }

  function handle<Value>(event:ObserveEvent<Value>) -> Expression<Real>? {
    if delayed {
      event.p <- event.p.graft();
    }
    return event.p.observeLazy(Boxed(event.x));
  }

  function handle<Value>(event:AssumeEvent<Value>) -> Expression<Real>? {
    if delayed {
      event.p <- event.p.graft();
    }
    if event.x.hasValue() {
      return event.p.observeLazy(event.x);
    } else {
      event.x.assume(event.p);
      return nil;
    }
  }

  function handle(event:FactorEvent) -> Expression<Real>? {
    return event.w;
  }

  function handle<Value>(record:SimulateRecord<Value>,
      event:SimulateEvent<Value>) -> Expression<Real>? {
    if delayed {
      event.p <- event.p.graft();
    }
    event.x <- record.x;
    return nil;
  }

  function handle<Value>(record:ObserveRecord<Value>,
      event:ObserveEvent<Value>) -> Expression<Real>? {
    /* observe events are replayed in the same way they are played, it's
     * only necessary to check that the observed values actually match */
    assert record.x == event.x;
    return handle(event);
  }

  function handle<Value>(record:AssumeRecord<Value>,
      event:AssumeEvent<Value>) -> Expression<Real>? {
    if delayed {
      event.p <- event.p.graft();
    }
    if event.x.hasValue() {
      /* assume events with a value already assigned are replayed in the
       * same way they are played, it's only necessary to check that the
       * observed values actually match */
      assert record.x.hasValue() && record.x.value() == event.x.value();
      return event.p.observeLazy(event.x);
    } else {
      event.x.assume(event.p);    
      if record.x.hasValue() {
        /* if the record has a value, we can set it now, even if its
         * simulation was delayed when originally played; such delays do not
         * change the distribution, only the way it is computed */
        if record.x.dfdx? {
          event.x.setPilot(simulate_propose(record.x.x!, record.x.dfdx!, scale));
        } else {
          event.x.setValue(record.x.value());
        }
      }
      return nil;
    }
  }

  function handle(record:FactorRecord, event:FactorEvent) ->
      Expression<Real>? {
    /* factor events are replayed in the same way they are played */
    return event.w;
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
