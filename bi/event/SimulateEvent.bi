/**
 * Event triggered by a *simulate*, typically from the `<~` operator.
 *
 * - p: The distribution.
 */
final class SimulateEvent<Value>(p:Distribution<Value>) < ValueEvent<Value> {
  /**
   * Value associated with the event (once simulated).
   */
  v:Value?;
  
  /**
   * Distribution associated with the event.
   */
  p:Distribution<Value> <- p;

  function hasValue() -> Boolean {
    return v?;
  }
  
  function value() -> Value {
    assert v?;
    return v!;
  }

  function isSimulate() -> Boolean {
    return true;
  }

  function playImmediate() -> Real {
    v <- p.value();
    return 0.0;
  }

  function skipImmediate(trace:Queue<Record>) -> Real {
    coerce<Value>(trace);  // skip
    v <- p.value();
    return 0.0;
  }

  function replayImmediate(trace:Queue<Record>) -> Real {
    auto r <- coerce<Value>(trace);
    auto w <- p.observe(r.value());
    if w != -inf {
      v <- r.value();
      w <- 0.0;
    }
    return w;
  }

  function downdateImmediate(trace:Queue<Record>) -> Real {
    auto r <- coerce<Value>(trace);
    auto w <- p.observeWithDowndate(r.value());
    if w != -inf {
      v <- r.value();
      w <- 0.0;
    }
    return w;
  }
  
  function proposeImmediate(trace:Queue<Record>) -> Real {
    auto r <- coerce<Value>(trace);
    auto w <- p.observe(r.value());
    if w != -inf {
      v <- r.value();
    }
    return w;
  }
  
  function record(trace:Queue<Record>) {
    trace.pushBack(FixedRecord<Value>(v!));
  }
}

/**
 * Create a SimulateEvent.
 */
function SimulateEvent<Value>(p:Distribution<Value>) ->
    SimulateEvent<Value> {
  evt:SimulateEvent<Value>(p);
  return evt;
}
