/**
 * Event triggered by a *simulate*, typically from the `<~` operator.
 *
 * - p: The distribution.
 */
final class SimulateEvent<Value>(p:Distribution<Value>) <
    ValueEvent<Value> {
  /**
   * Value associated with the event (once simulated).
   */
  v:Value?;
  
  /**
   * Distribution associated with the event.
   */
  p:Distribution<Value> <- p;

  function isSimulate() -> Boolean {
    return true;
  }

  function hasValue() -> Boolean {
    return v?;
  }

  function value() -> Value {
    assert v?;
    return v!;
  }

  function playImmediate() -> Real {
    v <- p.value();
    return 0.0;
  }
    
  function replayImmediate(trace:Queue<Event>) -> Real {
    auto evt <- coerce<Value>(trace);
    v <- p.value(evt.value());
    return 0.0;
  }

  function proposeImmediate(trace:Queue<Event>) -> Real {
    auto evt <- coerce<Value>(trace);
    auto w <- p.observe(evt.value());
    if w == -inf {
      /* hack: in this case the proposal is outside of the support of the 
       * distribution; this can cause later problems in the program (e.g.
       * invalid parameters to subsequent distributions), so simulate
       * something valid to replace this with, but the weight remains -inf */
      v <- p.simulate();
    } else {
      v <- evt.value();
    }
    return w;
  }

  function skipImmediate(trace:Queue<Event>) -> Real {
    coerce<Value>(trace);  // skip
    v <- p.value();
    return 0.0;
  }

  function downdateImmediate(trace:Queue<Event>) -> Real {
    auto evt <- coerce<Value>(trace);
    v <- p.valueWithDowndate(evt.value());
    return 0.0;
  }
  
  function record(trace:Queue<Event>) {
    trace.pushBack(FixedEvent<Value>(v!));
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
