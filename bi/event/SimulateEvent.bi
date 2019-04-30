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
  
  function skipImmediate() -> Real {
    coerce<Value>(trace);
    return playImmediate();
  }
  
  function replayImmediate(trace:Queue<Event>) -> Real {
    auto evt <- coerce<Value>(trace);
    v <- evt.value();
    p.update(v!);
    p.detach();
    return 0.0;
  }

  function downdateImmediate(trace:Queue<Event>) -> Real {
    auto evt <- coerce<Value>(trace);
    v <- evt.value();
    p.downdate(v!);
    p.detach();
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
