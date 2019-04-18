/**
 * Event triggered by an *observe*, typically from the `~>` operator.
 *
 * - v: The observation.
 * - p: The distribution.
 */
final class ObserveEvent<Value>(v:Value, p:Distribution<Value>) <
    ValueEvent<Value> {
  /**
   * Observation associated with the event.
   */
  v:Value <- v;
  
  /**
   * Distribution associated with the event.
   */
  p:Distribution<Value> <- p;

  function isObserve() -> Boolean {
    return true;
  }
  
  function hasValue() -> Boolean {
    return true;
  }

  function value() -> Value {
    return v;
  }

  function playImmediate() -> Real {
    auto w <- p.observe(v);
    p.update(v);
    return w;
  }
  
  function replayImmediate(trace:Queue<Event>) -> Real {
    auto w <- p.observe(v);
    p.update(v);
    return w;
  }

  function downdateImmediate(trace:Queue<Event>) -> Real {
    auto w <- p.observe(v);
    p.downdate(v);
    return w;
  }
}

/**
 * Create an ObserveEvent.
 */
function ObserveEvent<Value>(v:Value, p:Distribution<Value>) ->
    ObserveEvent<Value> {
  evt:ObserveEvent<Value>(v, p);
  return evt;
}
