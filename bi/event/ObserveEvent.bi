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
    if w > -inf {
      p.update(v);
    }
    p.detach();
    return w;
  }
  
  function replayImmediate(trace:Queue<Event>) -> Real {
    auto w <- p.observe(v);
    if w > -inf {
      p.update(v);
    }
    p.detach();
    return w;
  }

  function proposeImmediate(trace:Queue<Event>) -> Real {
    return replayImmediate(trace);
  }

  function skipImmediate(trace:Queue<Event>) -> Real {
    return playImmediate();
  }

  function downdateImmediate(trace:Queue<Event>) -> Real {
    auto w <- p.observe(v);
    if w > -inf {
      p.downdate(v);
    }
    p.detach();
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
