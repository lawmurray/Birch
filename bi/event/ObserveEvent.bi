/**
 * Event triggered by an *observe*, typically from the `~>` operator.
 *
 * - v: The observation.
 * - p: The distribution.
 */
final class ObserveEvent<Value>(v:Value, p:Distribution<Value>) < Event {
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
  
  function playImmediate() -> Real {
    return p.observe(v);
  }
  
  function replayImmediate(trace:Queue<Record>) -> Real {
    return p.observe(v);
  }

  function skipImmediate(trace:Queue<Record>) -> Real {
    return p.observe(v);
  }

  function downdateImmediate(trace:Queue<Record>) -> Real {
    return p.observeWithDowndate(v);
  }

  function proposeImmediate(trace:Queue<Record>) -> Real {
    return p.observe(v);
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
