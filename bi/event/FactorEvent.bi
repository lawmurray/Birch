/**
 * Event triggered for a factor.
 *
 * - w: Log-weight.
 */
final class FactorEvent(w:Real) < Event {
  /**
   * Log-weight associated with the event.
   */
  w:Real <- w;

  function isFactor() -> Boolean {
    return true;
  }

  function playImmediate() -> Real {
    return w;
  }
    
  function replayImmediate(trace:Queue<Record>) -> Real {
    return w;
  }

  function proposeImmediate(trace:Queue<Record>) -> Real {
    return w;
  }

  function skipImmediate(trace:Queue<Record>) -> Real {
    return w;
  }

  function downdateImmediate(trace:Queue<Record>) -> Real {
    return w;
  }
}

/**
 * Create a FactorEvent.
 */
function FactorEvent(w:Real) -> FactorEvent {
  evt:FactorEvent(w);
  return evt;
}
