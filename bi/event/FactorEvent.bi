/**
 * Event triggered for a factor.
 *
 * - w: Log-weight.
 */
class FactorEvent(w:Real) < Event {
  /**
   * Log-weight associated with the event.
   */
  w:Real <- w;

  function isFactor() -> Boolean {
    return true;
  }

  function observe() -> Real {
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
