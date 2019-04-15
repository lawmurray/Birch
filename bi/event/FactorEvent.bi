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

  function accept(h:EventHandler) -> Real {
    return h.handle(this);
  }

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
