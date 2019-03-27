/**
 * Delayed-mode event handler. This is the event handler used for delayed
 * sampling. It:
 *
 *   * immediately evaluates random events with values as observations, but
 *   * delays evaluation of random events without values.
 */
class DelayedHandler < Handler {
  function handle(evt:FactorEvent) -> Real {
    return evt.observe();
  }
  
  function handle(evt:RandomEvent) -> Real {
    if evt.hasValue() {
      return evt.observe();
    } else {
      evt.assume();
      return 0.0;
    }
  }
}

function DelayedHandler() -> DelayedHandler {
  o:DelayedHandler;
  return o;
}
