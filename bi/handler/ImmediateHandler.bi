/**
 * Immediate-mode event handler. This:
 *
 *   * immediately evaluates random events with values as observations, and
 *   * immediately evaluates random events without values as simulations.
 */
class ImmediateHandler < Handler {
  function handle(evt:FactorEvent) -> Real {
    return evt.observe();
  }
  
  function handle(evt:RandomEvent) -> Real {
    if evt.hasValue() {
      return evt.observe();
    } else {
      evt.value();
      return 0.0;
    }
  }
}

function ImmediateHandler() -> ImmediateHandler {
  o:ImmediateHandler;
  return o;
}
