/**
 * Event handler that eagerly instantiates random variates.
 */
class EagerHandler < Handler {
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

function EagerHandler() -> EagerHandler {
  o:EagerHandler;
  return o;
}
