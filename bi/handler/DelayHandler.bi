/**
 * Event handler that delays instantiation of random variates.
 */
class DelayHandler < Handler {
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

function DelayHandler() -> DelayHandler {
  o:DelayHandler;
  return o;
}
