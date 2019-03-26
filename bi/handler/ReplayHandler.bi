/**
 * Event handler that replays a trace of events.
 */
class ReplayHandler < Handler {
  /**
   * The trace of events.
   */
  trace:List<Event>;

  function handle(evt:FactorEvent) -> Real {

  }
  
  function handle(evt:RandomEvent) -> Real {

  }
}

function ReplayHandler() -> ReplayHandler {
  o:ReplayHandler;
  return o;
}
