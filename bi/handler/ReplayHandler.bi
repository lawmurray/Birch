/**
 * Event handler that replays a trace of events.
 */
class ReplayHandler(trace:List<Event>) < Handler {
  /**
   * The trace of events.
   */
  trace:List<Event> <- trace;

  function handle(evt:FactorEvent) -> Real {
    trace.popFront();
  }
  
  function handle(evt:RandomEvent) -> Real {
    trace.popFront();
  }
}

function ReplayHandler(trace:List<Event>) -> ReplayHandler {
  o:ReplayHandler(trace);
  return o;
}
