/**
 * Event handler that collects a trace of events.
 */
class TraceHandler < Handler {
  /**
   * The trace of events.
   */
  trace:List<Event>;

  /**
   * Base handler.
   */
  h:Handler;
  
  function handle(evt:FactorEvent) -> Real {
    trace.pushBack(evt);
    return h.handle(evt);
  }
  
  function handle(evt:RandomEvent) -> Real {
    trace.pushBack(evt);
    return h.handle(evt);
  }
}
