/**
 * Event handler that records a trace of events from another handler.
 *
 * - BaseHandler: Type of the other handler.
 */
class TraceHandler<BaseHandler> < BaseHandler {
  /**
   * Recorded trace of events.
   */
  record:List<Event>;
  
  function handle(evt:FactorEvent) -> Real {
    trace.pushBack(evt);
    return h.handle(evt);
  }
  
  function handle(evt:RandomEvent) -> Real {
    trace.pushBack(evt);
    return h.handle(evt);
  }
}
