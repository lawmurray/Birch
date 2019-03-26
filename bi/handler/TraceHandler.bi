/**
 * Event handler that collects a trace of events.
 */
class TraceHandler(h:Handler) < Handler {
  /**
   * The trace of events.
   */
  trace:List<Event>;

  /**
   * Base handler.
   */
  h:Handler <- h;
  
  function handle(evt:FactorEvent) -> Real {
    trace.pushBack(evt);
    return h.handle(evt);
  }
  
  function handle(evt:RandomEvent) -> Real {
    trace.pushBack(evt);
    return h.handle(evt);
  }
  
  function rebase(h:Handler) {
    this.h <- h;
  }
  
  function replay() {
    h <- ReplayHandler(trace);
    o:List<Event>;
    trace <- o;
  }
}

function TraceHandler(h:Handler) -> TraceHandler {
  o:TraceHandler(h);
  return o;
}
