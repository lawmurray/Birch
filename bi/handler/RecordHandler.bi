/**
 * Event handler that records events into trace. This is preceded
 * by another event handler that first applies actions to the events.
 *
 * - h: The preceding handler.
 */
final class RecordHandler(h:Handler) < Handler {
  /**
   * Preceding handler.
   */
  h:Handler <- h;

  /**
   * Trace.
   */
  trace:Trace;
  
  function handle(event:Event) -> Real {
    auto w <- h.handle(event);
    trace.pushBack(event.record());
    return w;
  }
}

/**
 * Create a RecordHandler.
 */
function RecordHandler(h:Handler) -> RecordHandler {
  o:RecordHandler(h);
  return o;
}
