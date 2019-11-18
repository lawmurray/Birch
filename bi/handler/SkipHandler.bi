/**
 * Event handler that skips events in trace.  This is followed
 * by another event handler that then applies actions to the events.
 *
 * - trace: The trace.
 * - h: The following handler.
 */
final class SkipHandler(trace:Trace, h:Handler) < TraceHandler(trace) {
  /**
   * Following handler.
   */
  h:Handler <- h;

  function handle(event:Event, record:Record) -> Real {
    return h.handle(event);
  }
}
