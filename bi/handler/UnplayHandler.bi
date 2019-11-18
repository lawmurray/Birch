/**
 * Event handler that applies the *unplay* action to each event.
 *
 * - trace: The trace.
 */
final class UnplayHandler(trace:Trace) < TraceHandler(trace) {
  function handle(event:Event, record:Record) -> Real {
    return event.unplay(record);
  }
}

/**
 * Create an UnplayHandler.
 */
function UnplayHandler(trace:Trace) -> UnplayHandler {
  o:UnplayHandler(trace);
  return o;
}
