/**
 * Event handler that applies the *unplay* action to each event.
 */
final class ReplayHandler(trace:Trace) < TraceHandler(trace) {
  function handle(event:Event, record:Record) -> Real {
    return event.replay(record);
  }
}

/**
 * Create a ReplayHandler.
 */
function ReplayHandler(trace:Trace) -> ReplayHandler {
  o:ReplayHandler(trace);
  return o;
}
