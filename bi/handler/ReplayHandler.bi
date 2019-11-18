/**
 * Event handler that applies the *unplay* action to each event.
 */
final class ReplayHandler < TraceHandler {
  function handle(event:Event, record:Record) -> Real {
    return event.replay(record);
  }
}
