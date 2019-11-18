/**
 * Event handler that applies the *unplay* action to each event.
 */
final class UnplayHandler < TraceHandler {
  function handle(event:Event, record:Record) -> Real {
    return event.unplay(record);
  }
}
