/**
 * Event handler that records events into trace. This is typically preceded
 * by another event handler that first applies actions to the events.
 */
final class RecordHandler < TraceHandler {
  function handle(event:Event) -> Real {
    trace.pushBack(event.record());
  }
}
