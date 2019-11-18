/**
 * Event handler that applies the *undelay* action to each event.
 *
 * !!! tip
 *     UndelayHandler is thread-safe, and can be used via the singleton
 *     `undelay`.
 */
final class UndelayHandler < TraceHandler {
  function handle(record:Record, event:Event) -> Real {
    return event.undelay(record);
  }
}

/**
 * Singleton UndelayHandler.
 */
undelay:UndelayHandler;
