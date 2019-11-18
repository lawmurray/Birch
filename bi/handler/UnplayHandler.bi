/**
 * Event handler that applies the *unplay* action to each event.
 *
 * !!! tip
 *     UnplayHandler is thread-safe, and can be used via the singleton
 *     `unplay`.
 */
final class UnplayHandler < TraceHandler {
  function handle(record:Record, event:Event) -> Real {
    return event.unplay(record);
  }
}

/**
 * Singleton UnplayHandler.
 */
unplay:UnplayHandler;
