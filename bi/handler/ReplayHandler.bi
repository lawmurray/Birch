/**
 * Event handler that applies the *unplay* action to each event.
 *
 * !!! tip
 *     ReplayHandler is thread-safe, and can be used via the singleton
 *     `replay`.
 */
final class ReplayHandler < TraceHandler {
  function handle(record:Record, event:Event) -> Real {
    return event.replay(record);
  }
}

/**
 * Singleton ReplayHandler.
 */
replay:ReplayHandler;
