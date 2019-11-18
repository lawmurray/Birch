/**
 * Event handler that applies the *redelay* action to each event.
 *
 * !!! tip
 *     RedelayHandler is thread-safe, and can be used via the singleton
 *     `redelay`.
 */
final class RedelayHandler < TraceHandler {
  function handle(record:Record, event:Event) -> Real {
    return event.redelay(record);
  }
}

/**
 * Singleton RedelayHandler.
 */
redelay:RedelayHandler;
