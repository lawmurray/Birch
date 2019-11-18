/**
 * Event handler that applies the *propoose* action to each event.
 *
 * !!! tip
 *     ProposeHandler is thread-safe, and can be used via the singleton
 *     `propose`.
 */
final class ProposeHandler < TraceHandler {
  function handle(record:Record, event:Event) -> Real {
    return event.propose(record);
  }
}

/**
 * Singleton ProposeHandler.
 */
propose:ProposeHandler;
