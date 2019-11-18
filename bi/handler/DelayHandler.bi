/**
 * Event handler that applies the *delay* action to each event.
 *
 * !!! tip
 *     DelayHandler is thread-safe, and can be used via the singleton `delay`.
 */
final class DelayHandler < Handler {
  function handle(event:Event) -> Real {
    return event.delay();
  }
}

/**
 * Singleton DelayHandler.
 */
delay:DelayHandler;
