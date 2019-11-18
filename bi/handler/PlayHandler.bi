/**
 * Event handler that applies the *play* action to each event.
 *
 * !!! tip
 *     PlayHandler is thread-safe, and can be used via the singleton `play`.
 */
class PlayHandler < Handler {
  function handle(event:Event) -> Real {
    return event.play();
  }
}

/**
 * Singleton PlayHandler.
 */
play:PlayHandler;
