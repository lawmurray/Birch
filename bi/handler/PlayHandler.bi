/**
 * Event handler that applies the *play* action to each event.
 */
class PlayHandler < Handler {
  function handle(event:Event) -> Real {
    return event.play();
  }
}

/**
 * Create a PlayHandler.
 */
function PlayHandler() -> PlayHandler {
  o:PlayHandler;
  return o;
}
