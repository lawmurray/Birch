/**
 * Event handler that applies the *delay* action to each event.
 */
final class DelayHandler < Handler {
  function handle(event:Event) -> Real {
    return event.delay();
  }
}

/**
 * Create a DelayHandler.
 */
function DelayHandler() -> DelayHandler {
  o:DelayHandler;
  return o;
}
