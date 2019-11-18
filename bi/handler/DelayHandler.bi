/**
 * Event handler that applies the *delay* action to each event.
 */
final class DelayHandler < Handler {
  function handle(event:Event) -> Real {
    return event.delay();
  }
}
