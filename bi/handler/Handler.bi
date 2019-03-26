/**
 * Abstract event handler.
 */
class Handler {
  /**
   * Handle a sequence of events.
   *
   * Returns: Log-weight.
   */
  function handle(evt:Event!) -> Real {
    auto w <- 0.0;
    while evt? {
      w <- w + evt!.accept(this);
    }
    return w;
  }

  /**
   * Handle a factor event.
   *
   * - evt: The event.
   *
   * Returns: Log-weight.
   */
  function handle(evt:FactorEvent) -> Real;
  
  /**
   * Handle a random event.
   *
   * - evt: The event.
   *
   * Returns: Log-weight.
   */
  function handle(evt:RandomEvent) -> Real;

  /**
   * If this is a trace handler, change the base handler.
   */
  function rebase(h:Handler) {
    assert false;
  }
  
  /**
   * If this is a trace handler, change the base handler so as to replay the
   * current trace, while building a new trace.
   */
  function replay() {
    assert false;
  }
}
