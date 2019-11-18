/**
 * Abstract event handler that requires a trace.
 */
abstract class TraceHandler < Handler {
  /**
   * Trace.
   */
  trace:Queue<Record>;
  
  final function handle(event:Event) -> Real {
    return handle(event, trace.popFront());
  }
  
  /**
   * Handle an event.
   *
   * - event: The event.
   * - rec: The front record of the trace.
   *
   * Returns: Log-weight adjustment.
   */
  abstract function handle(event:Event, rec:Record) -> Real;
}
