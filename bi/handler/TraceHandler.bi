/**
 * Abstract event handler that requires a trace.
 *
 * - trace: The trace.
 */
abstract class TraceHandler(trace:Trace) < Handler {
  /**
   * Trace.
   */
  trace:Trace <- trace;
  
  final function handle(event:Event) -> Real {
    return handle(event, trace.popFront());
  }
  
  /**
   * Handle an event.
   *
   * - event: The event.
   * - record: The front record of the trace.
   *
   * Returns: Log-weight adjustment.
   */
  abstract function handle(event:Event, record:Record) -> Real;
}
