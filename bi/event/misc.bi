/**
 * Coerce a value out of an event trace. This tries to cast the first event
 * in the trace to ValueEvent and return it.
 */
function coerce<Value>(trace:Queue<Event>) -> ValueEvent<Value>? {
  auto r <- ValueEvent<Value>?(trace.front());
  if r? {
    trace.popFront();
    return r;
  } else {
    return nil;
  }
}
