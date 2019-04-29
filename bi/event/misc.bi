/**
 * Coerce a value out of an event trace. This tries to cast the first event
 * in the trace to ValueEvent and return it.
 */
function coerce<Value>(trace:Queue<Event>) -> ValueEvent<Value> {
  if !trace.empty() {
    auto r <- ValueEvent<Value>?(trace.popFront());
    if r? {
      return r!;
    }
  }
  error("incompatible trace");
}
