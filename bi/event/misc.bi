/**
 * Coerce a value out of a trace. This tries to cast the first record
 * in the trace to ValueRecord and return it.
 */
function coerce<Value>(trace:Queue<Record>) -> ValueRecord<Value> {
  r:ValueRecord<Value>?;
  if !trace.empty() {
    r <- ValueRecord<Value>?(trace.popFront());
  }
  if !r? {
    error("incompatible trace");
  }
  cpp{{
  return std::move(r.get());
  }}
}
