/**
 * *Assume* event triggered during the execution of a model, typically by
 * the `~` operator.
 *
 * - x: Associated random variate.
 * - p: Associated distribution.
 *
 * The Event class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Event.svg"></object>
 * </center>
 */
final class AssumeEvent<Value>(x:Random<Value>, p:Distribution<Value>) <
    Event {
  /**
   * Random variate.
   */
  x:Random<Value> <- x;
  
  /**
   * Distribution.
   */
  p:Distribution<Value> <- p;

  function record() -> Record {
    return AssumeRecord(x);
  }
  
  function coerce(record:Record) -> AssumeRecord<Value> {
    auto r <- AssumeRecord<Value>?(record);
    if !r? {
      error("incompatible trace");
    }
    return r!;
  }

  function accept(handler:PlayHandler) -> Real {
    return handler.handle(this);
  }
  
  function accept(handler:MoveHandler) -> Real {
    return handler.handle(this);
  }

  function accept(record:Record, handler:PlayHandler) -> Real {
    return handler.handle(coerce(record), this);
  }

  function accept(record:Record, handler:MoveHandler) -> Real {
    return handler.handle(coerce(record), this);
  }
}

/**
 * Create an AssumeEvent.
 */
function AssumeEvent<Value>(x:Random<Value>, p:Distribution<Value>) ->
    AssumeEvent<Value> {
  return construct<AssumeEvent<Value>>(x, p);
}
