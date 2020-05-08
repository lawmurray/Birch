/**
 * Event triggered by an *observe*, typically from the `~>` operator.
 *
 * - x: Associated value.
 * - p: Associated distribution.
 *
 * The Event class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Event.svg"></object>
 * </center>
 */
final class ObserveEvent<Value>(x:Value, p:Distribution<Value>) < Event {
  /**
   * Value.
   */
  x:Value <- x;
  
  /**
   * Distribution.
   */
  p:Distribution<Value> <- p;

  function record() -> Record {
    return ObserveRecord(x);
  }

  function coerce(record:Record) -> ObserveRecord<Value> {
    auto r <- ObserveRecord<Value>?(record);
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
 * Create an ObserveEvent.
 */
function ObserveEvent<Value>(x:Value, p:Distribution<Value>) ->
    ObserveEvent<Value> {
  evt:ObserveEvent<Value>(x, p);
  return evt;
}
