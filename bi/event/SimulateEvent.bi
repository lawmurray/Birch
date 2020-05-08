/**
 * Event triggered by a *simulate*, typically from the `<~` operator.
 *
 * - p: Associated distribution.
 *
 * The Event class hierarchy is as follows:
 * <center>
 * <object type="image/svg+xml" data="../../figs/Event.svg"></object>
 * </center>
 */
final class SimulateEvent<Value>(p:Distribution<Value>) < Event {
  /**
   * Value.
   */
  x:Value?;
  
  /**
   * Distribution.
   */
  p:Distribution<Value> <- p;

  function value() -> Value {
    return x!;
  }

  function record() -> Record {
    return SimulateRecord(x!);
  }
  
  function coerce(record:Record) -> SimulateRecord<Value> {
    auto r <- SimulateRecord<Value>?(record);
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
 * Create a SimulateEvent.
 */
function SimulateEvent<Value>(p:Distribution<Value>) -> SimulateEvent<Value> {
  evt:SimulateEvent<Value>(p);
  return evt;
}
