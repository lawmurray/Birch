/**
 * Event triggered by a *simulate*, typically from the `<~` operator.
 *
 * - p: Associated distribution.
 */
final class SimulateEvent<Value>(p:Distribution<Value>) < ValueEvent<Value> {
  /**
   * Associated value.
   */
  v:Value?;
  
  /**
   * Associated distribution.
   */
  p:Distribution<Value> <- p;

  function hasValue() -> Boolean {
    return v?;
  }
  
  function value() -> Value {
    assert v?;
    return v!;
  }

  function play() -> Real {
    v <- p.value();
    return 0.0;
  }

  function replay(record:Record) -> Real {
    auto value <- coerce(record);
    if p.observe(value) > -inf {
      v <- value;
    }
    return 0.0;
  }

  function delay() -> Real {
    return play();
  }
  
  function redelay(record:Record) -> Real {
    return replay(record);
  }
  
  function propose(record:Record) -> Real {
    auto value <- coerce(record);
    if p.observe(value) > -inf {
      v <- value;
    }
    return 0.0;
  }

  function record() -> Record {
    return ImmediateRecord<Value>(v!);
  }
}

/**
 * Create a SimulateEvent.
 */
function SimulateEvent<Value>(p:Distribution<Value>) ->
    SimulateEvent<Value> {
  evt:SimulateEvent<Value>(p);
  return evt;
}
