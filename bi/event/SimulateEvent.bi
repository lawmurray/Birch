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

  function playMove() -> Real {
    return play();
  }

  function playDelay() -> Real {
    p <- p.graft();
    return play();
  }
  
  function playDelayMove() -> Real {
    p <- p.graft();
    return playMove();
  }

  function replay(record:Record) -> Real {
    auto value <- coerce(record);
    auto w <- p.observe(value);
    if w > -inf {
      v <- value;
      w <- 0.0;
    }
    return w;
  }

  function replayMove(record:Record) -> Real {
    return replay(record);
  }
  
  function replayDelay(record:Record) -> Real {
    p <- p.graft();
    return replay(record);
  }

  function replayDelayMove(record:Record) -> Real {
    p <- p.graft();
    return replayMove(record);
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
