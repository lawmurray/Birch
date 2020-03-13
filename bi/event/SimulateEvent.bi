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

  function replayMove(record:Record, scale:Real) -> Real {
    return replay(record);
  }
  
  function replayDelay(record:Record) -> Real {
    p <- p.graft();
    return replay(record);
  }

  function replayDelayMove(record:Record, scale:Real) -> Real {
    p <- p.graft();
    return replayMove(record, scale);
  }

  function record() -> Record {
    return ImmediateRecord<Value>(v!);
  }
}

/**
 * Create a SimulateEvent.
 */
function SimulateEvent(p:Distribution<Real>) -> SimulateEvent<Real> {
  evt:SimulateEvent<Real>(p);
  return evt;
}

/**
 * Create a SimulateEvent.
 */
function SimulateEvent(p:Distribution<Real[_]>) -> SimulateEvent<Real[_]> {
  evt:SimulateEvent<Real[_]>(p);
  return evt;
}

/**
 * Create a SimulateEvent.
 */
function SimulateEvent(p:Distribution<Real[_,_]>) -> SimulateEvent<Real[_,_]> {
  evt:SimulateEvent<Real[_,_]>(p);
  return evt;
}

/**
 * Create a SimulateEvent.
 */
function SimulateEvent(p:Distribution<Integer>) -> SimulateEvent<Integer> {
  evt:SimulateEvent<Integer>(p);
  return evt;
}

/**
 * Create a SimulateEvent.
 */
function SimulateEvent(p:Distribution<Integer[_]>) -> SimulateEvent<Integer[_]> {
  evt:SimulateEvent<Integer[_]>(p);
  return evt;
}

/**
 * Create a SimulateEvent.
 */
function SimulateEvent(p:Distribution<Boolean>) -> SimulateEvent<Boolean> {
  evt:SimulateEvent<Boolean>(p);
  return evt;
}
