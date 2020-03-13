/**
 * Event triggered by an *observe*, typically from the `~>` operator.
 *
 * - v: Associated observation.
 * - p: Associated distribution.
 */
final class ObserveEvent<Value>(v:Value, p:Distribution<Value>) <
    ValueEvent<Value> {
  /**
   * Observation associated with the event.
   */
  v:Value <- v;
  
  /**
   * Distribution associated with the event.
   */
  p:Distribution<Value> <- p;

  function hasValue() -> Boolean {
    return true;
  }
  
  function value() -> Value {
    return v;
  }
  
  function play() -> Real {
    return p.observe(v);
  }
  
  function playMove() -> Real {
    return p.observeLazy(Boxed(v));
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
    return play();
  }

  function replayDelay(record:Record) -> Real {
    return playDelay();
  }

  function replayMove(record:Record, scale:Real) -> Real {
    return playMove();
  }
  
  function replayDelayMove(record:Record, scale:Real) -> Real {
    return playDelayMove();
  }

  function record() -> Record {
    return ImmediateRecord<Value>(v);
  }
}

/**
 * Create an ObserveEvent.
 */
function ObserveEvent(v:Real, p:Distribution<Real>) -> ObserveEvent<Real> {
  evt:ObserveEvent<Real>(v, p);
  return evt;
}

/**
 * Create an ObserveEvent.
 */
function ObserveEvent(v:Real[_], p:Distribution<Real[_]>) ->
    ObserveEvent<Real[_]> {
  evt:ObserveEvent<Real[_]>(v, p);
  return evt;
}

/**
 * Create an ObserveEvent.
 */
function ObserveEvent(v:Real[_,_], p:Distribution<Real[_,_]>) ->
    ObserveEvent<Real[_,_]> {
  evt:ObserveEvent<Real[_,_]>(v, p);
  return evt;
}

/**
 * Create an ObserveEvent.
 */
function ObserveEvent(v:Integer, p:Distribution<Integer>) ->
    ObserveEvent<Integer> {
  evt:ObserveEvent<Integer>(v, p);
  return evt;
}

/**
 * Create an ObserveEvent.
 */
function ObserveEvent(v:Integer[_], p:Distribution<Integer[_]>) ->
    ObserveEvent<Integer[_]> {
  evt:ObserveEvent<Integer[_]>(v, p);
  return evt;
}

/**
 * Create an ObserveEvent.
 */
function ObserveEvent(v:Boolean, p:Distribution<Boolean>) ->
    ObserveEvent<Boolean> {
  evt:ObserveEvent<Boolean>(v, p);
  return evt;
}
