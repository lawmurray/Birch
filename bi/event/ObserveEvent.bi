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
  
  function replay(record:Record) -> Real {
    assert v.value() == coerce(record);
    return p.observe(v);
  }

  function unplay(record:Record) -> Real {
    assert v.value() == coerce(record);
    return p.observeWithDowndate(v);
  }
  
  function delay() -> Real {
    return play();
  }

  function redelay() -> Real {
    return replay();
  }

  function undelay() -> Real {
    return unplay();
  }

  function propose(record:Record) -> Real {
    if record.hasValue() {
      assert v.value() == coerce(record);
      return p.observe(v);
    } else {
      return 0.0;
    }
  }

  function record() -> Record {
    return ImmediateRecord<Value>(v);
  }
}

/**
 * Create an ObserveEvent.
 */
function ObserveEvent<Value>(v:Value, p:Distribution<Value>) ->
    ObserveEvent<Value> {
  evt:ObserveEvent<Value>(v, p);
  return evt;
}
