/**
 * Event triggered by an *assume*, typically from the `~` operator.
 *
 * - v: The random variate.
 * - p: The distribution.
 */
final class AssumeEvent<Value>(v:Random<Value>, p:Distribution<Value>) <
    ValueEvent<Value> {
  /**
   * Random variable associated with the event.
   */
  v:Random<Value> <- v;
  
  /**
   * Distribution associated with the event.
   */
  p:Distribution<Value> <- p;

  function isAssume() -> Boolean {
    return true;
  }
  
  function hasValue() -> Boolean {
    return v.hasValue();
  }
  
  function value() -> Value {
    assert hasValue();
    return v.value();
  }

  function playImmediate() -> Real {
    auto w <- 0.0;
    if v.hasValue() {
      w <- p.observe(v.value());
    } else {
      v <- p.simulate();
    }
    p.update(v.value());
    p.detach();
    return w;
  }

  function playDelay() -> Real {
    auto w <- 0.0;
    if v.hasValue() {
      w <- p.observe(v.value());
      p.update(v.value());
      p.detach();
    } else {
      v.assume(p);
    }
    return w;
  }
  
  function replayImmediate(trace:Queue<Event>) -> Real {
    auto w <- 0.0;
    auto evt <- coerce<Value>(trace);
    if v.hasValue() {
      assert v.value() == evt.value();
      w <- p.observe(evt.value());
    } else {
      v <- evt.value();
    }
    p.update(evt.value());
    p.detach();
    return w;
  }

  function replayDelay(trace:Queue<Event>) -> Real {
    auto w <- 0.0;
    auto evt <- coerce<Value>(trace);
    if v.hasValue() {
      assert v.value() == evt.value();
      w <- p.observe(evt.value());
      p.update(evt.value());
      p.detach();
    } else {
      if evt.hasValue() {
        v.assumeUpdate(p, evt.value());
      } else {
        v.assume(p);
      }
    }
    return w;
  }

  function downdateImmediate(trace:Queue<Event>) -> Real {
    auto w <- 0.0;
    auto evt <- coerce<Value>(trace);
    if v.hasValue() {
      assert v.value() == evt.value();
      w <- p.observe(evt.value());
    } else {
      v <- evt.value();
    }
    p.downdate(evt.value());
    p.detach();
    return w;
  }

  function downdateDelay(trace:Queue<Event>) -> Real {
    auto w <- 0.0;
    auto evt <- coerce<Value>(trace);
    if v.hasValue() {
      assert v.value() == evt.value();
      w <- p.observe(evt.value());
      p.downdate(evt.value());
      p.detach();
    } else {
      if evt.hasValue() {
        v.assumeDowndate(p, evt.value());
      } else {
        v.assume(p);
      }
    }
    return w;
  }

  function record(trace:Queue<Event>) {
    trace.pushBack(RandomEvent<Value>(v));
  }
}

/**
 * Create an AssumeEvent.
 */
function AssumeEvent<Value>(v:Random<Value>, p:Distribution<Value>) ->
    AssumeEvent<Value> {
  evt:AssumeEvent<Value>(v, p);
  return evt;
}
