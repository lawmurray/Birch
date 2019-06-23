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
      v <- p.value();
    }
    return w;
  }

  function playDelay() -> Real {
    auto w <- 0.0;
    if v.hasValue() {
      w <- p.observe(v.value());
    } else {
      p.assume(v);
    }
    return w;
  }
    
  function replayImmediate(trace:Queue<Event>) -> Real {
    auto w <- 0.0;
    auto evt <- coerce<Value>(trace);
    if v.hasValue() {
      //assert evt.hasValue() && v.value() == evt.value();
      w <- p.observe(evt.value());
    } else {
      v <- p.set(evt.value());
    }
    return w;
  }

  function replayDelay(trace:Queue<Event>) -> Real {
    auto w <- 0.0;
    auto evt <- coerce<Value>(trace);
    if v.hasValue() {
      //assert evt.hasValue() && v.value() == evt.value();
      w <- p.observe(evt.value());
    } else {
      p.assume(v, evt.value());
    }
    return w;
  }
  
  function skipImmediate(trace:Queue<Event>) -> Real {
    coerce<Value>(trace);
    return playImmediate();
  }

  function skipDelay(trace:Queue<Event>) -> Real {
    coerce<Value>(trace);
    return playDelay();
  }

  function downdateImmediate(trace:Queue<Event>) -> Real {
    auto w <- 0.0;
    auto evt <- coerce<Value>(trace);
    if v.hasValue() {
      //assert evt.hasValue() && v.value() == evt.value();
      w <- p.observeWithDowndate(evt.value());
    } else {
      v <- p.setWithDowndate(evt.value());
    }
    return w;
  }

  function downdateDelay(trace:Queue<Event>) -> Real {
    auto w <- 0.0;
    auto evt <- coerce<Value>(trace);
    if v.hasValue() {
      //assert evt.hasValue() && v.value() == evt.value();
      w <- p.observeWithDowndate(evt.value());
    } else {
      p.assumeWithDowndate(v, evt.value());
    }
    return w;
  }

  function proposeImmediate(trace:Queue<Event>) -> Real {
    auto w <- 0.0;
    auto evt <- coerce<Value>(trace);
    if v.hasValue() {
      //assert evt.hasValue() && v.value() == evt.value();
      w <- p.observe(evt.value());
    } else {
      w <- p.observe(evt.value());
      if w == -inf {
        /* hack: in this case the proposal is outside of the support of the 
         * distribution; this can cause later problems in the program (e.g.
         * invalid parameters to subsequent distributions), so simulate
         * something valid to replace this with, but the weight remains
         * -inf */
        v <- p.simulate();
      } else {
        v <- evt.value();
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
