/**
 * Event triggered by an *assume*, typically from the `~` operator.
 *
 * - v: Associated random variate.
 * - p: Associated distribution.
 */
final class AssumeEvent<Value>(v:Random<Value>, p:Distribution<Value>) <
    ValueEvent<Value> {
  /**
   * Associated random variate.
   */
  v:Random<Value> <- v;
  
  /**
   * Associated distribution.
   */
  p:Distribution<Value> <- p;

  function hasValue() -> Boolean {
    return true;
  }
  
  function value() -> Value {
    return v.value();
  }
  
  function play() -> Real {
    auto w <- 0.0;
    if v.hasValue() {
      w <- p.observe(v.value());
    } else {
      v <- p.value();
    }
    return w;
  }

  function replay(record:Record) -> Real {
    auto w <- 0.0;
    auto value <- coerce(record);
    if v.hasValue() {
      assert v.value() == value;
      w <- p.observe(value);
    } else {
      w <- p.observe(value);
      if w != -inf {
        v <- value;
        w <- 0.0;
      }
    }
    return w;
  }

  function unplay(record:Record) -> Real {
    auto w <- 0.0;
    auto value <- coerce(record);
    if v.hasValue() {
      assert v.value() == value;
      w <- p.observe(value);
    } else {
      w <- p.observeWithDowndate(value);
      if w != -inf {
        v <- value;
        w <- 0.0;
      }
    }
    return w;
  }

  function delay() -> Real {
    auto w <- 0.0;
    if v.hasValue() {
      w <- p.observe(v.value());
    } else {
      p.assume(v);
    }
    return w;
  }

  function redelay(record:Record) -> Real {
    auto w <- 0.0;
    auto value <- coerce(record);
    if v.hasValue() {
      assert v.value() == value;
      w <- p.observe(value);
    } else {
      p.assume(v, value);
    }
    return w;
  }

  function undelay(record:Record) -> Real {
    auto w <- 0.0;
    auto value <- coerce(record);
    if v.hasValue() {
      assert v.value() == value;
      w <- p.observeWithDowndate(value);
    } else {
      p.assumeWithDowndate(v, value);
    }
    return w;
  }

  function propose(record:Record) -> Real {
    auto w <- 0.0;
    auto value <- coerce(record);
    if v.hasValue() {
      assert v.value() == value;
      w <- p.observe(value);
    } else {
      w <- p.observe(value);
      if w != -inf {
        v <- value;
      }
    }
    return w;
  }
  
  function record() -> Record {
    if v.hasValue() {
      return ImmediateRecord(v);
    } else {
      return DelayRecord(v);
    }
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
