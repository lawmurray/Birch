/**
 * Event triggered by an *assume*, typically from the `~` operator.
 *
 * - v: The random variate.
 * - p: The distribution.
 */
final class AssumeEvent<Value>(v:Random<Value>, p:Distribution<Value>) <
    ValueEvent<Value> {
  /**
   * Random variate associated with the event.
   */
  v:Random<Value> <- v;
  
  /**
   * Distribution associated with the event.
   */
  p:Distribution<Value> <- p;

  /**
   * Did the random variate have a value when the event was triggered?
   */
  assigned:Boolean <- v.hasValue();

  function hasValue() -> Boolean {
    return true;
  }
  
  function value() -> Value {
    return v.value();
  }

  function isAssume() -> Boolean {
    return true;
  }
  
  function playImmediate() -> Real {
    auto w <- 0.0;
    if assigned {
      w <- p.observe(v.value());
    } else {
      v <- p.value();
    }
    return w;
  }

  function playDelay() -> Real {
    auto w <- 0.0;
    if assigned {
      w <- p.observe(v.value());
    } else {
      p.assume(v);
    }
    return w;
  }
  
  function skipImmediate(trace:Queue<Record>) -> Real {
    if !assigned {
      coerce<Value>(trace);
    }
    return playImmediate();
  }

  function skipDelay(trace:Queue<Record>) -> Real {
    if !assigned {
      coerce<Value>(trace);
    }
    return playDelay();
  }

  function replayImmediate(trace:Queue<Record>) -> Real {
    auto w <- 0.0;
    if assigned {
      w <- p.observe(v.value());
    } else {
      auto r <- coerce<Value>(trace);
      w <- p.observe(r.value());
      if w != -inf {
        v <- r.value();
        w <- 0.0;
      }
    }
    return w;
  }

  function replayDelay(trace:Queue<Record>) -> Real {
    auto w <- 0.0;
    if assigned {
      w <- p.observe(v.value());
    } else {
      auto r <- coerce<Value>(trace);      
      p.assume(v, r.value());
    }
    return w;
  }

  function downdateImmediate(trace:Queue<Record>) -> Real {
    auto w <- 0.0;
    if assigned {
      w <- p.observe(v.value());
    } else {
      auto r <- coerce<Value>(trace);
      w <- p.observeWithDowndate(r.value());
      if w != -inf {
        v <- r.value();
        w <- 0.0;
      }
    }
    return w;
  }

  function downdateDelay(trace:Queue<Record>) -> Real {
    auto w <- 0.0;
    if assigned {
      w <- p.observeWithDowndate(v.value());
    } else {
      auto r <- coerce<Value>(trace);      
      p.assumeWithDowndate(v, r.value());
    }
    return w;
  }

  function proposeImmediate(trace:Queue<Record>) -> Real {
    auto w <- 0.0;
    if assigned {
      w <- p.observe(v.value());
    } else {
      auto r <- coerce<Value>(trace);
      w <- p.observe(r.value());
      if w != -inf {
        v <- r.value();
      }
    }
    return w;
  }

  function record(trace:Queue<Record>) {
    if !assigned {
      trace.pushBack(RandomRecord<Value>(v));
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
