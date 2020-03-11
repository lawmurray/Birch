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
      p.value();
      v.set(p);
    }
    return w;
  }

  function playMove() -> Real {
    auto w <- 0.0;
    if v.hasValue() {
      w <- p.observeLazy(v);
    } else {
      p.value();
      v.set(p);
    }
    return w;
  }

  function playDelay() -> Real {
    auto w <- 0.0;
    if v.hasValue() {
      p <- p.graft();
      w <- p.observe(v.value());
    } else {
      p.assume(v);
    }
    return w;
  }

  function playDelayMove() -> Real {
    auto w <- 0.0;
    if v.hasValue() {
      p <- p.graft();
      w <- p.observeLazy(v);
    } else {
      v.assume(p);
    }
    return w;
  }

  function replay(record:Record) -> Real {
    auto value <- coerce(record);
    auto w <- p.observe(value);
    if !v.hasValue() && w > -inf {
      v.set(p);
      w <- 0.0;
    }
    return w;
  }

  function replayMove(record:Record, scale:Real) -> Real {
    auto w <- 0.0;
    if v.hasValue() {
      w <- p.observeLazy(v);
    } else {
      auto random <- coerceRandom(record);
      assert random.hasValue();
      if random.dfdx? {
        p.set(simulate_propose(random.x!, random.dfdx!, scale));
      } else {
        p.set(random.x!);
      }
      v.set(p);
    }
    return w;
  }

  function replayDelay(record:Record) -> Real {
    auto w <- 0.0;
    if v.hasValue() {
      p <- p.graft();
      w <- p.observe(v.value());
    } else {
      auto random <- coerceRandom(record);
      if random.hasValue() {
        p <- p.graft();
        p.set(random.x!);
        v.set(p);
      } else {
        v.assume(p);
      }
    }
    return w;
  }

  function replayDelayMove(record:Record, scale:Real) -> Real {
    auto w <- 0.0;
    if v.hasValue() {
      p <- p.graft();
      w <- p.observeLazy(v);
    } else {
      auto random <- coerceRandom(record);
      if random.hasValue() {
        p <- p.graft();
        if random.dfdx? {
          p.set(simulate_propose(random.x!, random.dfdx!, scale));
        } else {
          p.set(random.x!);
        }
        v.set(p);
      } else {
        v.assume(p);
      }
    }
    return w;
  }
  
  function record() -> Record {
    return DelayRecord<Value>(v);
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

function simulate_propose(x:Real, d:Real, τ:Real) -> Real {
  return simulate_gaussian(x + τ*d, 2.0*τ);
}

function simulate_propose(x:Real[_], d:Real[_], τ:Real) -> Real[_] {
  return simulate_multivariate_gaussian(x + τ*d, 2.0*τ);
}

function simulate_propose(x:Real[_,_], d:Real[_,_], τ:Real) -> Real[_,_] {
  return simulate_matrix_gaussian(x + τ*d, 2.0*τ);
}

function simulate_propose(x:Integer, d:Integer, τ:Real) -> Integer {
  return x;
}

function simulate_propose(x:Integer[_], d:Integer[_], τ:Real) -> Integer[_] {
  return x;
}

function simulate_propose(x:Boolean, d:Boolean, τ:Real) -> Boolean {
  return x;
}

function logpdf_propose(x':Real, x:Real, d:Real, τ:Real) -> Real {
  return logpdf_gaussian(x', x + τ*d, 2.0*τ);
}

function logpdf_propose(x':Real[_], x:Real[_], d:Real[_], τ:Real) -> Real {
  return logpdf_multivariate_gaussian(x', x + τ*d, 2.0*τ);
}

function logpdf_propose(x':Real[_,_], x:Real[_,_], d:Real[_,_], τ:Real) -> Real {
  return logpdf_matrix_gaussian(x', x + τ*d, 2.0*τ);
}

function logpdf_propose(x':Integer, x:Integer, d:Integer, τ:Real) -> Real {
  return 0.0;
}

function logpdf_propose(x':Integer[_], x:Integer[_], d:Integer[_], τ:Real) -> Real {
  return 0.0;
}

function logpdf_propose(x':Boolean, x:Boolean, d:Boolean, τ:Real) -> Real {
  return 0.0;
}

function ratio_propose(trace':Trace, trace:Trace, τ:Real) -> Real {
  auto α <- 0.0;
  auto r' <- trace'.walk();
  auto r <- trace.walk();
  while r'? && r? {
    α <- α + r'!.ratio(r!, τ);
  }
  assert !r'? && !r?;
  return α;
}
