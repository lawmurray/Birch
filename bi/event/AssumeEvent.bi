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

  function playMove() -> Real {
    auto w <- 0.0;
    if v.hasValue() {
      auto ψ <- p.lazy(v);
      if ψ? {
        w <- ψ!.value();
        ψ!.grad(1.0);
      } else {
        w <- p.observe(v.value());
      }
    } else {
      v <- p.value();
    }
    return w;
  }

  function playDelayMove() -> Real {
    auto w <- 0.0;
    if v.hasValue() {
      p <- p.graft();
      auto ψ <- p.lazy(v);
      if ψ? {
        w <- ψ!.value();
        ψ!.grad(1.0);
      } else {
        w <- p.observe(v.value());
      }
    } else {
      p.assume(v);
    }
    return w;
  }

  function replay(record:Record) -> Real {
    auto value <- coerce(record);
    auto w <- p.observe(value);
    if !v.hasValue() && w > -inf {
      v <- value;
      w <- 0.0;
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
        p.assume(v, random.value());
      } else {
        p.assume(v);
      }
    }
    return w;
  }

  function replayMove(record:Record) -> Real {
    auto w <- 0.0;
    if v.hasValue() {
      auto ψ <- p.lazy(v);
      if ψ? {
        w <- ψ!.value();
        ψ!.grad(1.0);
      } else {
        w <- p.observe(v.value());
      }
    } else {
      auto random <- coerceRandom(record);
      assert random.hasValue();
      if random.dfdx? {
        v <- simulate_propose(random.x!, random.dfdx!);
      } else {
        v <- random.x!;
      }
    }
    return w;
  }

  function replayDelayMove(record:Record) -> Real {
    auto w <- 0.0;
    if v.hasValue() {
      p <- p.graft();
      auto ψ <- p.lazy(v);
      if ψ? {
        w <- ψ!.value();
        ψ!.grad(1.0);
      } else {
        w <- p.observe(v.value());
      }
    } else {
      auto random <- coerceRandom(record);
      if random.hasValue() {
        if random.dfdx? {
          p.assume(v, simulate_propose(random.x!, random.dfdx!));
        } else {
          p.assume(v, random.x!);
        }
      } else {
        p.assume(v);
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

function simulate_propose(x:Real, d:Real) -> Real {
  return simulate_gaussian(x + 0.004*d, 2.0*0.004);
}

function simulate_propose(x:Real[_], d:Real[_]) -> Real[_] {
  return simulate_multivariate_gaussian(x + 0.004*d, 2.0*0.004);
}

function simulate_propose(x:Real[_,_], d:Real[_,_]) -> Real[_,_] {
  return simulate_matrix_gaussian(x + 0.004*d, 2.0*0.004);
}

function simulate_propose(x:Integer, d:Integer) -> Integer {
  return x;
}

function simulate_propose(x:Integer[_], d:Integer[_]) -> Integer[_] {
  return x;
}

function simulate_propose(x:Boolean, d:Boolean) -> Boolean {
  return x;
}

function logpdf_propose(x':Real, x:Real, d:Real) -> Real {
  return logpdf_gaussian(x', x + 0.004*d, 2.0*0.004);
}

function logpdf_propose(x':Real[_], x:Real[_], d:Real[_]) -> Real {
  return logpdf_multivariate_gaussian(x', x + 0.004*d, 2.0*0.004);
}

function logpdf_propose(x':Real[_,_], x:Real[_,_], d:Real[_,_]) -> Real {
  return logpdf_matrix_gaussian(x', x + 0.004*d, 2.0*0.004);
}

function logpdf_propose(x':Integer, x:Integer, d:Integer) -> Real {
  return 0.0;
}

function logpdf_propose(x':Integer[_], x:Integer[_], d:Integer[_]) -> Real {
  return 0.0;
}

function logpdf_propose(x':Boolean, x:Boolean, d:Boolean) -> Real {
  return 0.0;
}

function ratio_propose(trace':Trace, trace:Trace) -> Real {
  auto α <- 0.0;
  auto record' <- trace'.walk();
  auto record <- trace.walk();
  while record'? && record? {
    α <- α + record'!.ratio(record!);
  }
  assert !record'? && !record?;
  return α;
}
