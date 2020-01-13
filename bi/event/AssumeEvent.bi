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

  function replayDelay(record:Record) -> Real {
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

  function replayMove(record:Record) -> Real {
    auto w <- 0.0;
    auto value <- coerce(record);
    if v.hasValue() {
      assert v.value() == value;
      auto ψ <- p.lazy(v);
      if ψ? {
        w <- ψ!.value();
        ψ!.grad(1.0);
      } else {
        w <- p.observe(value);
      }
    } else {
      if v.dfdx? {
        auto u <- simulate_propose(value, v.dfdx!);
        w <- p.observe(u);
        if w != -inf {
          v <- u;
          w <- 0.0;
        }
      } else {
        w <- p.observe(value);
        if w != -inf {
          v <- value;
          w <- 0.0;
        }
      }
    }
    return w;
  }

  function replayDelayMove(record:Record) -> Real {
    auto w <- 0.0;
    auto value <- coerce(record);
    if v.hasValue() {
      assert v.value() == value;
      auto ψ <- p.lazy(v);
      if ψ? {
        w <- ψ!.value();
        ψ!.grad(1.0);
      } else {
        w <- p.observe(value);
      }
    } else {
      if v.dfdx? {
        p.assume(v, simulate_propose(value, v.dfdx!));
      } else {
        p.assume(v, value);
      }
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
      return ImmediateRecord<Value>(v.value());
    } else {
      return DelayRecord<Value>(v);
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

function simulate_propose(x:Real, d:Real) -> Real {
  return simulate_gaussian(x + 0.03*d, 0.06);
}

function simulate_propose(x:Real[_], d:Real[_]) -> Real[_] {
  return simulate_multivariate_gaussian(x + d, 1.0);
}

function simulate_propose(x:Real[_,_], d:Real[_,_]) -> Real[_,_] {
  return simulate_matrix_gaussian(x + d, 1.0);
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
  return logpdf_gaussian(x', x + 0.03*d, 0.06);
}

function logpdf_propose(x':Real[_], x:Real[_], d:Real[_]) -> Real {
  return logpdf_multivariate_gaussian(x', x + d, 1.0);
}

function logpdf_propose(x':Real[_,_], x:Real[_,_], d:Real[_,_]) -> Real {
  return logpdf_matrix_gaussian(x', x + d, 1.0);
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
