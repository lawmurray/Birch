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
      v.assume(p);
      v.value();
    }
    return w;
  }

  function playMove() -> Real {
    auto w <- 0.0;
    if v.hasValue() {
      w <- p.observeLazy(v);
    } else {
      v.assume(p);
      v.pilot();
    }
    return w;
  }

  function playDelay() -> Real {
    auto w <- 0.0;
    if v.hasValue() {
      p <- p.graft();
      w <- p.observe(v.value());
    } else {
      v.assume(p);
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
    auto w <- 0.0;
    if v.hasValue() {
      w <- p.observe(v.value());
    } else {
      auto value <- coerce(record);
      if p.logpdf(value) > -inf {
        v.assume(p);
        v.proposeValue(value);
      } else {
        w <- -inf;
      }
    }
    return w;
  }

  function replayMove(record:Record, scale:Real) -> Real {
    auto w <- 0.0;
    if v.hasValue() {
      w <- p.observeLazy(v);
    } else {
      auto random <- coerceRandom(record);
      v.assume(p);
      if random.dfdx? {
        v.proposePilot(simulate_propose(random.value(), random.dfdx!, scale));
      } else {
        v.proposeValue(random.value());
      }
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
      v.assume(p);
      if random.hasValue() {
        v.proposeValue(random.value());
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
      v.assume(p);
      if random.hasValue() {
        if random.dfdx? {
          v.proposePilot(simulate_propose(random.value(), random.dfdx!, scale));
        } else {
          v.proposeValue(random.value());
        }
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
function AssumeEvent(v:Random<Real>, p:Distribution<Real>) ->
    AssumeEvent<Real> {
  evt:AssumeEvent<Real>(v, p);
  return evt;
}

/**
 * Create an AssumeEvent.
 */
function AssumeEvent(v:Random<Real[_]>, p:Distribution<Real[_]>) ->
    AssumeEvent<Real[_]> {
  evt:AssumeEvent<Real[_]>(v, p);
  return evt;
}

/**
 * Create an AssumeEvent.
 */
function AssumeEvent(v:Random<Real[_,_]>, p:Distribution<Real[_,_]>) ->
    AssumeEvent<Real[_,_]> {
  evt:AssumeEvent<Real[_,_]>(v, p);
  return evt;
}

/**
 * Create an AssumeEvent.
 */
function AssumeEvent(v:Random<Integer>, p:Distribution<Integer>) ->
    AssumeEvent<Integer> {
  evt:AssumeEvent<Integer>(v, p);
  return evt;
}

/**
 * Create an AssumeEvent.
 */
function AssumeEvent(v:Random<Integer[_]>, p:Distribution<Integer[_]>) ->
    AssumeEvent<Integer[_]> {
  evt:AssumeEvent<Integer[_]>(v, p);
  return evt;
}

/**
 * Create an AssumeEvent.
 */
function AssumeEvent(v:Random<Boolean>, p:Distribution<Boolean>) ->
    AssumeEvent<Boolean> {
  evt:AssumeEvent<Boolean>(v, p);
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
