import delay.Delay;
import math;
import random;

/**
 * Abstract delay variate with real vector value.
 *
 * `D` Number of dimensions.
 */
class DelayRealVector(D:Integer) < Delay {
  /**
   * Value.
   */
  x:Real[D];
  
  /**
   * Weight.
   */
  w:Real;

  /**
   * Value conversion.
   */
  operator -> Real[_] {
    return value();
  }
  
  /**
   * Value assignment.
   */
  operator <- x:Real[_] {
    assert isUninitialized();
    set(x);
    realize();
  }

  function initialize() {
    super.initialize();
  }

  function initialize(u:DelayRealVector) {
    super.initialize(u);
  }

  function value() -> Real[_] {
    if (isMissing()) {
      return simulate();
    } else {
      return x;
    }
  }
  
  function set(x:Real[_]) {
    assert length(x) == D;
    
    this.x <- x;
    this.missing <- false;
  }
  
  function setWeight(w:Real) {
    this.w <- w;
  }
  
  function simulate() -> Real[_] {
    graft();
    realize();
    return x;
  }
  
  function observe(x:Real[_]) -> Real {
    graft();
    set(x);
    absorb(1);
    realize();
    return w;
  }
}
