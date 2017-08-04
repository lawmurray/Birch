import distribution.Delay;
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
    set(x);
  }

  function initialize() {
    super.initialize();
  }

  function initialize(u:DelayRealVector) {
    super.initialize(u);
  }

  function value() -> Real[_] {
    if (isMissing()) {
      graft();
      realize();
    }
    return x;
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
    return value();
  }
  
  function observe(x:Real[_]) -> Real {
    graft();
    set(x);
    realize();
    return w;
  }
}
