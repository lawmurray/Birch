import delay.Delay;
import math;
import random;

/**
 * Abstract delay variate with Integer value.
 */
class DelayInteger < Delay {
  /**
   * Value.
   */
  x:Integer;
  
  /**
   * Weight.
   */
  w:Real;

  /**
   * Value conversion.
   */
  operator -> Integer {
    return value();
  }

  /**
   * Value assignment.
   */
  operator <- x:Integer {
    assert isUninitialized();
    set(x);
    realize();
  }

  /**
   * String assignment.
   */
  operator <- s:String {
    set(Integer(s));
  }

  function initialize() {
    super.initialize();
  }

  function initialize(u:DelayInteger) {
    super.initialize(u);
  }
  
  function value() -> Integer {
    if (isMissing()) {
      return simulate();
    } else {
      return x;
    }
  }

  function set(x:Integer) {
    this.x <- x;
    this.missing <- false;
  }
  
  function setWeight(w:Real) {
    this.w <- w;
  }
    
  function simulate() -> Integer {
    graft();
    realize();
    return x;
  }

  function observe(x:Integer) -> Real {
    graft();
    set(x);
    realize();
    return w;
  }
}
