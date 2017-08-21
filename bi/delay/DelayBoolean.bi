import delay.Delay;
import math;
import random;

/**
 * Abstract delay variate with Boolean value.
 */
class DelayBoolean < Delay {
  /**
   * Value.
   */
  x:Boolean;
  
  /**
   * Weight.
   */
  w:Real;

  /**
   * Value conversion.
   */
  operator -> Boolean {
    return value();
  }

  /**
   * Value assignment.
   */
  operator <- x:Boolean {
    assert isUninitialized();
    set(x);
    realize();
  }

  /**
   * String assignment.
   */
  operator <- s:String {
    set(Boolean(s));
  }

  function initialize() {
    super.initialize();
  }

  function initialize(u:DelayBoolean) {
    super.initialize(u);
  }
  
  function value() -> Boolean {
    if (isMissing()) {
      return simulate();
    } else {
      return x;
    }
  }

  function set(x:Boolean) {
    this.x <- x;
    this.missing <- false;
  }
  
  function setWeight(w:Real) {
    this.w <- w;
  }
    
  function simulate() -> Boolean {
    graft();
    realize();
    return x;
  }

  function observe(x:Boolean) -> Real {
    graft();
    set(x);
    realize();
    return w;
  }
}
