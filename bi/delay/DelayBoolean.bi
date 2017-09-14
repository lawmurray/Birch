import delay.Delay;
import math;
import math.simulate;

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
      simulate();
    }
    return x;
  }

  function set(x:Boolean) {
    this.x <- x;
    this.missing <- false;
  }
  
  function setWeight(w:Real) {
    this.w <- w;
  }
}
