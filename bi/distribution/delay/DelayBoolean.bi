/**
 * Abstract delay variate with Boolean value.
 */
class DelayBoolean < Delay {
  /**
   * Value.
   */
  x:Boolean;

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

  function tildeRight(left:Boolean) -> Real {
    set(left);
    observe();
    return w;
  }
}
