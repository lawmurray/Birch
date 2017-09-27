/**
 * Abstract delay variate with Integer value.
 */
class DelayInteger < Delay {
  /**
   * Value.
   */
  x:Integer;

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
  
  function value() -> Integer {
    if (isMissing()) {
      simulate();
    }
    return x;
  }

  function set(x:Integer) {
    this.x <- x;
    this.missing <- false;
  }
  
  function setWeight(w:Real) {
    this.w <- w;
  }
  
  function tildeRight(left:Integer) -> Real {
    set(left);
    observe();
    return w;
  }
}
