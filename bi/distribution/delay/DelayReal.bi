/**
 * Abstract delay variate with Real value.
 */
class DelayReal < Delay {
  /**
   * Value.
   */
  x:Real;
  
  /**
   * Weight.
   */
  w:Real;

  /**
   * Value conversion.
   */
  operator -> Real {
    return value();
  }

  /**
   * Value assignment.
   */
  operator <- x:Real {
    assert isUninitialized();
    set(x);
    realize();
  }

  /**
   * String assignment.
   */
  operator <- s:String {
    set(Real(s));
  }
  
  function value() -> Real {
    if (isMissing()) {
      simulate();
    }
    return x;
  }

  function set(x:Real) {
    this.x <- x;
    this.missing <- false;
  }
  
  function setWeight(w:Real) {
    this.w <- w;
  }

  function tildeRight(left:Real) -> Real {
    set(left);
    observe();
    return w;
  }
}
