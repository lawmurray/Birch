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

  function initialize() {
    super.initialize();
  }

  function initialize(u:DelayReal) {
    super.initialize(u);
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
}
