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

  function value() -> Real[_] {
    if (isMissing()) {
      simulate();
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

  function tildeRight(left:Real[_]) -> Real {
    set(left);
    observe();
    return w;
  }
}
