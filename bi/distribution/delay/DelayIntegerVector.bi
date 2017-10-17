/**
 * Abstract delay variate with integer vector value.
 *
 * `D` Number of dimensions.
 */
class DelayIntegerVector(D:Integer) < Delay {
  /**
   * Value.
   */
  x:Integer[D];
  
  /**
   * Value conversion.
   */
  operator -> Integer[_] {
    return value();
  }
  
  /**
   * Value assignment.
   */
  operator <- x:Integer[_] {
    assert isUninitialized();
    set(x);
    realize();
  }

  function value() -> Integer[_] {
    if (isMissing()) {
      simulate();
    }
    return x;
  }
  
  function set(x:Integer[_]) {
    assert length(x) == D;
    
    this.x <- x;
    this.missing <- false;
  }
  
  function setWeight(w:Real) {
    this.w <- w;
  }

  function tildeRight(left:Integer[_]) -> Real {
    set(left);
    observe();
    return w;
  }
}
