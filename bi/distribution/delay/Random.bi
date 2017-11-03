/**
 * Random variate.
 *
 * - Value: Value type.
 */
class Random<Value> < Delay {
  /**
   * Value.
   */
  x:Value;

  /**
   * Value conversion.
   */
  operator -> Value {
    return value();
  }

  /**
   * Value assignment.
   */
  operator <- x:Value {
    assert isUninitialized();
    set(x);
    realize();
  }
    
  function value() -> Value {
    if (isMissing()) {
      simulate();
    }
    return x;
  }

  function set(x:Value) {
    this.x <- x;
    this.missing <- false;
  }
  
  function setWeight(w:Real) {
    this.w <- w;
  }
  
  function tildeLeft() -> Random<Value> {
    simulate();
    return this;
  }
  
  function tildeRight(left:Value) -> Real {
    set(left);
    observe();
    return w;
  }
}
