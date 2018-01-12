/**
 * Random variate.
 *
 * - Value: Value type.
 */
class Random<Value> < Delay {
  /**
   * Value.
   */
  x:Value?;

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

  /**
   * Optional value assignment.
   */
  operator <- x:Value? {
    assert isUninitialized();
    if (x?) {
      set(x!);
      realize();
    }
  }
    
  function value() -> Value {
    if (isMissing()) {
      realize();
    }
    assert x?;
    return x!;
  }

  function isMissing() -> Boolean {
    return !(x?);
  }

  function set(x:Value) {
    this.x <- x;
  }
  
  function setWeight(w:Real) {
    this.w <- w;
  }
  
  function simulate() -> Value {
    realize();
    return value();
  }
  
  function observe(x:Value) -> Real {
    set(x);
    realize();
    return w;
  }
}
