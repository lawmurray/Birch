/**
 * Delayed expression.
 *
 * - Value: Value type.
 */
class Expression<Value> < Delay {  
  /**
   * Value conversion.
   */
  operator -> Value {
    return value();
  }
  
  /**
   * Value conversion.
   */
  function value() -> Value {
    assert false;
  }
  
  /**
   * Are the values of any random variables within this expression missing?
   */
  function isMissing() -> Boolean {
    assert false;
  }
}

function value(x:Expression<Real>) -> Real {
  return x.value();
}

function value(x:Expression<Integer>) -> Integer {
  return x.value();
}

function value(x:Expression<Real[_]>) -> Real[_] {
  return x.value();
}

function value(x:Expression<Integer[_]>) -> Integer[_] {
  return x.value();
}

function value(x:Expression<Real[_,_]>) -> Real[_,_] {
  return x.value();
}

function value(x:Expression<Integer[_,_]>) -> Integer[_,_] {
  return x.value();
}

function value(x:Real) -> Real {
  return x;
}

function value(x:Integer) -> Integer {
  return x;
}

function value(x:Real[_]) -> Real[_] {
  return x;
}

function value(x:Integer[_]) -> Integer[_] {
  return x;
}

function value(x:Real[_,_]) -> Real[_,_] {
  return x;
}

function value(x:Integer[_,_]) -> Integer[_,_] {
  return x;
}

function isMissing(x:Expression<Real>) -> Boolean {
  return x.isMissing();
}

function isMissing(x:Expression<Integer>) -> Boolean {
  return x.isMissing();
}

function isMissing(x:Expression<Real[_]>) -> Boolean {
  return x.isMissing();
}

function isMissing(x:Expression<Integer[_]>) -> Boolean {
  return x.isMissing();
}

function isMissing(x:Expression<Real[_,_]>) -> Boolean {
  return x.isMissing();
}

function isMissing(x:Expression<Integer[_,_]>) -> Boolean {
  return x.isMissing();
}

function isMissing(x:Real) -> Boolean {
  return false;
}

function isMissing(x:Integer) -> Boolean {
  return false;
}

function isMissing(x:Real[_]) -> Boolean {
  return false;
}

function isMissing(x:Integer[_]) -> Boolean {
  return false;
}

function isMissing(x:Real[_,_]) -> Boolean {
  return false;
}

function isMissing(x:Integer[_,_]) -> Boolean {
  return false;
}
