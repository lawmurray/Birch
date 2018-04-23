/**
 * Delayed multiplication.
 */
class Literal<Value>(value:Value) < Expression<Value> {  
  /**
   * Value.
   */
  value:Value <- value;

  function doValue() -> Value {
    return value;
  }
}

function Literal(value:Boolean) -> Literal<Boolean> {
  m:Literal<Boolean>(value);
  return m;
}

function Literal(value:Integer) -> Literal<Integer> {
  m:Literal<Integer>(value);
  return m;
}

function Literal(value:Real) -> Literal<Real> {
  m:Literal<Real>(value);
  return m;
}

function Literal(value:Boolean[_]) -> Literal<Boolean[_]> {
  m:Literal<Boolean[_]>(value);
  return m;
}

function Literal(value:Integer[_]) -> Literal<Integer[_]> {
  m:Literal<Integer[_]>(value);
  return m;
}

function Literal(value:Real[_]) -> Literal<Real[_]> {
  m:Literal<Real[_]>(value);
  return m;
}

function Literal(value:Boolean[_,_]) -> Literal<Boolean[_,_]> {
  m:Literal<Boolean[_,_]>(value);
  return m;
}

function Literal(value:Integer[_,_]) -> Literal<Integer[_,_]> {
  m:Literal<Integer[_,_]>(value);
  return m;
}

function Literal(value:Real[_,_]) -> Literal<Real[_,_]> {
  m:Literal<Real[_,_]>(value);
  return m;
}
