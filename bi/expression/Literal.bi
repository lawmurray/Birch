/**
 * Delayed multiplication.
 */
class Literal<Value> < Expression<Value> {  
  //
}

function Literal(value:Boolean) -> Literal<Boolean> {
  m:Literal<Boolean>;
  m.x <- value;
  return m;
}

function Literal(value:Integer) -> Literal<Integer> {
  m:Literal<Integer>;
  m.x <- value;
  return m;
}

function Literal(value:Real) -> Literal<Real> {
  m:Literal<Real>;
  m.x <- value;
  return m;
}

function Literal(value:Boolean[_]) -> Literal<Boolean[_]> {
  m:Literal<Boolean[_]>;
  m.x <- value;
  return m;
}

function Literal(value:Integer[_]) -> Literal<Integer[_]> {
  m:Literal<Integer[_]>;
  m.x <- value;
  return m;
}

function Literal(value:Real[_]) -> Literal<Real[_]> {
  m:Literal<Real[_]>;
  m.x <- value;
  return m;
}

function Literal(value:Boolean[_,_]) -> Literal<Boolean[_,_]> {
  m:Literal<Boolean[_,_]>;
  m.x <- value;
  return m;
}

function Literal(value:Integer[_,_]) -> Literal<Integer[_,_]> {
  m:Literal<Integer[_,_]>;
  m.x <- value;
  return m;
}

function Literal(value:Real[_,_]) -> Literal<Real[_,_]> {
  m:Literal<Real[_,_]>;
  m.x <- value;
  return m;
}
