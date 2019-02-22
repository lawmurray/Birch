/**
 * Boxed value.
 */
class Boxed<Value>(x:Value) < Expression<Value> {  
  /**
   * Value.
   */
  x:Value <- x;

  operator <- x:Value {
    this.x <- x;
  }

  function value() -> Value {
    return x;
  }

  function boxed() -> Boxed<Value> {
    return this;
  }
}

function Boxed(x:Boolean) -> Boxed<Boolean> {
  m:Boxed<Boolean>(x);
  return m;
}

function Boxed(x:Integer) -> Boxed<Integer> {
  m:Boxed<Integer>(x);
  return m;
}

function Boxed(x:Real) -> Boxed<Real> {
  m:Boxed<Real>(x);
  return m;
}

function Boxed(x:Boolean[_]) -> Boxed<Boolean[_]> {
  m:Boxed<Boolean[_]>(x);
  return m;
}

function Boxed(x:Integer[_]) -> Boxed<Integer[_]> {
  m:Boxed<Integer[_]>(x);
  return m;
}

function Boxed(x:Real[_]) -> Boxed<Real[_]> {
  m:Boxed<Real[_]>(x);
  return m;
}

function Boxed(x:Boolean[_,_]) -> Boxed<Boolean[_,_]> {
  m:Boxed<Boolean[_,_]>(x);
  return m;
}

function Boxed(x:Integer[_,_]) -> Boxed<Integer[_,_]> {
  m:Boxed<Integer[_,_]>(x);
  return m;
}

function Boxed(x:Real[_,_]) -> Boxed<Real[_,_]> {
  m:Boxed<Real[_,_]>(x);
  return m;
}
