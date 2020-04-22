/**
 * Boxed value.
 */
final class Boxed<Value> < Expression<Value> {  
  function rows() -> Integer {
    return global.rows(get());
  }

  function columns() -> Integer {
    return global.columns(get());
  }

  function doValue() -> Value {
    assert false;  // should never arrive here, as x set by factory function
    return get();
  }

  function doPilot() -> Value {
    assert false;  // should never arrive here, as x set by factory function
    return get();
  }

  function doGrad(d:Value) {
    //
  }
}

function Boxed(x:Real) -> Boxed<Real> {
  m:Boxed<Real>;
  m.x <- x;
  return m;
}

function Boxed(x:Integer) -> Boxed<Integer> {
  m:Boxed<Integer>;
  m.x <- x;
  return m;
}

function Boxed(x:Boolean) -> Boxed<Boolean> {
  m:Boxed<Boolean>;
  m.x <- x;
  return m;
}

function Boxed(x:Real[_]) -> Boxed<Real[_]> {
  m:Boxed<Real[_]>;
  m.x <- x;
  return m;
}

function Boxed(x:Integer[_]) -> Boxed<Integer[_]> {
  m:Boxed<Integer[_]>;
  m.x <- x;
  return m;
}

function Boxed(x:Real[_,_]) -> Boxed<Real[_,_]> {
  m:Boxed<Real[_,_]>;
  m.x <- x;
  return m;
}

function Boxed(x:Integer[_,_]) -> Boxed<Integer[_,_]> {
  m:Boxed<Integer[_,_]>;
  m.x <- x;
  return m;
}
