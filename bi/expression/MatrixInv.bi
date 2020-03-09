/**
 * Lazy `inv`.
 */
final class MatrixInv<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {
  function rows() -> Integer {
    return single.rows();
  }
  
  function columns() -> Integer {
    return single.columns();
  }

  function doValue(x:Argument) -> Value {
    return inv(x);
  }

  function doGradient(d:Value, x:Argument) -> Argument {
    ///@todo
    assert false;
  }
}

/**
 * Lazy `inv`.
 */
function inv(x:Expression<Real[_,_]>) -> MatrixInv<Real[_,_],Real[_,_]> {
  m:MatrixInv<Real[_,_],Real[_,_]>(x);
  return m;
}

/**
 * Lazy `inv`.
 */
function inv(x:Expression<LLT>) -> MatrixInv<LLT,Real[_,_]> {
  m:MatrixInv<LLT,Real[_,_]>(x);
  return m;
}
