/**
 * Lazy `llt`.
 */
final class MatrixLLT<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {
  function rows() -> Integer {
    return single.rows();
  }
  
  function columns() -> Integer {
    return single.columns();
  }

  function doValue(x:Argument) -> Value {
    return llt(x);
  }

  function doGrad(d:Value, x:Argument) -> Argument {
    ///@todo
    assert false;
  }
}

/**
 * Lazy `inv`.
 */
function llt(x:Expression<Real[_,_]>) -> MatrixLLT<Real[_,_],LLT> {
  m:MatrixLLT<Real[_,_],LLT>(x);
  return m;
}
