/**
 * Lazy `llt`.
 */
final class MatrixLLT<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {
  override function rows() -> Integer {
    return single.rows();
  }
  
  override function columns() -> Integer {
    return single.columns();
  }

  override function computeValue(x:Argument) -> Value {
    return llt(x);
  }

  override function computeGrad(d:Value, x:Argument) -> Argument {
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
