/**
 * Lazy `matrix`.
 */
final class Matrix<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {
  override function rows() -> Integer {
    return single.columns();
  }
  
  override function columns() -> Integer {
    return single.rows();
  }

  override function computeValue(x:Argument) -> Value {
    return matrix(x);
  }

  override function computeGrad(d:Value, x:Argument) -> Argument {
    ///@todo
    assert false;
  }
}

/**
 * Lazy `matrix`.
 */
function matrix(x:Expression<LLT>) -> Matrix<LLT,Real[_,_]> {
  m:Matrix<LLT,Real[_,_]>(x);
  return m;
}
