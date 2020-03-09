/**
 * Lazy `matrix`.
 */
final class Matrix<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {
  function rows() -> Integer {
    return single.columns();
  }
  
  function columns() -> Integer {
    return single.rows();
  }

  function doValue(x:Argument) -> Value {
    return matrix(x);
  }

  function doGradient(d:Value, x:Argument) -> Argument {
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
