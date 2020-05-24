/**
 * Lazy `inv`.
 */
final class MatrixInv<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {
  override function rows() -> Integer {
    return single.rows();
  }
  
  override function columns() -> Integer {
    return single.columns();
  }

  override function computeValue(x:Argument) -> Value {
    return inv(x);
  }

  override function computeGrad(d:Value, x:Argument) -> Argument {
    ///@todo
    assert false;
  }
}

/**
 * Lazy `inv`.
 */
function inv(x:Expression<Real[_,_]>) -> Expression<Real[_,_]> {
  if x.isConstant() {
    return box(matrix(inv(x.value())));
  } else {
    m:MatrixInv<Real[_,_],Real[_,_]>(x);
    return m;
  }
}

/**
 * Lazy `inv`.
 */
function inv(x:Expression<LLT>) -> Expression<Real[_,_]> {
  if x.isConstant() {
    return box(matrix(inv(x.value())));
  } else {
    m:MatrixInv<LLT,Real[_,_]>(x);
    return m;
  }
}
