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
function matrix(x:Expression<LLT>) -> Expression<Real[_,_]> {
  if x.isConstant() {
    return box(matrix(x.value()));
  } else {
    m:Matrix<LLT,Real[_,_]>(x);
    return m;
  }
}

/**
 * Lazy `matrix`.
 */
function matrix(x:Expression<Real[_,_]>) -> Expression<Real[_,_]> {
  return x;
}

/**
 * Lazy `matrix`.
 */
function matrix(x:Expression<Real[_]>) -> Expression<Real[_,_]> {
  if x.isConstant() {
    return box(matrix(x.value()));
  } else {
    m:Matrix<Real[_],Real[_,_]>(x);
    return m;
  }
}

/**
 * Lazy `matrix`.
 */
function matrix(x:Expression<Real>) -> Expression<Real[_,_]> {
  if x.isConstant() {
    return box(matrix(x.value()));
  } else {
    m:Matrix<Real,Real[_,_]>(x);
    return m;
  }
}
