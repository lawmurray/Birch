/**
 * Lazy matrix multiply.
 */
final class MatrixMultiply<Left,Right,Value>(left:Left, right:Right) <
    MatrixBinaryExpression<Left,Right,Value>(left, right) {  
  override function rows() -> Integer {
    return left.rows();
  }
  
  override function columns() -> Integer {
    return right.columns();
  }

  override function doValue() {
    x <- left.value()*right.value();
  }

  override function doGet() {
    x <- left.get()*right.get();
  }

  override function doPilot() {
    x <- left.pilot()*right.pilot();
  }

  override function doMove(κ:Kernel) {
    x <- left.move(κ)*right.move(κ);
  }

  override function doGrad() {
    left.grad(D!*transpose(right.get()));
    right.grad(transpose(left.get())*D!);
  }

  override function graftLinearMatrixGaussian() ->
      TransformLinearMatrix<MatrixGaussian>? {
    y:TransformLinearMatrix<MatrixGaussian>?;
    z:MatrixGaussian?;
    
    if (y <- right.graftLinearMatrixGaussian())? {
      y!.leftMultiply(matrix(left));
    } else if (z <- right.graftMatrixGaussian())? {
      y <- TransformLinearMatrix<MatrixGaussian>(matrix(left), z!);
    }
    return y;
  }
  
  override function graftLinearMatrixNormalInverseGamma(
      compare:Distribution<Real[_]>) ->
      TransformLinearMatrix<MatrixNormalInverseGamma>? {
    y:TransformLinearMatrix<MatrixNormalInverseGamma>?;
    z:MatrixNormalInverseGamma?;

    if (y <- right.graftLinearMatrixNormalInverseGamma(compare))? {
      y!.leftMultiply(matrix(left));
    } else if (z <- right.graftMatrixNormalInverseGamma(compare))? {
      y <- TransformLinearMatrix<MatrixNormalInverseGamma>(matrix(left), z!);
    }
    return y;
  }

  override function graftLinearMatrixNormalInverseWishart(
      compare:Distribution<LLT>) ->
      TransformLinearMatrix<MatrixNormalInverseWishart>? {
    y:TransformLinearMatrix<MatrixNormalInverseWishart>?;
    z:MatrixNormalInverseWishart?;

    if (y <- right.graftLinearMatrixNormalInverseWishart(compare))? {
      y!.leftMultiply(matrix(left));
    } else if (z <- right.graftMatrixNormalInverseWishart(compare))? {
      y <- TransformLinearMatrix<MatrixNormalInverseWishart>(matrix(left), z!);
    }
    return y;
  }
}

/**
 * Lazy matrix multiply.
 */
operator (left:Expression<Real[_,_]>*right:Expression<Real[_,_]>) ->
    Expression<Real[_,_]> {
  assert left.columns() == right.rows();
  if left.isConstant() && right.isConstant() {
    return box(matrix(left.value()*right.value()));
  } else {
    m:MatrixMultiply<Expression<Real[_,_]>,Expression<Real[_,_]>,Real[_,_]>(left, right);
    return m;
  }
}

/**
 * Lazy matrix multiply.
 */
operator (left:Real[_,_]*right:Expression<Real[_,_]>) ->
    Expression<Real[_,_]> {
  if right.isConstant() {
    return box(matrix(left*right.value()));
  } else {
    return box(left)*right;
  }
}

/**
 * Lazy matrix multiply.
 */
operator (left:Expression<Real[_,_]>*right:Real[_,_]) ->
    Expression<Real[_,_]> {
  if left.isConstant() {
    return box(matrix(left.value()*right));
  } else {
    return left*box(right);
  }
}

/**
 * Lazy matrix multiply.
 */
operator (left:Expression<LLT>*right:Expression<Real[_,_]>) ->
    Expression<Real[_,_]> {
  assert left.columns() == right.rows();
  if left.isConstant() && right.isConstant() {
    return box(matrix(left.value()*right.value()));
  } else {
    m:MatrixMultiply<Expression<LLT>,Expression<Real[_,_]>,Real[_,_]>(left, right);
    return m;
  }
}

/**
 * Lazy matrix multiply.
 */
operator (left:LLT*right:Expression<Real[_,_]>) -> Expression<Real[_,_]> {
  if right.isConstant() {
    return box(matrix(left*right.value()));
  } else {
    return box(left)*right;
  }
}

/**
 * Lazy matrix multiply.
 */
operator (left:Expression<LLT>*right:Real[_,_]) -> Expression<Real[_,_]> {
  if left.isConstant() {
    return box(matrix(left.value()*right));
  } else {
    return left*box(right);
  }
}

/**
 * Lazy matrix multiply.
 */
operator (left:Expression<Real[_,_]>*right:Expression<LLT>) ->
    Expression<Real[_,_]> {
  assert left.columns() == right.rows();
  if left.isConstant() && right.isConstant() {
    return box(matrix(left.value()*right.value()));
  } else {
    m:MatrixMultiply<Expression<Real[_,_]>,Expression<LLT>,Real[_,_]>(left, right);
    return m;
  }
}

/**
 * Lazy matrix multiply.
 */
operator (left:Real[_,_]*right:Expression<LLT>) -> Expression<Real[_,_]> {
  if right.isConstant() {
    return box(matrix(left*right.value()));
  } else {
    return box(left)*right;
  }
}

/**
 * Lazy matrix multiply.
 */
operator (left:Expression<Real[_,_]>*right:LLT) -> Expression<Real[_,_]> {
  if left.isConstant() {
    return box(matrix(left.value()*right));
  } else {
    return left*box(right);
  }
}

/**
 * Lazy matrix multiply.
 */
operator (left:Expression<LLT>*right:Expression<LLT>) ->
    Expression<Real[_,_]> {
  assert left.columns() == right.rows();
  if left.isConstant() && right.isConstant() {
    return box(matrix(left.value()*right.value()));
  } else {
    m:MatrixMultiply<Expression<LLT>,Expression<LLT>,Real[_,_]>(left, right);
    return m;
  }
}

/**
 * Lazy matrix multiply.
 */
operator (left:LLT*right:Expression<LLT>) -> Expression<Real[_,_]> {
  if right.isConstant() {
    return box(matrix(left*right.value()));
  } else {
    return box(left)*right;
  }
}

/**
 * Lazy matrix multiply.
 */
operator (left:Expression<LLT>*right:LLT) -> Expression<Real[_,_]> {
  if left.isConstant() {
    return box(matrix(left.value()*right));
  } else {
    return left*box(right);
  }
}

/**
 * Lazy matrix multiply.
 */
operator (left:Expression<Real>*right:Expression<Real[_,_]>) ->
    Expression<Real[_,_]> {
  return diagonal(left, right.rows())*right;
}

/**
 * Lazy matrix multiply.
 */
operator (left:Real*right:Expression<Real[_,_]>) -> Expression<Real[_,_]> {
  return box(left)*right;
}

/**
 * Lazy matrix multiply.
 */
operator (left:Expression<Real>*right:Real[_,_]) -> Expression<Real[_,_]> {
  return left*box(right);
}

/**
 * Lazy matrix multiply.
 */
operator (left:Expression<Real[_,_]>*right:Expression<Real>) ->
    Expression<Real[_,_]> {
  return right*left;
}

/**
 * Lazy matrix multiply.
 */
operator (left:Real[_,_]*right:Expression<Real>) -> Expression<Real[_,_]> {
  return right*left;
}

/**
 * Lazy matrix multiply.
 */
operator (left:Expression<Real[_,_]>*right:Real) -> Expression<Real[_,_]> {
  return right*left;
}

/**
 * Lazy matrix multiply.
 */
operator (left:Expression<Real>*right:Expression<LLT>) ->
    Expression<Real[_,_]> {
  return diagonal(left, right.rows())*right;
}

/**
 * Lazy matrix multiply.
 */
operator (left:Real*right:Expression<LLT>) -> Expression<Real[_,_]> {
  return box(left)*right;
}

/**
 * Lazy matrix multiply.
 */
operator (left:Expression<Real>*right:LLT) -> Expression<Real[_,_]> {
  return left*box(right);
}

/**
 * Lazy matrix multiply.
 */
operator (left:Expression<LLT>*right:Expression<Real>) ->
    Expression<Real[_,_]> {
  return right*left;
}

/**
 * Lazy matrix multiply.
 */
operator (left:LLT*right:Expression<Real>) -> Expression<Real[_,_]> {
  return right*left;
}

/**
 * Lazy matrix multiply.
 */
operator (left:Expression<LLT>*right:Real) -> Expression<Real[_,_]> {
  return right*left;
}
