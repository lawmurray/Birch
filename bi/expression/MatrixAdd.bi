/**
 * Lazy matrix addition.
 */
final class MatrixAdd<Left,Right,Value>(left:Left, right:Right) <
    MatrixBinaryExpression<Left,Right,Value>(left, right) {  
  override function rows() -> Integer {
    return left.rows();
  }
  
  override function columns() -> Integer {
    return left.columns();
  }
    
  override function doValue() {
    x <- left.value() + right.value();
  }

  override function doPilot() {
    x <- left.pilot() + right.pilot();
  }

  override function doMove(κ:Kernel) {
    x <- left.move(κ) + right.move(κ);
  }

  override function doGrad() {
    left.grad(D!);
    right.grad(D!);
  }

  override function graftLinearMatrixGaussian() ->
      TransformLinearMatrix<MatrixGaussian>? {
    y:TransformLinearMatrix<MatrixGaussian>?;
    z:MatrixGaussian?;

    if (y <- left.graftLinearMatrixGaussian())? {
      y!.add(right);
    } else if (y <- right.graftLinearMatrixGaussian())? {
      y!.add(left);
    } else if (z <- left.graftMatrixGaussian())? {
      y <- TransformLinearMatrix<MatrixGaussian>(box(identity(z!.rows())), z!, right);
    } else if (z <- right.graftMatrixGaussian())? {
      y <- TransformLinearMatrix<MatrixGaussian>(box(identity(z!.rows())), z!, left);
    }
    return y;
  }
  
  override function graftLinearMatrixNormalInverseGamma(compare:Distribution<Real[_]>) ->
      TransformLinearMatrix<MatrixNormalInverseGamma>? {
    y:TransformLinearMatrix<MatrixNormalInverseGamma>?;
    z:MatrixNormalInverseGamma?;

    if (y <- left.graftLinearMatrixNormalInverseGamma(compare))? {
      y!.add(right);
    } else if (y <- right.graftLinearMatrixNormalInverseGamma(compare))? {
      y!.add(left);
    } else if (z <- left.graftMatrixNormalInverseGamma(compare))? {
      y <- TransformLinearMatrix<MatrixNormalInverseGamma>(box(identity(z!.rows())), z!, right);
    } else if (z <- right.graftMatrixNormalInverseGamma(compare))? {
      y <- TransformLinearMatrix<MatrixNormalInverseGamma>(box(identity(z!.rows())), z!, left);
    }
    return y;
  }

  override function graftLinearMatrixNormalInverseWishart(compare:Distribution<LLT>) ->
      TransformLinearMatrix<MatrixNormalInverseWishart>? {
    y:TransformLinearMatrix<MatrixNormalInverseWishart>?;
    z:MatrixNormalInverseWishart?;

    if (y <- left.graftLinearMatrixNormalInverseWishart(compare))? {
      y!.add(right);
    } else if (y <- right.graftLinearMatrixNormalInverseWishart(compare))? {
      y!.add(left);
    } else if (z <- left.graftMatrixNormalInverseWishart(compare))? {
      y <- TransformLinearMatrix<MatrixNormalInverseWishart>(box(identity(z!.rows())), z!, right);
    } else if (z <- right.graftMatrixNormalInverseWishart(compare))? {
      y <- TransformLinearMatrix<MatrixNormalInverseWishart>(box(identity(z!.rows())), z!, left);
    }
    return y;
  }
}

/**
 * Lazy matrix addition.
 */
operator (left:Expression<Real[_,_]> + right:Expression<Real[_,_]>) ->
    Expression<Real[_,_]> {
  assert left.rows() == right.rows();
  assert left.columns() == right.columns();
  if left.isConstant() && right.isConstant() {
    return box(matrix(left.value() + right.value()));
  } else {
    m:MatrixAdd<Expression<Real[_,_]>,Expression<Real[_,_]>,Real[_,_]>(left, right);
    return m;
  }
}

/**
 * Lazy matrix addition.
 */
operator (left:Real[_,_] + right:Expression<Real[_,_]>) ->
    Expression<Real[_,_]> {
  if right.isConstant() {
    return box(matrix(left + right.value()));
  } else {
    return box(left) + right;
  }
}

/**
 * Lazy matrix addition.
 */
operator (left:Expression<Real[_,_]> + right:Real[_,_]) ->
    Expression<Real[_,_]> {
  if left.isConstant() {
    return box(matrix(left.value() + right));
  } else {
    return left + box(right);
  }
}

/**
 * Lazy matrix addition.
 */
operator (left:Expression<LLT> + right:Expression<Real[_,_]>) ->
    Expression<Real[_,_]> {
  return matrix(left) + right;
}

/**
 * Lazy matrix addition.
 */
operator (left:LLT + right:Expression<Real[_,_]>) -> Expression<Real[_,_]> {
  return matrix(left) + right;
}

/**
 * Lazy matrix addition.
 */
operator (left:Expression<LLT> + right:Real[_,_]) -> Expression<Real[_,_]> {
  return matrix(left) + right;
}

/**
 * Lazy matrix addition.
 */
operator (left:Expression<Real[_,_]> + right:Expression<LLT>) ->
    Expression<Real[_,_]> {
  return left + matrix(right);
}

/**
 * Lazy matrix addition.
 */
operator (left:Real[_,_] + right:Expression<LLT>) -> Expression<Real[_,_]> {
  return left + matrix(right);
}

/**
 * Lazy matrix addition.
 */
operator (left:Expression<Real[_,_]> + right:LLT) -> Expression<Real[_,_]> {
  return left + matrix(right);
}

/**
 * Lazy matrix addition.
 */
operator (left:Expression<LLT> + right:Expression<LLT>) ->
    Expression<Real[_,_]> {
  return matrix(left) + matrix(right);
}

/**
 * Lazy matrix addition.
 */
operator (left:LLT + right:Expression<LLT>) -> Expression<Real[_,_]> {
  return matrix(left) + matrix(right);
}

/**
 * Lazy matrix addition.
 */
operator (left:Expression<LLT> + right:LLT) -> Expression<Real[_,_]> {
  return matrix(left) + matrix(right);
}
