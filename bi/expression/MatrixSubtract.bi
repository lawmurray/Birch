/**
 * Lazy matrix subtract.
 */
final class MatrixSubtract<Left,Right,Value>(left:Left, right:Right) <
    MatrixBinaryExpression<Left,Right,Value>(left, right) {  
  override function doRows() -> Integer {
    return left!.rows();
  }
  
  override function doColumns() -> Integer {
    return left!.columns();
  }

  override function doValue() {
    x <- left!.value() - right!.value();
  }

  override function doPilot() {
    x <- left!.pilot() - right!.pilot();
  }

  override function doMove(κ:Kernel) {
    x <- left!.move(κ) - right!.move(κ);
  }

  override function doGrad() {
    left!.grad(d!);
    right!.grad(-d!);
  }

  override function graftLinearMatrixGaussian() ->
      TransformLinearMatrix<MatrixGaussian>? {
    y:TransformLinearMatrix<MatrixGaussian>?;
    z:MatrixGaussian?;

    if (y <- left!.graftLinearMatrixGaussian())? {
      y!.subtract(right!);
    } else if (y <- right!.graftLinearMatrixGaussian())? {
      y!.negateAndAdd(left!);
    } else if (z <- left!.graftMatrixGaussian())? {
      y <- TransformLinearMatrix<MatrixGaussian>(box(identity(z!.rows())), z!, -right!);
    } else if (z <- right!.graftMatrixGaussian())? {
      y <- TransformLinearMatrix<MatrixGaussian>(box(diagonal(-1.0, z!.rows())), z!, left!);
    }
    return y;
  }
  
  override function graftLinearMatrixNormalInverseGamma(compare:Distribution<Real[_]>) ->
      TransformLinearMatrix<MatrixNormalInverseGamma>? {
    y:TransformLinearMatrix<MatrixNormalInverseGamma>?;
    z:MatrixNormalInverseGamma?;

    if (y <- left!.graftLinearMatrixNormalInverseGamma(compare))? {
      y!.subtract(right!);
    } else if (y <- right!.graftLinearMatrixNormalInverseGamma(compare))? {
      y!.negateAndAdd(left!);
    } else if (z <- left!.graftMatrixNormalInverseGamma(compare))? {
      y <- TransformLinearMatrix<MatrixNormalInverseGamma>(box(identity(z!.rows())), z!, -right!);
    } else if (z <- right!.graftMatrixNormalInverseGamma(compare))? {
      y <- TransformLinearMatrix<MatrixNormalInverseGamma>(box(diagonal(-1.0, z!.rows())), z!, left!);
    }
    return y;
  }

  override function graftLinearMatrixNormalInverseWishart(compare:Distribution<LLT>) ->
      TransformLinearMatrix<MatrixNormalInverseWishart>? {
    y:TransformLinearMatrix<MatrixNormalInverseWishart>?;
    z:MatrixNormalInverseWishart?;

    if (y <- left!.graftLinearMatrixNormalInverseWishart(compare))? {
      y!.subtract(right!);
    } else if (y <- right!.graftLinearMatrixNormalInverseWishart(compare))? {
      y!.negateAndAdd(left!);
    } else if (z <- left!.graftMatrixNormalInverseWishart(compare))? {
      y <- TransformLinearMatrix<MatrixNormalInverseWishart>(box(identity(z!.rows())), z!, -right!);
    } else if (z <- right!.graftMatrixNormalInverseWishart(compare))? {
      y <- TransformLinearMatrix<MatrixNormalInverseWishart>(box(diagonal(-1.0, z!.rows())), z!, left!);
    }
    return y;
  }
}

/**
 * Lazy matrix subtract.
 */
operator (left:Expression<Real[_,_]> - right:Expression<Real[_,_]>) ->
    Expression<Real[_,_]> {
  assert left.rows() == right.rows();
  assert left.columns() == right.columns();
  if left.isConstant() && right.isConstant() {
    return box(matrix(left.value() - right.value()));
  } else {
    m:MatrixSubtract<Expression<Real[_,_]>,Expression<Real[_,_]>,Real[_,_]>(left, right);
    return m;
  }
}

/**
 * Lazy matrix subtract.
 */
operator (left:Real[_,_] - right:Expression<Real[_,_]>) ->
    Expression<Real[_,_]> {
  if right.isConstant() {
    return box(matrix(left - right.value()));
  } else {
    return box(left) - right;
  }
}

/**
 * Lazy matrix subtract.
 */
operator (left:Expression<Real[_,_]> - right:Real[_,_]) ->
    Expression<Real[_,_]> {
  if left.isConstant() {
    return box(matrix(left.value() - right));
  } else {
    return left - box(right);
  }
}

/**
 * Lazy matrix subtract.
 */
operator (left:Expression<LLT> - right:Expression<Real[_,_]>) ->
    Expression<Real[_,_]> {
  return matrix(left) - right;
}

/**
 * Lazy matrix subtract.
 */
operator (left:LLT - right:Expression<Real[_,_]>) -> Expression<Real[_,_]> {
  return matrix(left) - right;
}

/**
 * Lazy matrix subtract.
 */
operator (left:Expression<LLT> - right:Real[_,_]) -> Expression<Real[_,_]> {
  return matrix(left) - right;
}

/**
 * Lazy matrix subtract.
 */
operator (left:Expression<Real[_,_]> - right:Expression<LLT>) ->
    Expression<Real[_,_]> {
  return left - matrix(right);
}

/**
 * Lazy matrix subtract.
 */
operator (left:Real[_,_] - right:Expression<LLT>) -> Expression<Real[_,_]> {
  return left - matrix(right);
}

/**
 * Lazy matrix subtract.
 */
operator (left:Expression<Real[_,_]> - right:LLT) -> Expression<Real[_,_]> {
  return left - matrix(right);
}

/**
 * Lazy matrix subtract.
 */
operator (left:Expression<LLT> - right:Expression<LLT>) ->
    Expression<Real[_,_]> {
  return matrix(left) - matrix(right);
}

/**
 * Lazy matrix subtract.
 */
operator (left:LLT - right:Expression<LLT>) -> Expression<Real[_,_]> {
  return matrix(left) - matrix(right);
}

/**
 * Lazy matrix subtract.
 */
operator (left:Expression<LLT> - right:LLT) -> Expression<Real[_,_]> {
  return matrix(left) - matrix(right);
}
