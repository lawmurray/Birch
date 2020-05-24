/**
 * Lazy matrix multiply.
 */
final class MatrixMultiply<Left,Right,Value>(left:Expression<Left>,
    right:Expression<Right>) <
    BinaryExpression<Left,Right,Value>(left, right) {  
  override function rows() -> Integer {
    return left.rows();
  }
  
  override function columns() -> Integer {
    return right.columns();
  }

  override function computeValue(l:Left, r:Right) -> Value {
    return l*r;
  }

  override function computeGrad(d:Value, l:Left, r:Right) -> (Left, Right) {
    return (d*transpose(r), transpose(l)*d);
  }

  override function graftLinearMatrixGaussian() ->
      TransformLinearMatrix<MatrixGaussian>? {
    y:TransformLinearMatrix<MatrixGaussian>?;
    z:MatrixGaussian?;
    
    if (y <- right.graftLinearMatrixGaussian())? {
      y!.leftMultiply(left);
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
      y!.leftMultiply(left);
    } else if (z <- right.graftMatrixNormalInverseGamma(compare))? {
      y <- TransformLinearMatrix<MatrixNormalInverseGamma>(matrix(left), z!);
    }
    return y;
  }

  override function graftLinearMatrixNormalInverseWishart(
      compare:Distribution<Real[_,_]>) ->
      TransformLinearMatrix<MatrixNormalInverseWishart>? {
    y:TransformLinearMatrix<MatrixNormalInverseWishart>?;
    z:MatrixNormalInverseWishart?;

    if (y <- right.graftLinearMatrixNormalInverseWishart(compare))? {
      y!.leftMultiply(left);
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
    m:MatrixMultiply<Real[_,_],Real[_,_],Real[_,_]>(left, right);
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
    return Boxed(left)*right;
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
    return left*Boxed(right);
  }
}

/**
 * Lazy matrix multiply.
 */
operator (left:Expression<Real>*right:Expression<Real[_,_]>) ->
    Expression<Real[_,_]> {
  if left.isConstant() && right.isConstant() {
    return box(matrix(left.value()*right.value()));
  } else {
    m:MatrixMultiply<Real[_,_],Real[_,_],Real[_,_]>(
        diagonal(left, right.rows()), right);
    return m;
  }
}

/**
 * Lazy matrix multiply.
 */
operator (left:Real*right:Expression<Real[_,_]>) -> Expression<Real[_,_]> {
  if right.isConstant() {
    return box(matrix(left*right.value()));
  } else {
    return Boxed(left)*right;
  }
}

/**
 * Lazy matrix multiply.
 */
operator (left:Expression<Real>*right:Real[_,_]) -> Expression<Real[_,_]> {
  if left.isConstant() {
    return box(matrix(left.value()*right));
  } else {
    return left*Boxed(right);
  }
}

/**
 * Lazy matrix multiply.
 */
operator (left:Expression<Real[_,_]>*right:Expression<Real>) ->
    Expression<Real[_,_]> {
  if left.isConstant() && right.isConstant() {
    return box(matrix(left.value()*right.value()));
  } else {
    m:MatrixMultiply<Real[_,_],Real[_,_],Real[_,_]>(left,
        diagonal(right, left.columns()));
    return m;
  }
}

/**
 * Lazy matrix multiply.
 */
operator (left:Real[_,_]*right:Expression<Real>) -> Expression<Real[_,_]> {
  if right.isConstant() {
    return box(matrix(left*right.value()));
  } else {
    return Boxed(left)*right;
  }
}

/**
 * Lazy matrix multiply.
 */
operator (left:Expression<Real[_,_]>*right:Real) -> Expression<Real[_,_]> {
  if left.isConstant() {
    return box(matrix(left.value()*right));
  } else {
    return left*Boxed(right);
  }
}
