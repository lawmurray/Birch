/**
 * Lazy matrix addition.
 */
final class MatrixAdd<Left,Right,Value>(left:Expression<Left>,
    right:Expression<Right>) < BinaryExpression<Left,Right,Value>(left, right) {  
  override function rows() -> Integer {
    assert left.rows() == right.rows();
    return left.rows();
  }
  
  override function columns() -> Integer {
    assert left.rows() == right.rows();
    return left.columns();
  }
    
  override function computeValue(l:Left, r:Right) -> Value {
    return l + r;
  }

  override function computeGrad(d:Value, l:Left, r:Right) -> (Left, Right) {
    return (d, d);
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

  override function graftLinearMatrixNormalInverseWishart(compare:Distribution<Real[_,_]>) ->
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
    m:MatrixAdd<Real[_,_],Real[_,_],Real[_,_]>(left, right);
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
