/**
 * Lazy matrix multiply.
 */
final class MatrixMultiply<Left,Right,Value>(left:Expression<Left>,
    right:Expression<Right>) < BinaryExpression<Left,Right,Value>(left, right) {  
  function rows() -> Integer {
    return left.rows();
  }
  
  function columns() -> Integer {
    return right.columns();
  }

  function doValue(l:Left, r:Right) -> Value {
    return l*r;
  }

  function doGradient(d:Value, l:Left, r:Right) -> (Left, Right) {
    return (d*transpose(r), transpose(l)*d);
  }

  function graftLinearMatrixGaussian() ->
      TransformLinearMatrix<MatrixGaussian>? {
    y:TransformLinearMatrix<MatrixGaussian>?;
    z:MatrixGaussian?;
    
    if (y <- right.graftLinearMatrixGaussian())? {
      y!.leftMultiply(left);
    } else if (z <- right.graftMatrixGaussian())? {
      y <- TransformLinearMatrix<MatrixGaussian>(left, z!);
    }
    return y;
  }
  
  function graftLinearMatrixNormalInverseGamma(compare:Distribution<Real[_]>) ->
      TransformLinearMatrix<MatrixNormalInverseGamma>? {
    y:TransformLinearMatrix<MatrixNormalInverseGamma>?;
    z:MatrixNormalInverseGamma?;

    if (y <- right.graftLinearMatrixNormalInverseGamma(compare))? {
      y!.leftMultiply(left);
    } else if (z <- right.graftMatrixNormalInverseGamma(compare))? {
      y <- TransformLinearMatrix<MatrixNormalInverseGamma>(left, z!);
    }
    return y;
  }

  function graftLinearMatrixNormalInverseWishart(compare:Distribution<Real[_,_]>) ->
      TransformLinearMatrix<MatrixNormalInverseWishart>? {
    y:TransformLinearMatrix<MatrixNormalInverseWishart>?;
    z:MatrixNormalInverseWishart?;

    if (y <- right.graftLinearMatrixNormalInverseWishart(compare))? {
      y!.leftMultiply(left);
    } else if (z <- right.graftMatrixNormalInverseWishart(compare))? {
      y <- TransformLinearMatrix<MatrixNormalInverseWishart>(left, z!);
    }
    return y;
  }
}

/**
 * Lazy matrix multiply.
 */
operator (left:Expression<Real[_,_]>*right:Expression<Real[_,_]>) ->
    MatrixMultiply<Real[_,_],Real[_,_],Real[_,_]> {
  assert left.columns() == right.rows();
  m:MatrixMultiply<Real[_,_],Real[_,_],Real[_,_]>(left, right);
  return m;
}

/**
 * Lazy matrix multiply.
 */
operator (left:Real[_,_]*right:Expression<Real[_,_]>) ->
    MatrixMultiply<Real[_,_],Real[_,_],Real[_,_]> {
  return Boxed(left)*right;
}

/**
 * Lazy matrix multiply.
 */
operator (left:Expression<Real[_,_]>*right:Real[_,_]) ->
    MatrixMultiply<Real[_,_],Real[_,_],Real[_,_]> {
  return left*Boxed(right);
}

/**
 * Lazy matrix multiply.
 */
operator (left:Expression<Real>*right:Expression<Real[_,_]>) ->
    MatrixMultiply<Real[_,_],Real[_,_],Real[_,_]> {
  return diagonal(left, right.rows())*right;
}

/**
 * Lazy matrix multiply.
 */
operator (left:Real*right:Expression<Real[_,_]>) ->
    MatrixMultiply<Real[_,_],Real[_,_],Real[_,_]> {
  return Boxed(left)*right;
}

/**
 * Lazy matrix multiply.
 */
operator (left:Expression<Real>*right:Real[_,_]) ->
    MatrixMultiply<Real[_,_],Real[_,_],Real[_,_]> {
  return left*Boxed(right);
}

/**
 * Lazy matrix multiply.
 */
operator (left:Expression<Real[_,_]>*right:Expression<Real>) ->
    MatrixMultiply<Real[_,_],Real[_,_],Real[_,_]> {
  return left*diagonal(right, left.columns());
}

/**
 * Lazy matrix multiply.
 */
operator (left:Real[_,_]*right:Expression<Real>) ->
    MatrixMultiply<Real[_,_],Real[_,_],Real[_,_]> {
  return Boxed(left)*right;
}

/**
 * Lazy matrix multiply.
 */
operator (left:Expression<Real[_,_]>*right:Real) ->
    MatrixMultiply<Real[_,_],Real[_,_],Real[_,_]> {
  return left*Boxed(right);
}
