/**
 * Lazy matrix multiply.
 */
final class MatrixDivide<Left,Right,Value>(left:Expression<Left>,
    right:Expression<Right>) < BinaryExpression<Left,Right,Value>(left, right) {  
  function rows() -> Integer {
    return left.rows();
  }
  
  function columns() -> Integer {
    return right.columns();
  }

  function doValue(l:Left, r:Right) -> Value {
    return l/r;
  }

  function doGradient(d:Value, l:Left, r:Right) -> (Left, Right) {
    return (d/r, transpose(l)*(-d/(r*r)));
  }

  function graftLinearMatrixGaussian() ->
      TransformLinearMatrix<MatrixGaussian>? {
    y:TransformLinearMatrix<MatrixGaussian>?;
    z:MatrixGaussian?;
    
    if (y <- left.graftLinearMatrixGaussian())? {
      y!.divide(right);
    } else if (z <- left.graftMatrixGaussian())? {
      y <- TransformLinearMatrix<MatrixGaussian>(
          diagonal(1.0/right, z!.rows()), z!);
    }
    return y;
  }
  
  function graftLinearMatrixNormalInverseGamma() ->
      TransformLinearMatrix<MatrixNormalInverseGamma>? {
    y:TransformLinearMatrix<MatrixNormalInverseGamma>?;
    z:MatrixNormalInverseGamma?;

    if (y <- left.graftLinearMatrixNormalInverseGamma())? {
      y!.divide(right);
    } else if (z <- left.graftMatrixNormalInverseGamma())? {
      y <- TransformLinearMatrix<MatrixNormalInverseGamma>(
          diagonal(1.0/right, z!.rows()), z!);
    }
    return y;
  }

  function graftLinearMatrixNormalInverseWishart() ->
      TransformLinearMatrix<MatrixNormalInverseWishart>? {
    y:TransformLinearMatrix<MatrixNormalInverseWishart>?;
    z:MatrixNormalInverseWishart?;

    if (y <- left.graftLinearMatrixNormalInverseWishart())? {
      y!.divide(right);
    } else if (z <- right.graftMatrixNormalInverseWishart())? {
      y <- TransformLinearMatrix<MatrixNormalInverseWishart>(
          diagonal(1.0/right, z!.rows()), z!);
    }
    return y;
  }
}

/**
 * Lazy matrix divide.
 */
operator (left:Expression<Real[_,_]>/right:Expression<Real>) ->
    MatrixDivide<Real[_,_],Real,Real[_,_]> {
  m:MatrixDivide<Real[_,_],Real,Real[_,_]>(left, right);
  return m;
}

/**
 * Lazy matrix divide.
 */
operator (left:Real[_,_]/right:Expression<Real>) ->
    MatrixDivide<Real[_,_],Real,Real[_,_]> {
  return Boxed(left)/right;
}

/**
 * Lazy matrix divide.
 */
operator (left:Expression<Real[_,_]>/right:Real) ->
    MatrixDivide<Real[_,_],Real,Real[_,_]> {
  return left/Boxed(right);
}
