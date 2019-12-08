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
      TransformLinearMatrix<DelayMatrixGaussian>? {
    y:TransformLinearMatrix<DelayMatrixGaussian>?;
    z:DelayMatrixGaussian?;
    
    if (y <- right.graftLinearMatrixGaussian())? {
      y!.leftMultiply(left.value());
    } else if (z <- right.graftMatrixGaussian())? {
      y <- TransformLinearMatrix<DelayMatrixGaussian>(
          left.value(), z!);
    }
    return y;
  }
  
  function graftLinearMatrixNormalInverseGamma() ->
      TransformLinearMatrix<DelayMatrixNormalInverseGamma>? {
    y:TransformLinearMatrix<DelayMatrixNormalInverseGamma>?;
    z:DelayMatrixNormalInverseGamma?;

    if (y <- right.graftLinearMatrixNormalInverseGamma())? {
      y!.leftMultiply(left.value());
    } else if (z <- right.graftMatrixNormalInverseGamma())? {
      y <- TransformLinearMatrix<DelayMatrixNormalInverseGamma>(
          left.value(), z!);
    }
    return y;
  }

  function graftLinearMatrixNormalInverseWishart() ->
      TransformLinearMatrix<DelayMatrixNormalInverseWishart>? {
    y:TransformLinearMatrix<DelayMatrixNormalInverseWishart>?;
    z:DelayMatrixNormalInverseWishart?;

    if (y <- right.graftLinearMatrixNormalInverseWishart())? {
      y!.leftMultiply(left.value());
    } else if (z <- right.graftMatrixNormalInverseWishart())? {
      y <- TransformLinearMatrix<DelayMatrixNormalInverseWishart>(
          left.value(), z!);
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
  m:MatrixMultiply<Real[_,_],Real[_,_],Real[_,_]>(diagonal(left,
      right.rows()), right);
  return m;
}

/**
 * Lazy matrix multiply.
 */
operator (left:Real*right:Expression<Real[_,_]>) ->
    MatrixMultiply<Real[_,_],Real[_,_],Real[_,_]> {
  return diagonal(Boxed(left), right.rows())*right;
}

/**
 * Lazy matrix multiply.
 */
operator (left:Expression<Real>*right:Real[_,_]) ->
    MatrixMultiply<Real[_,_],Real[_,_],Real[_,_]> {
  return diagonal(left, rows(right))*Boxed(right);
}

/**
 * Lazy matrix multiply.
 */
operator (left:Expression<Real[_,_]>*right:Expression<Real>) ->
    MatrixMultiply<Real[_,_],Real[_,_],Real[_,_]> {
  m:MatrixMultiply<Real[_,_],Real[_,_],Real[_,_]>(diagonal(right,
      left.rows()), left);
  return m;
}

/**
 * Lazy multivariate multiply.
 */
operator (left:Real[_,_]*right:Expression<Real>) ->
    MatrixMultiply<Real[_,_],Real[_,_],Real[_,_]> {
  return diagonal(right, right.rows())*Boxed(left);
}

/**
 * Lazy multivariate multiply.
 */
operator (left:Expression<Real[_,_]>*right:Real) ->
    MatrixMultiply<Real[_,_],Real[_,_],Real[_,_]> {
  return diagonal(Boxed(right), left.rows())*left;
}
