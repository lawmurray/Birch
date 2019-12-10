/**
 * Lazy matrix addition.
 */
final class MatrixAdd<Left,Right,Value>(left:Expression<Left>,
    right:Expression<Right>) < BinaryExpression<Left,Right,Value>(left, right) {  
  function rows() -> Integer {
    assert left.rows() == right.rows();
    return left.rows();
  }
  
  function columns() -> Integer {
    assert left.rows() == right.rows();
    return left.columns();
  }
    
  function doValue(l:Left, r:Right) -> Value {
    return l + r;
  }

  function doGradient(d:Value, l:Left, r:Right) -> (Left, Right) {
    return (d, d);
  }

  function graftLinearMatrixGaussian() ->
      TransformLinearMatrix<DelayMatrixGaussian>? {
    y:TransformLinearMatrix<DelayMatrixGaussian>?;
    z:DelayMatrixGaussian?;

    if (y <- left.graftLinearMatrixGaussian())? {
      y!.add(right);
    } else if (y <- right.graftLinearMatrixGaussian())? {
      y!.add(left);
    } else if (z <- left.graftMatrixGaussian())? {
      y <- TransformLinearMatrix<DelayMatrixGaussian>(
          Boxed(identity(z!.rows())), z!, right);
    } else if (z <- right.graftMatrixGaussian())? {
      y <- TransformLinearMatrix<DelayMatrixGaussian>(
          Boxed(identity(z!.rows())), z!, left);
    }
    return y;
  }
  
  function graftLinearMatrixNormalInverseGamma() ->
      TransformLinearMatrix<DelayMatrixNormalInverseGamma>? {
    y:TransformLinearMatrix<DelayMatrixNormalInverseGamma>?;
    z:DelayMatrixNormalInverseGamma?;

    if (y <- left.graftLinearMatrixNormalInverseGamma())? {
      y!.add(right);
    } else if (y <- right.graftLinearMatrixNormalInverseGamma())? {
      y!.add(left);
    } else if (z <- left.graftMatrixNormalInverseGamma())? {
      y <- TransformLinearMatrix<DelayMatrixNormalInverseGamma>(
          Boxed(identity(z!.rows())), z!, right);
    } else if (z <- right.graftMatrixNormalInverseGamma())? {
      y <- TransformLinearMatrix<DelayMatrixNormalInverseGamma>(
          Boxed(identity(z!.rows())), z!, left);
    }
    return y;
  }

  function graftLinearMatrixNormalInverseWishart() ->
      TransformLinearMatrix<DelayMatrixNormalInverseWishart>? {
    y:TransformLinearMatrix<DelayMatrixNormalInverseWishart>?;
    z:DelayMatrixNormalInverseWishart?;

    if (y <- left.graftLinearMatrixNormalInverseWishart())? {
      y!.add(right);
    } else if (y <- right.graftLinearMatrixNormalInverseWishart())? {
      y!.add(left);
    } else if (z <- left.graftMatrixNormalInverseWishart())? {
      y <- TransformLinearMatrix<DelayMatrixNormalInverseWishart>(
          Boxed(identity(z!.rows())), z!, right);
    } else if (z <- right.graftMatrixNormalInverseWishart())? {
      y <- TransformLinearMatrix<DelayMatrixNormalInverseWishart>(
          Boxed(identity(z!.rows())), z!, left);
    }
    return y;
  }
}

/**
 * Lazy matrix addition.
 */
operator (left:Expression<Real[_,_]> + right:Expression<Real[_,_]>) ->
    MatrixAdd<Real[_,_],Real[_,_],Real[_,_]> {
  assert left.rows() == right.rows();
  assert left.columns() == right.columns();
  m:MatrixAdd<Real[_,_],Real[_,_],Real[_,_]>(left, right);
  return m;
}

/**
 * Lazy matrix addition.
 */
operator (left:Real[_,_] + right:Expression<Real[_,_]>) ->
    MatrixAdd<Real[_,_],Real[_,_],Real[_,_]> {
  return Boxed(left) + right;
}

/**
 * Lazy matrix addition.
 */
operator (left:Expression<Real[_,_]> + right:Real[_,_]) ->
    MatrixAdd<Real[_,_],Real[_,_],Real[_,_]> {
  return left + Boxed(right);
}
